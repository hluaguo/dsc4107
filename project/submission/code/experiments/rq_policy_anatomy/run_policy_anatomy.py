"""RQ-E: anatomy of RQ-D α-selected deep policies vs Leland/Imaki-style bands.

On the frozen Heston ATM 1M cell we retrain (or reload selection metadata from
RQ-D JSON when present) with the same OOS α-selection as ``run_risk_matched``,
then compare position paths to ``BandDelta(0.02)`` and calendar ``TimeBased``
on the shared validation pool. Cost and CVaR stay on ``hedge.rollout``; anatomy
metrics are computed only in this script from replayed positions.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import bs, hedge, metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import (  # noqa: E402
    _ArrayPositionFn,
    resolve_device,
    strategy_spec,
)
from experiments.rq_risk_matched.run_risk_matched import (  # noqa: E402
    FULL_ALPHAS,
    QUICK_ALPHAS,
    select_risk_matched,
    train_single_net,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"
RQ_D_RESULTS = ROOT / "experiments" / "rq_risk_matched" / "results"

# Lit bars compare deep paths to this band; ε for no-trade mass matches its width.
BAND_WIDTH = 0.02
NO_TRADE_EPS = BAND_WIDTH
MEAN_DD_MULT = 1.25
NO_TRADE_RATIO = 0.80

# RQ-A grid includes daily rebalance on 21 steps; TimeBased is context only.
TIME_COMPARATOR = {
    "label": "time_f1",
    "strategy": "time",
    "frequency": 1,
    "delta_band": None,
    "price_band": None,
}

BAND_COMPARATOR = {
    "label": "band_delta_0.02",
    "strategy": "band",
    "frequency": 1,
    "delta_band": BAND_WIDTH,
    "price_band": None,
}


def load_val_paths(cfg: B.BenchConfig) -> list[tuple[int, np.ndarray]]:
    """One simulate per val seed so anatomy passes do not regenerate paths."""
    return [
        (int(seed), cfg.simulate(cfg.model, seed, cfg.n_val_paths))
        for seed in cfg.val_seeds
    ]


def pooled_rollout_from_paths(
    cfg: B.BenchConfig, val_paths: list[tuple[int, np.ndarray]], spec: dict
) -> dict:
    cost_chunks: list[np.ndarray] = []
    pnl_chunks: list[np.ndarray] = []
    for _seed, paths in val_paths:
        fn = B.make_classical_position_fn(cfg, spec)
        result = hedge.rollout(
            paths,
            fn,
            cfg.K,
            cfg.kind,
            cfg.premium,
            cfg.cost,
            cfg.dt,
        )
        cost_chunks.append(result.costs)
        pnl_chunks.append(result.pnl)
    costs = np.concatenate(cost_chunks)
    pnl = np.concatenate(pnl_chunks)
    return {
        "mean_cost": float(costs.mean()),
        "cvar95": metrics.cvar95(pnl),
        "pnl_mean": float(pnl.mean()),
    }


def pooled_deep_rollout_from_paths(
    cfg: B.BenchConfig,
    val_paths: list[tuple[int, np.ndarray]],
    nets,
    device: str,
) -> dict:
    from hedging import deep

    cost_chunks: list[np.ndarray] = []
    pnl_chunks: list[np.ndarray] = []
    for _seed, paths in val_paths:
        member_pnls: list[np.ndarray] = []
        member_costs: list[np.ndarray] = []
        for net in nets:
            tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
            targets = deep.hedge_positions(tensor, net, cfg.K, cfg.dt, cfg.T)
            result = hedge.rollout(
                paths,
                _ArrayPositionFn(targets),
                cfg.K,
                cfg.kind,
                cfg.premium,
                cfg.cost,
                cfg.dt,
            )
            member_pnls.append(result.pnl)
            member_costs.append(result.costs)
        if len(nets) == 1:
            pnl = member_pnls[0]
            costs = member_costs[0]
        else:
            pnl = np.mean(member_pnls, axis=0)
            costs = np.mean(member_costs, axis=0)
        pnl_chunks.append(pnl)
        cost_chunks.append(costs)
    costs = np.concatenate(cost_chunks)
    pnl = np.concatenate(pnl_chunks)
    return {
        "mean_cost": float(costs.mean()),
        "cvar95": metrics.cvar95(pnl),
        "pnl_mean": float(pnl.mean()),
    }


def load_rq_d_selection_hints(quick: bool) -> dict[int, float] | None:
    """Reuse chosen training-α per torch seed when RQ-D artifacts exist (no weight checkpoints)."""
    path = RQ_D_RESULTS / ("risk_matched_quick.json" if quick else "risk_matched.json")
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    hints: dict[int, float] = {}
    for row in payload.get("deep", []):
        if row.get("kind") != "seed":
            continue
        hints[int(row["torch_seed"])] = float(row["alpha"])
    return hints or None


def collect_positions(
    paths: np.ndarray, position_fn: hedge.PositionFn
) -> np.ndarray:
    """Positions after each decision step, shape (n_paths, n_steps)."""
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    pos = np.zeros(n_paths)
    cols: list[np.ndarray] = []
    for j in range(n_steps):
        spot = paths[:, j]
        pos = np.asarray(position_fn(j, spot, pos), dtype=float)
        cols.append(pos.copy())
    return np.stack(cols, axis=1)


def bs_delta_path(paths: np.ndarray, cfg: B.BenchConfig) -> np.ndarray:
    """Frictionless BS delta at each decision spot — band occupancy reference."""
    n_steps = paths.shape[1] - 1
    out = np.empty((paths.shape[0], n_steps))
    for j in range(n_steps):
        tau = max(cfg.T - j * cfg.dt, 0.0)
        out[:, j] = bs.bs_delta(
            paths[:, j], cfg.K, tau, cfg.r, cfg.sigma_impl, cfg.kind
        )
    return out


def anatomy_from_positions(positions: np.ndarray, band_width: float) -> dict:
    """Path-pooled anatomy; no-trade mass uses the same ε as the band comparator."""
    n_paths, n_steps = positions.shape
    prev = np.zeros((n_paths, 1))
    delta = np.concatenate([positions[:, :1] - prev, positions[:, 1:] - positions[:, :-1]], axis=1)
    abs_delta = np.abs(delta)
    flat_delta = abs_delta.ravel()
    no_trade = float((abs_delta <= band_width).mean())
    trade_rate = float((abs_delta > 1e-12).mean())
    mean_abs_delta = float(flat_delta.mean())
    mean_abs_delta_when_trade = float(flat_delta[flat_delta > 1e-12].mean()) if np.any(flat_delta > 1e-12) else 0.0
    # Turnover timing: fraction of |Δδ| mass at each step index (pooled over paths).
    step_mass = abs_delta.sum(axis=0)
    total = step_mass.sum()
    if total > 0:
        timing = (step_mass / total).tolist()
    else:
        timing = (np.zeros(n_steps) / max(n_steps, 1)).tolist()
    return {
        "no_trade_mass": no_trade,
        "trade_rate": trade_rate,
        "mean_abs_delta": mean_abs_delta,
        "mean_abs_delta_when_trading": mean_abs_delta_when_trade,
        "turnover_timing": timing,
    }


def time_in_band(positions: np.ndarray, delta_bs: np.ndarray, band_width: float) -> float:
    """Imaki/Leland-style occupancy: position within ±band of frictionless delta."""
    return float((np.abs(positions - delta_bs) <= band_width).mean())


def pooled_anatomy(
    cfg: B.BenchConfig,
    *,
    val_paths: list[tuple[int, np.ndarray]] | None = None,
    spec: dict | None = None,
    position_fn: hedge.PositionFn | None = None,
    nets: list | None = None,
    device: str = "cpu",
) -> dict:
    """Aggregate anatomy over the same val_seeds × n_val_paths as RQ-A/RQ-D."""
    from hedging import deep

    chunks: list[dict] = []
    band_occ_chunks: list[float] = []
    batches = val_paths
    if batches is None:
        batches = load_val_paths(cfg)
    for _seed, paths in batches:
        delta_bs = bs_delta_path(paths, cfg)
        if nets is not None:
            tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
            member_pos = [
                deep.hedge_positions(tensor, net, cfg.K, cfg.dt, cfg.T) for net in nets
            ]
            if len(nets) == 1:
                positions = member_pos[0]
            else:
                positions = np.mean(member_pos, axis=0)
        elif position_fn is not None:
            positions = collect_positions(paths, position_fn)
        elif spec is not None:
            fn = B.make_classical_position_fn(cfg, spec)
            positions = collect_positions(paths, fn)
        else:
            raise ValueError("need spec, position_fn, or nets")
        chunks.append(anatomy_from_positions(positions, NO_TRADE_EPS))
        band_occ_chunks.append(time_in_band(positions, delta_bs, BAND_WIDTH))
    # Pooled means weighted by path count per seed batch.
    weights = np.array([cfg.n_val_paths] * len(batches), dtype=float)
    weights /= weights.sum()

    def wmean(key: str) -> float:
        return float(sum(w * c[key] for w, c in zip(weights, chunks)))

    timing = np.zeros(len(chunks[0]["turnover_timing"]))
    for w, c in zip(weights, chunks):
        timing += w * np.asarray(c["turnover_timing"])
    return {
        "no_trade_mass": wmean("no_trade_mass"),
        "trade_rate": wmean("trade_rate"),
        "mean_abs_delta": wmean("mean_abs_delta"),
        "mean_abs_delta_when_trading": wmean("mean_abs_delta_when_trading"),
        "time_in_band_vs_bs_delta": float(np.mean(band_occ_chunks)),
        "turnover_timing": timing.tolist(),
    }


def classify_seed(
    seed_row: dict,
    band_anatomy: dict,
    classical_cvar: float,
) -> str:
    """Lit acceptance bars vs ``BandDelta(0.02)`` for one α-selected seed."""
    if seed_row["cvar95"] > classical_cvar:
        return "inconclusive"
    band_nt = band_anatomy["no_trade_mass"]
    band_mad = band_anatomy["mean_abs_delta"]
    nt = seed_row["no_trade_mass"]
    mad = seed_row["mean_abs_delta"]
    band_like = nt >= NO_TRADE_RATIO * band_nt and mad <= MEAN_DD_MULT * band_mad
    if band_like:
        return "band_like"
    return "distinct"


def acceptance_verdict(
    seed_rows: list[dict],
    band_anatomy: dict,
    classical_cvar: float,
) -> str:
    """Headline verdict from CVaR-better seeds only; else inconclusive."""
    cvar_better = [r for r in seed_rows if r["cvar95"] <= classical_cvar]
    if not cvar_better:
        return "inconclusive"
    labels = [classify_seed(r, band_anatomy, classical_cvar) for r in cvar_better]
    if all(l == "band_like" for l in labels):
        return "band_like"
    if all(l == "distinct" for l in labels):
        return "distinct"
    return "inconclusive"


def train_and_select_nets(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    alphas: tuple[float, ...],
    alpha_hints: dict[int, float] | None,
    target_cvar: float,
    target_cost: float,
    val_paths: list[tuple[int, np.ndarray]],
) -> tuple[list, list[dict]]:
    """Same α grid + OOS selection as RQ-D; optional hints restrict retraining."""

    selected_nets: list = []
    seed_rows: list[dict] = []
    for torch_seed in hyper.torch_seeds:
        sweep_rows: list[dict] = []
        nets_by_alpha: dict[float, torch.nn.Module] = {}
        train_alphas = alphas
        if alpha_hints and int(torch_seed) in alpha_hints:
            hint = float(alpha_hints[int(torch_seed)])
            train_alphas = tuple(a for a in alphas if abs(a - hint) < 1e-9) or (hint,)
        for alpha in train_alphas:
            run_hyper = replace(hyper, alpha=float(alpha))
            net, _ = train_single_net(cfg, run_hyper, int(torch_seed), device)
            nets_by_alpha[float(alpha)] = net
            stats = pooled_deep_rollout_from_paths(cfg, val_paths, [net], device)
            sweep_rows.append(
                {
                    "alpha": float(alpha),
                    **stats,
                }
            )
        if len(sweep_rows) == 1:
            chosen = {**sweep_rows[0], "alpha": sweep_rows[0]["alpha"]}
        else:
            chosen = select_risk_matched(
                [
                    {
                        "mean_cost": r["mean_cost"],
                        "cvar95": r["cvar95"],
                        "pnl_mean": r["pnl_mean"],
                        "alpha": r["alpha"],
                    }
                    for r in sweep_rows
                ],
                target_cvar,
                target_cost,
            )
        selected_nets.append(nets_by_alpha[float(chosen["alpha"])])
        seed_rows.append(
            {
                "torch_seed": int(torch_seed),
                "alpha": float(chosen["alpha"]),
                "mean_cost": chosen["mean_cost"],
                "cvar95": chosen["cvar95"],
                "pnl_mean": chosen.get("pnl_mean", 0.0),
            }
        )
    return selected_nets, seed_rows


def save_plot(band_row: dict, seed_rows: list[dict], path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.bar(
        ["band_0.02"]
        + [f"seed_{r['torch_seed']}" for r in seed_rows],
        [band_row["no_trade_mass"]] + [r["no_trade_mass"] for r in seed_rows],
        color="tab:orange",
    )
    ax.set_ylabel("no-trade mass (|Δδ| ≤ 0.02)")
    ax.set_title("No-trade mass vs BandDelta(0.02)")
    ax2 = axes[1]
    ax2.bar(
        ["band_0.02"]
        + [f"seed_{r['torch_seed']}" for r in seed_rows],
        [band_row["mean_abs_delta"]] + [r["mean_abs_delta"] for r in seed_rows],
        color="tab:blue",
    )
    ax2.set_ylabel("mean |Δδ|")
    ax2.set_title("Mean adjustment size")
    fig.suptitle("RQ-E: policy anatomy")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="smoke-sized pools and training")
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for deep training (auto, cpu, cuda)",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    cfg = B.BenchConfig()
    hyper = Hyper()
    alphas = QUICK_ALPHAS if args.quick else FULL_ALPHAS
    if args.quick:
        cfg, hyper = B._quick_setup(cfg, hyper)

    started = time.time()
    alpha_hints = load_rq_d_selection_hints(args.quick)

    val_paths = load_val_paths(cfg)

    band_spec = strategy_spec(BAND_COMPARATOR)
    time_spec = strategy_spec(TIME_COMPARATOR)
    band_rollout = pooled_rollout_from_paths(cfg, val_paths, band_spec)
    time_rollout = pooled_rollout_from_paths(cfg, val_paths, time_spec)
    best_spec = strategy_spec({**B.load_classical_best(), "label": "classical_best"})
    best_rollout = pooled_rollout_from_paths(cfg, val_paths, best_spec)
    classical_cvar = best_rollout["cvar95"]

    band_anatomy = pooled_anatomy(cfg, val_paths=val_paths, spec=BAND_COMPARATOR)
    time_anatomy = pooled_anatomy(cfg, val_paths=val_paths, spec=TIME_COMPARATOR)
    band_row = {**BAND_COMPARATOR, **band_rollout, **band_anatomy, "kind": "comparator"}
    time_row = {**TIME_COMPARATOR, **time_rollout, **time_anatomy, "kind": "reference"}

    selected_nets, rollout_seeds = train_and_select_nets(
        cfg,
        hyper,
        device,
        alphas,
        alpha_hints,
        classical_cvar,
        best_rollout["mean_cost"],
        val_paths,
    )

    seed_rows: list[dict] = []
    for net, base in zip(selected_nets, rollout_seeds):
        anatomy = pooled_anatomy(cfg, val_paths=val_paths, nets=[net], device=device)
        rollout = pooled_deep_rollout_from_paths(cfg, val_paths, [net], device)
        row = {
            "label": f"deep_seed_{base['torch_seed']}_selected",
            "kind": "seed",
            **base,
            **rollout,
            **anatomy,
        }
        row["seed_verdict"] = classify_seed(row, band_anatomy, classical_cvar)
        seed_rows.append(row)

    ensemble_rollout = pooled_deep_rollout_from_paths(
        cfg, val_paths, selected_nets, device
    )
    ensemble_anatomy = pooled_anatomy(
        cfg, val_paths=val_paths, nets=selected_nets, device=device
    )
    ensemble_row = {
        "label": "deep_ensemble_selected",
        "kind": "ensemble",
        "torch_seeds": list(hyper.torch_seeds),
        **ensemble_rollout,
        **ensemble_anatomy,
    }

    verdict = acceptance_verdict(seed_rows, band_anatomy, classical_cvar)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "research_question": "rq_e_policy_anatomy",
        "acceptance": {
            "band_like": verdict == "band_like",
            "distinct": verdict == "distinct",
            "inconclusive": verdict == "inconclusive",
            "bars": {
                "no_trade_mass_ratio_min": NO_TRADE_RATIO,
                "mean_abs_delta_ratio_max": MEAN_DD_MULT,
                "no_trade_epsilon": NO_TRADE_EPS,
                "comparator_band_width": BAND_WIDTH,
                "requires_cvar95_lte_classical_best": True,
            },
            "classical_best": best_rollout,
        },
        "bench": {
            "name": cfg.name,
            "model": cfg.model,
            "cost": cfg.cost,
            "n_steps": cfg.n_steps,
            "val_seeds": list(cfg.val_seeds),
            "n_val_paths": cfg.n_val_paths,
            "quick": args.quick,
        },
        "hyper": asdict(hyper),
        "alpha_grid": list(alphas),
        "rq_d_alpha_hints_used": alpha_hints is not None,
        "comparators": {
            "band_delta_0.02": band_row,
            "time_f1": time_row,
        },
        "deep": seed_rows + [ensemble_row],
        "wall_time_s": time.time() - started,
    }
    json_name = "policy_anatomy_quick.json" if args.quick else "policy_anatomy.json"
    json_path = OUT / json_name
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("policy_anatomy_quick.png" if args.quick else "policy_anatomy.png")
        save_plot(band_row, seed_rows, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
