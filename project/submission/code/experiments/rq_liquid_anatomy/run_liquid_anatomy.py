"""RQ-J: policy anatomy of fixed liquid-10bp deep recipes vs the re-swept band champion.

Two training recipes only (no OOS α sweep): CVaR loss at ``alpha=0.05`` and
Lagrange excess-CVaR at ``Hyper().alpha`` (0.65). Anatomy Lit bars match RQ-E
(0.8× no-trade mass, 1.25× mean |Δδ|) against the val-Sharpe **band** champion
at ``cost=0.001``; ε for no-trade mass is the champion band width, not Leland ε→0.
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

from hedging import bench as B  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import resolve_device, strategy_spec  # noqa: E402
from experiments.rq_cost_stress.run_cost_stress import (  # noqa: E402
    classical_grid_only,
    mean_val_sharpe,
    pick_champion,
)
from experiments.rq_hard_cvar.run_hard_cvar import train_seed_net as train_lagrange_net
from experiments.rq_policy_anatomy.run_policy_anatomy import (  # noqa: E402
    acceptance_verdict,
    classify_seed,
    load_val_paths,
    pooled_anatomy,
    pooled_deep_rollout_from_paths,
    pooled_rollout_from_paths,
)
from experiments.rq_risk_matched.run_risk_matched import train_single_net  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "results"

LIQUID_COST = 0.001
RM_FIXED_ALPHA = 0.05
LAGRANGE_ALPHA = 0.65
# Fixed expanded train seeds (aligned with planned RQ-I; beyond RQ-H’s {0,1,2}).
EXPANDED_TORCH_SEEDS = (0, 1, 2, 3, 4, 5)

NO_TRADE_RATIO = 0.80
MEAN_DD_MULT = 1.25


def sweep_classical_hull(
    cfg: B.BenchConfig, val_paths: list[tuple[int, np.ndarray]]
) -> tuple[list[dict], dict]:
    classical_rows: list[dict] = []
    for cand in classical_grid_only():
        spec = strategy_spec(cand)
        stats = pooled_rollout_from_paths(cfg, val_paths, spec)
        sharpe = mean_val_sharpe(cfg, spec)
        classical_rows.append(
            {
                "label": cand["label"],
                "kind": "classical",
                "spec": spec,
                "strategy": cand["strategy"],
                "delta_band": cand.get("delta_band"),
                "price_band": cand.get("price_band"),
                "mean_sharpe": sharpe,
                **stats,
            }
        )
    champion_row = pick_champion(classical_rows)
    champion = {
        "label": champion_row["label"],
        "spec": champion_row["spec"],
        "strategy": champion_row.get("strategy"),
        "delta_band": champion_row.get("delta_band"),
        "price_band": champion_row.get("price_band"),
        "mean_sharpe": champion_row["mean_sharpe"],
        "mean_cost": champion_row["mean_cost"],
        "cvar95": champion_row["cvar95"],
    }
    return classical_rows, champion


def band_comparator_from_champion(champion: dict) -> tuple[dict, float]:
    """Primary anatomy comparator: champion if it is a band, else δ-band matching family."""
    label = champion["label"]
    spec = champion["spec"]
    if champion.get("strategy") == "band" and champion.get("delta_band") is not None:
        width = float(champion["delta_band"])
        return (
            {
                "label": label,
                "strategy": "band",
                "frequency": 1,
                "delta_band": width,
                "price_band": None,
                "role": "champion_band",
            },
            width,
        )
    if champion.get("strategy") == "band" and champion.get("price_band") is not None:
        width = float(champion["price_band"])
        return (
            {
                "label": f"{label}_anatomy_proxy",
                "strategy": "band",
                "frequency": 1,
                "delta_band": width,
                "price_band": None,
                "role": "price_champion_proxy_delta_band",
            },
            width,
        )
    width = 0.05
    return (
        {
            "label": "band_delta_0.05",
            "strategy": "band",
            "frequency": 1,
            "delta_band": width,
            "price_band": None,
            "role": "fallback_band_delta",
        },
        width,
    )


def pooled_anatomy_with_band_width(
    cfg: B.BenchConfig,
    band_width: float,
    *,
    val_paths,
    spec=None,
    nets=None,
    device: str = "cpu",
):
    """RQ-E pooling with ε = champion band width (imported helper uses 0.02)."""
    import experiments.rq_policy_anatomy.run_policy_anatomy as pa

    old_eps = pa.NO_TRADE_EPS
    old_bw = pa.BAND_WIDTH
    pa.NO_TRADE_EPS = band_width
    pa.BAND_WIDTH = band_width
    try:
        return pooled_anatomy(
            cfg,
            val_paths=val_paths,
            spec=spec,
            nets=nets,
            device=device,
        )
    finally:
        pa.NO_TRADE_EPS = old_eps
        pa.BAND_WIDTH = old_bw


def train_fixed_recipes(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    champion: dict,
) -> tuple[list[dict], list[dict]]:
    """Return (rm_rows, lag_rows) with rollout + anatomy filled later by caller."""
    target_cvar = champion["cvar95"]
    rm_meta: list[dict] = []
    lag_meta: list[dict] = []
    rm_nets: list = []
    lag_nets: list = []

    rm_hyper = replace(hyper, alpha=RM_FIXED_ALPHA)
    for torch_seed in hyper.torch_seeds:
        net, _ = train_single_net(cfg, rm_hyper, int(torch_seed), device)
        rm_nets.append(net)
        rm_meta.append(
            {
                "label": f"rm_alpha_{RM_FIXED_ALPHA}_seed_{torch_seed}",
                "kind": "seed",
                "recipe": "risk_matched_fixed_alpha",
                "torch_seed": int(torch_seed),
                "training_alpha": RM_FIXED_ALPHA,
            }
        )

    lag_hyper = replace(hyper, alpha=LAGRANGE_ALPHA)
    for torch_seed in hyper.torch_seeds:
        net, history, meta = train_lagrange_net(
            cfg, lag_hyper, int(torch_seed), target_cvar, device
        )
        lag_nets.append(net)
        lag_meta.append(
            {
                "label": f"lag_alpha_{LAGRANGE_ALPHA}_seed_{torch_seed}",
                "kind": "seed",
                "recipe": "lagrange_excess_cvar",
                "torch_seed": int(torch_seed),
                "training_alpha": LAGRANGE_ALPHA,
                "train_constraint": meta,
                "final_train_loss": history[-1] if history else None,
            }
        )

    return (rm_meta, rm_nets), (lag_meta, lag_nets)


def enrich_rows(
    cfg: B.BenchConfig,
    val_paths,
    rows: list[dict],
    nets: list,
    device: str,
    band_anatomy: dict,
    champion_cvar: float,
    band_width: float,
) -> list[dict]:
    out: list[dict] = []
    for meta, net in zip(rows, nets):
        rollout = pooled_deep_rollout_from_paths(cfg, val_paths, [net], device)
        anatomy = pooled_anatomy_with_band_width(
            cfg, band_width, val_paths=val_paths, nets=[net], device=device
        )
        row = {**meta, **rollout, **anatomy}
        row["seed_verdict"] = classify_seed(row, band_anatomy, champion_cvar)
        out.append(row)
    return out


def save_plot(
    band_row: dict,
    seed_rows: list[dict],
    band_width: float,
    path: pathlib.Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    labels = ["band_champ"] + [
        f"s{r['torch_seed']}_{'rm' if r['recipe'] == 'risk_matched_fixed_alpha' else 'lag'}"
        for r in seed_rows
    ]
    ax = axes[0]
    ax.bar(
        labels,
        [band_row["no_trade_mass"]] + [r["no_trade_mass"] for r in seed_rows],
        color="tab:orange",
    )
    ax.set_ylabel(f"no-trade mass (|Δδ| ≤ {band_width})")
    ax.set_title("No-trade mass vs 10 bp band champion")
    ax.tick_params(axis="x", rotation=45)
    ax2 = axes[1]
    ax2.bar(
        labels,
        [band_row["mean_abs_delta"]] + [r["mean_abs_delta"] for r in seed_rows],
        color="tab:blue",
    )
    ax2.set_ylabel("mean |Δδ|")
    ax2.set_title("Mean adjustment size")
    ax2.tick_params(axis="x", rotation=45)
    fig.suptitle("RQ-J: liquid policy anatomy @ 10 bp")
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
    cfg = replace(B.BenchConfig(), cost=LIQUID_COST)
    hyper = replace(Hyper(), torch_seeds=EXPANDED_TORCH_SEEDS)
    if args.quick:
        cfg, hyper_quick = B._quick_setup(cfg, hyper)
        cfg = replace(cfg, cost=LIQUID_COST)
        hyper = replace(hyper_quick, torch_seeds=(0,))

    started = time.time()
    val_paths = load_val_paths(cfg)

    classical_rows, champion = sweep_classical_hull(cfg, val_paths)
    band_cand, band_width = band_comparator_from_champion(champion)
    band_spec = strategy_spec(band_cand)
    band_rollout = pooled_rollout_from_paths(cfg, val_paths, band_spec)
    band_anatomy = pooled_anatomy_with_band_width(
        cfg, band_width, val_paths=val_paths, spec=band_cand
    )
    band_row = {
        **band_cand,
        **band_rollout,
        **band_anatomy,
        "kind": "comparator",
        "band_width_anatomy": band_width,
    }
    champion_cvar = champion["cvar95"]

    (rm_meta, rm_nets), (lag_meta, lag_nets) = train_fixed_recipes(
        cfg, hyper, device, champion
    )
    rm_rows = enrich_rows(
        cfg, val_paths, rm_meta, rm_nets, device, band_anatomy, champion_cvar, band_width
    )
    lag_rows = enrich_rows(
        cfg, val_paths, lag_meta, lag_nets, device, band_anatomy, champion_cvar, band_width
    )
    all_seed_rows = rm_rows + lag_rows
    verdict = acceptance_verdict(all_seed_rows, band_anatomy, champion_cvar)

    frozen_cb = B.load_classical_best()
    frozen_spec = strategy_spec({**frozen_cb, "label": "classical_best"})

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "research_question": "rq_j_liquid_policy_anatomy",
        "acceptance": {
            "band_like": verdict == "band_like",
            "distinct": verdict == "distinct",
            "inconclusive": verdict == "inconclusive",
            "bars": {
                "no_trade_mass_ratio_min": NO_TRADE_RATIO,
                "mean_abs_delta_ratio_max": MEAN_DD_MULT,
                "no_trade_epsilon": band_width,
                "comparator_band_width": band_width,
                "requires_cvar95_lte_champion": True,
                "not_leland_epsilon_zero": True,
            },
            "champion": {
                "label": champion["label"],
                "mean_cost": champion["mean_cost"],
                "cvar95": champion_cvar,
            },
        },
        "bench": {
            "name": cfg.name,
            "model": cfg.model,
            "cost": cfg.cost,
            "cost_bp": int(round(cfg.cost * 10_000)),
            "n_steps": cfg.n_steps,
            "val_seeds": list(cfg.val_seeds),
            "n_val_paths": cfg.n_val_paths,
            "quick": args.quick,
        },
        "hyper": asdict(hyper),
        "recipes": {
            "risk_matched_fixed_alpha": {
                "training_alpha": RM_FIXED_ALPHA,
                "description": "single CVaR-loss α=0.05 per torch seed (no OOS α sweep)",
            },
            "lagrange_excess_cvar": {
                "training_alpha": LAGRANGE_ALPHA,
                "target_cvar95_oos_champion": champion_cvar,
            },
        },
        "torch_seeds_documentation": {
            "expanded_set": list(EXPANDED_TORCH_SEEDS),
            "note": "Fixed for RQ-I alignment; six seeds beyond RQ-H {0,1,2}",
        },
        "classical": classical_rows,
        "champion": champion,
        "comparators": {"band_champion": band_row},
        "deep": {
            "risk_matched_fixed_alpha": rm_rows,
            "lagrange_excess_cvar": lag_rows,
        },
        "audit": {
            "loaded_classical_best_json": False,
            "champion_from_val_sharpe_sweep": True,
            "champion_matches_frozen_cb_spec": champion["spec"] == frozen_spec,
            "frozen_cb_spec_for_audit_only": frozen_spec,
        },
        "wall_time_s": time.time() - started,
    }
    json_name = "liquid_anatomy_quick.json" if args.quick else "liquid_anatomy.json"
    json_path = OUT / json_name
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("liquid_anatomy_quick.png" if args.quick else "liquid_anatomy.png")
        save_plot(band_row, all_seed_rows, band_width, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
