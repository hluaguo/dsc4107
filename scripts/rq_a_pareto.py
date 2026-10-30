"""RQ-A: classical cost–CVaR frontier vs unconstrained deep hedging on the frozen cell.

The main research question is not another Sharpe leaderboard on a 4×5 grid but
whether Buehler-style deep hedging clears a Sepp-style classical hull once band
and calendar policies are both on the table. We therefore fix ``BenchConfig()``
and score every policy on the same validation paths (``val_seeds`` ×
``n_val_paths``) with two tail-aware objectives: mean proportional transaction
cost along each path and CVaR95 of terminal P&L, both from ``hedge.rollout``.
Hyperparameters stay at
``Hyper()`` defaults so the learned side is the unconstrained recipe from the
Deep Hedging replication, not a tuned competitor.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import hedge, metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "rq_a"


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def classical_candidates() -> list[dict]:
    """Grid named so JSON rows stay identifiable without re-parsing strategy dicts."""
    points: list[dict] = []
    for frequency in (1, 3, 7, 21):
        points.append(
            {
                "label": f"time_f{frequency}",
                "strategy": "time",
                "frequency": frequency,
                "delta_band": None,
                "price_band": None,
            }
        )
    for delta_band in (0.05, 0.10, 0.20):
        points.append(
            {
                "label": f"band_delta_{delta_band:.2f}",
                "strategy": "band",
                "frequency": 1,
                "delta_band": delta_band,
                "price_band": None,
            }
        )
    for price_band in (0.01, 0.02, 0.05):
        points.append(
            {
                "label": f"band_price_{price_band:.2f}",
                "strategy": "band",
                "frequency": 1,
                "delta_band": None,
                "price_band": price_band,
            }
        )
    best = dict(B.load_classical_best())
    best["label"] = "classical_best"
    points.append(best)
    return points


def strategy_spec(row: dict) -> dict:
    return {
        k: row[k]
        for k in ("strategy", "frequency", "delta_band", "price_band")
        if k in row
    }


def pooled_rollout_metrics(cfg: B.BenchConfig, spec: dict) -> dict:
    """One fresh position function per seed batch because band anchors are path-local."""
    cost_chunks: list[np.ndarray] = []
    pnl_chunks: list[np.ndarray] = []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
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


class _ArrayPositionFn:
    """Replay network-chosen targets through ``hedge.rollout`` for cost accounting."""

    def __init__(self, targets: np.ndarray):
        self.targets = targets

    def __call__(self, j: int, spot: np.ndarray, pos: np.ndarray) -> np.ndarray:
        return self.targets[:, j]


def pooled_deep_rollout_metrics(cfg: B.BenchConfig, nets, device: str) -> dict:
    """Same rollout ledger as classical: positions then ``hedge.rollout`` for cost and P&L."""
    from hedging import deep

    cost_chunks: list[np.ndarray] = []
    pnl_chunks: list[np.ndarray] = []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
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
            # Pathwise average matches ``NetworkStrategy`` cash-flow mixing, but both
            # axes stay on the shared rollout books rather than ``evaluate_network``.
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


def weakly_dominated(point: dict, front: list[dict]) -> bool:
    """True if some frontier point is at least as good on cost and CVaR (lower is better)."""
    mc, cv = point["mean_cost"], point["cvar95"]
    for other in front:
        if other["mean_cost"] <= mc and other["cvar95"] <= cv:
            return True
    return False


def acceptance_verdict(
    deep_rows: list[dict], classical_rows: list[dict], best_row: dict
) -> str:
    front = classical_rows
    deep_targets = deep_rows
    if all(weakly_dominated(row, front) for row in deep_targets):
        return "dominated"
    n_seeds = sum(1 for row in deep_rows if row.get("kind") == "seed")
    need = math.ceil(2 * n_seeds / 3) if n_seeds else 2
    wins = sum(
        1
        for row in deep_rows
        if row.get("kind") == "seed"
        and row["cvar95"] < best_row["cvar95"]
        and row["mean_cost"] <= best_row["mean_cost"]
    )
    if wins >= need:
        return "improves"
    return "flat"


def save_plot(classical_rows: list[dict], deep_rows: list[dict], path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cx = [r["mean_cost"] for r in classical_rows]
    cy = [r["cvar95"] for r in classical_rows]
    ax.scatter(cx, cy, c="tab:orange", s=55, label="classical grid + best", zorder=2)
    for row in classical_rows:
        if row.get("label") == "classical_best":
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:red",
                s=120,
                marker="*",
                label="classical_best",
                zorder=3,
            )
    for row in deep_rows:
        marker = "D" if row.get("kind") == "ensemble" else "o"
        color = "tab:blue" if row.get("kind") == "seed" else "tab:purple"
        ax.scatter(
            [row["mean_cost"]],
            [row["cvar95"]],
            c=color,
            marker=marker,
            s=70,
            zorder=4,
        )
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-A: frozen Heston ATM 1M @ 20 bp")
    ax.legend(loc="best", fontsize=8)
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
    if args.quick:
        cfg, hyper = B._quick_setup(cfg, hyper)

    started = time.time()
    classical_rows: list[dict] = []
    for cand in classical_candidates():
        spec = strategy_spec(cand)
        stats = pooled_rollout_metrics(cfg, spec)
        classical_rows.append(
            {
                "label": cand["label"],
                "kind": "classical",
                "spec": spec,
                **stats,
            }
        )

    best_row = next(r for r in classical_rows if r["label"] == "classical_best")

    nets, _ = B.train_ensemble(cfg, hyper, device=device)
    deep_rows: list[dict] = []
    for seed, net in zip(hyper.torch_seeds, nets):
        stats = pooled_deep_rollout_metrics(cfg, [net], device)
        deep_rows.append(
            {
                "label": f"deep_seed_{seed}",
                "kind": "seed",
                "torch_seed": int(seed),
                **stats,
            }
        )
    ensemble_stats = pooled_deep_rollout_metrics(cfg, nets, device)
    deep_rows.append(
        {
            "label": "deep_ensemble",
            "kind": "ensemble",
            "torch_seeds": list(hyper.torch_seeds),
            **ensemble_stats,
        }
    )

    verdict = acceptance_verdict(deep_rows, classical_rows, best_row)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
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
        "classical": classical_rows,
        "deep": deep_rows,
        "acceptance": {
            "dominated": verdict == "dominated",
            "improves": verdict == "improves",
            "flat": verdict == "flat",
            "classical_best": {
                "mean_cost": best_row["mean_cost"],
                "cvar95": best_row["cvar95"],
            },
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("pareto_quick.json" if args.quick else "pareto.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("pareto_quick.png" if args.quick else "pareto.png")
        save_plot(classical_rows, deep_rows, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
