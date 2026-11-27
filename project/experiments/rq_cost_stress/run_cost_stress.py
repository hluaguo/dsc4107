"""RQ-G: classical hull vs deep hedging across proportional cost levels.

Sepp-style calendar and band rules span a cost–tail frontier at each friction
level; Buehler-style deep hedging is trained fresh at that ``BenchConfig.cost``
without retuning anything else. ``bench/classical_best.json`` is the frozen
20 bp champion only — here we re-score the RQ-A grid at each
``k ∈ {0.001, 0.002, 0.005}`` and pick that cost's Sharpe leader on the shared
validation pool, never importing the pinned JSON at ``k ≠ 0.002``.
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
from hedging import hedge, metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import (  # noqa: E402
    acceptance_verdict,
    pooled_deep_rollout_metrics,
    pooled_rollout_metrics,
    resolve_device,
    strategy_spec,
    weakly_dominated,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

# Same grid family as RQ-A ``classical_candidates`` minus the pinned JSON row.
COST_LEVELS = (0.001, 0.002, 0.005)


def classical_grid_only() -> list[dict]:
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
    return points


def mean_val_sharpe(cfg: B.BenchConfig, spec: dict) -> float:
    """Sharpe on the same OOS paths used for cost/CVaR so the champion is auditable."""
    sharpes: list[float] = []
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
        sharpes.append(metrics.sharpe(result.pnl))
    return float(np.mean(sharpes))


def pick_champion(classical_rows: list[dict]) -> dict:
    return max(classical_rows, key=lambda row: row["mean_sharpe"])


def per_k_dominated(deep_rows: list[dict], classical_rows: list[dict]) -> bool:
    return all(weakly_dominated(row, classical_rows) for row in deep_rows)


def overall_verdict(per_k: dict[str, str]) -> str:
    if all(v == "dominated" for v in per_k.values()):
        return "dominated_all_k"
    if any(v == "improves" for v in per_k.values()):
        return "clears_some_k"
    return "mixed"


def save_plot(cells: dict, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, len(COST_LEVELS), figsize=(5 * len(COST_LEVELS), 4.5))
    if len(COST_LEVELS) == 1:
        axes = [axes]
    for ax, k in zip(axes, COST_LEVELS):
        key = f"{k:.3f}"
        cell = cells[key]
        classical_rows = cell["classical"]
        deep_rows = cell["deep"]
        cx = [r["mean_cost"] for r in classical_rows]
        cy = [r["cvar95"] for r in classical_rows]
        ax.scatter(cx, cy, c="tab:orange", s=45, label="classical grid", zorder=2)
        champ = cell["champion"]
        ax.scatter(
            [champ["mean_cost"]],
            [champ["cvar95"]],
            c="tab:red",
            s=100,
            marker="*",
            label="champion",
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
                s=55,
                zorder=4,
            )
        ax.set_xlabel("mean path proportional cost")
        ax.set_ylabel("CVaR95")
        ax.set_title(f"k = {k:.3f} ({int(k * 1e4)} bp)")
        ax.legend(loc="best", fontsize=7)
    fig.suptitle("RQ-G: deep vs classical hull by cost", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_cell(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    cost: float,
) -> dict:
    grid = classical_grid_only()
    classical_rows: list[dict] = []
    for cand in grid:
        spec = strategy_spec(cand)
        stats = pooled_rollout_metrics(cfg, spec)
        sharpe = mean_val_sharpe(cfg, spec)
        classical_rows.append(
            {
                "label": cand["label"],
                "kind": "classical",
                "spec": spec,
                "mean_sharpe": sharpe,
                **stats,
            }
        )

    champion = pick_champion(classical_rows)
    frozen_cb = B.load_classical_best()
    frozen_spec = strategy_spec({**frozen_cb, "label": "classical_best"})
    champion_matches_frozen = cost == 0.002 and champion["spec"] == frozen_spec

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

    verdict = acceptance_verdict(deep_rows, classical_rows, champion)
    return {
        "cost": cost,
        "cost_bp": int(round(cost * 10_000)),
        "verdict": verdict,
        "classical": classical_rows,
        "champion": {
            "label": champion["label"],
            "spec": champion["spec"],
            "mean_sharpe": champion["mean_sharpe"],
            "mean_cost": champion["mean_cost"],
            "cvar95": champion["cvar95"],
        },
        "deep": deep_rows,
        "acceptance": {
            "dominated": verdict == "dominated",
            "improves": verdict == "improves",
            "flat": verdict == "flat",
            "weakly_dominated_all_deep": per_k_dominated(deep_rows, classical_rows),
        },
        "audit": {
            "loaded_classical_best_json": False,
            "champion_from_val_sharpe_sweep": True,
            "champion_matches_frozen_cb_at_20bp": champion_matches_frozen,
        },
    }


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
    base_cfg = B.BenchConfig()
    hyper = Hyper()
    if args.quick:
        base_cfg, hyper = B._quick_setup(base_cfg, hyper)

    started = time.time()
    cells: dict[str, dict] = {}
    per_k_verdicts: dict[str, str] = {}

    for cost in COST_LEVELS:
        cfg = replace(base_cfg, cost=cost)
        cell = run_cell(cfg, hyper, device, cost)
        key = f"{cost:.3f}"
        cells[key] = cell
        per_k_verdicts[key] = cell["verdict"]
        print(f"k={cost:.3f} verdict={cell['verdict']}")

    verdict = overall_verdict(per_k_verdicts)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "research_question": (
            "Does unconstrained deep hedging remain weakly dominated by the "
            "classical cost–CVaR hull re-swept at each proportional cost?"
        ),
        "cost_levels": list(COST_LEVELS),
        "per_k_verdicts": per_k_verdicts,
        "cells": cells,
        "acceptance": {
            "dominated_all_k": verdict == "dominated_all_k",
            "clears_some_k": verdict == "clears_some_k",
            "mixed": verdict == "mixed",
        },
        "bench_base": {
            "name": base_cfg.name,
            "model": base_cfg.model,
            "default_cost": B.BenchConfig().cost,
            "n_steps": base_cfg.n_steps,
            "val_seeds": list(base_cfg.val_seeds),
            "n_val_paths": base_cfg.n_val_paths,
            "quick": args.quick,
        },
        "hyper": asdict(hyper),
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("cost_stress_quick.json" if args.quick else "cost_stress.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"overall_verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("cost_stress_quick.png" if args.quick else "cost_stress.png")
        save_plot(cells, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
