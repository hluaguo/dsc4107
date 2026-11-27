"""RQ-K: liquid 10 bp unconstrained ``Hyper()`` baseline vs re-swept classical champion.

Default CVaR training (``alpha=0.65``) with no risk-match selection and no Lagrange
constraint. Purpose: isolate whether RQ-H's clears require those knobs when scored
against the same 10 bp val-Sharpe champion as the other liquid RQs.
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

from hedging import bench as B  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import (  # noqa: E402
    pooled_deep_rollout_metrics,
    pooled_rollout_metrics,
    resolve_device,
    strategy_spec,
    weakly_dominated,
)
from experiments.rq_cost_stress.run_cost_stress import (  # noqa: E402
    classical_grid_only,
    mean_val_sharpe,
    pick_champion,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

LIQUID_COST = 0.001

# Lead lock with RQ-I: more training-path batches than the frozen three-seed cell.
RQ_I_K_TRAIN_SEEDS: tuple[int, ...] = (1001, 1002, 1003, 1004, 1005, 1006)
# Six torch inits beyond RQ-H's (0, 1, 2) so variance is visible without shrinking seeds.
RQ_I_K_TORCH_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def seed_clears_champion(row: dict, champion: dict) -> bool:
    return (
        row["mean_cost"] < champion["mean_cost"]
        and row["cvar95"] <= champion["cvar95"]
    )


def count_clearing_seeds(seed_rows: list[dict], champion: dict) -> int:
    return sum(1 for row in seed_rows if seed_clears_champion(row, champion))


def lead_verdict(deep_rows: list[dict], classical_rows: list[dict], champion: dict) -> str:
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    if any(seed_clears_champion(row, champion) for row in seed_rows):
        return "clears"
    if all(weakly_dominated(row, classical_rows) for row in deep_rows):
        return "dominated"
    return "mixed"


def soft_interesting(
    seed_rows: list[dict], champion: dict, verdict: str, n_seeds: int
) -> bool:
    if verdict == "clears":
        return True
    n_clear = count_clearing_seeds(seed_rows, champion)
    if n_seeds and n_clear >= (n_seeds + 1) // 2:
        return True
    cost_ceiling = champion["mean_cost"] * 1.01
    return any(
        row["cvar95"] <= champion["cvar95"] and row["mean_cost"] <= cost_ceiling
        for row in seed_rows
    )


def sweep_classical_hull(cfg: B.BenchConfig) -> tuple[list[dict], dict]:
    classical_rows: list[dict] = []
    for cand in classical_grid_only():
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
    champion_row = pick_champion(classical_rows)
    champion = {
        "label": champion_row["label"],
        "spec": champion_row["spec"],
        "mean_sharpe": champion_row["mean_sharpe"],
        "mean_cost": champion_row["mean_cost"],
        "cvar95": champion_row["cvar95"],
    }
    return classical_rows, champion


def run_unconstrained_arm(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
) -> list[dict]:
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
    return deep_rows


def save_plot(
    classical_rows: list[dict],
    champion: dict,
    deep_rows: list[dict],
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cx = [r["mean_cost"] for r in classical_rows]
    cy = [r["cvar95"] for r in classical_rows]
    ax.scatter(cx, cy, c="tab:orange", s=45, label="classical grid (10 bp)", zorder=2)
    ax.scatter(
        [champion["mean_cost"]],
        [champion["cvar95"]],
        c="tab:red",
        s=140,
        marker="*",
        label="val-Sharpe champion",
        zorder=3,
    )
    for row in deep_rows:
        if row.get("kind") == "ensemble":
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:purple",
                marker="D",
                s=90,
                label="deep ensemble",
                zorder=5,
            )
        elif row.get("kind") == "seed":
            clears = seed_clears_champion(row, champion)
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:green" if clears else "tab:blue",
                marker="o",
                s=55,
                zorder=4,
            )
    ax.scatter([], [], c="tab:blue", marker="o", s=55, label="deep seeds")
    ax.scatter([], [], c="tab:green", marker="o", s=55, label="seed clears champion")
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-K: liquid 10 bp — unconstrained Hyper() vs re-swept hull")
    ax.legend(loc="best", fontsize=7)
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
    base_cfg = replace(
        B.BenchConfig(),
        cost=LIQUID_COST,
        train_seeds=RQ_I_K_TRAIN_SEEDS,
    )
    hyper = replace(Hyper(), torch_seeds=RQ_I_K_TORCH_SEEDS)
    if args.quick:
        base_cfg, hyper = B._quick_setup(base_cfg, hyper)
        base_cfg = replace(base_cfg, cost=LIQUID_COST)

    started = time.time()
    classical_rows, champion = sweep_classical_hull(base_cfg)
    deep_rows = run_unconstrained_arm(base_cfg, hyper, device)

    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    verdict = lead_verdict(deep_rows, classical_rows, champion)
    n_clear = count_clearing_seeds(seed_rows, champion)
    interesting = soft_interesting(
        seed_rows, champion, verdict, len(hyper.torch_seeds)
    )

    frozen_cb = B.load_classical_best()
    frozen_spec = strategy_spec({**frozen_cb, "label": "classical_best"})

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "soft_interesting": interesting,
        "research_question": "rq_k_liquid_10bp_unconstrained_baseline",
        "bench": {
            "name": base_cfg.name,
            "model": base_cfg.model,
            "cost": base_cfg.cost,
            "cost_bp": int(round(base_cfg.cost * 10_000)),
            "n_steps": base_cfg.n_steps,
            "train_seeds": list(base_cfg.train_seeds),
            "val_seeds": list(base_cfg.val_seeds),
            "n_val_paths": base_cfg.n_val_paths,
            "n_train_paths": base_cfg.n_train_paths,
            "quick": args.quick,
        },
        "hyper": asdict(hyper),
        "training": {
            "mechanism": "default_cvar_loss",
            "alpha": float(hyper.alpha),
            "risk_match_selection": False,
            "lagrange_excess_cvar": False,
            "rq_i_k_train_seeds": list(RQ_I_K_TRAIN_SEEDS),
            "rq_i_k_torch_seeds": list(RQ_I_K_TORCH_SEEDS),
        },
        "classical": classical_rows,
        "champion": champion,
        "deep": deep_rows,
        "acceptance": {
            "dominated": verdict == "dominated",
            "clears": verdict == "clears",
            "mixed": verdict == "mixed",
            "n_seeds_clearing_champion": n_clear,
            "n_torch_seeds": len(hyper.torch_seeds),
            "rule": (
                "clears if any seed has OOS mean_cost < 10bp champion "
                "AND cvar95 <= champion; dominated if every deep row "
                "(seeds + ensemble) is weakly dominated by some classical "
                "grid point; else mixed"
            ),
            "soft_interesting_rule": (
                "clears OR any clearing seed OR half+ seeds clear OR "
                "cvar<=champion with mean_cost within 1% of champion"
            ),
            "champion": {
                "label": champion["label"],
                "mean_cost": champion["mean_cost"],
                "cvar95": champion["cvar95"],
            },
        },
        "audit": {
            "loaded_classical_best_json": False,
            "champion_from_val_sharpe_sweep": True,
            "champion_matches_frozen_cb_spec": champion["spec"] == frozen_spec,
            "frozen_cb_spec_for_audit_only": frozen_spec,
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / (
        "liquid_baseline_quick.json" if args.quick else "liquid_baseline.json"
    )
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict} clearing_seeds={n_clear} interesting={interesting}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / (
            "liquid_baseline_quick.png" if args.quick else "liquid_baseline.png"
        )
        save_plot(classical_rows, champion, deep_rows, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
