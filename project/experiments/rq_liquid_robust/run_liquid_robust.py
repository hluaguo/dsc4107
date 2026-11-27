"""RQ-I: liquid 10 bp robustness — fixed RQ-H recipes, expanded train seeds.

RQ-H found one torch seed per arm clearing the re-swept 10 bp champion. Here both
arms reuse those **fixed** training recipes (RM ``alpha=0.05``, Lagrange at
``Hyper().alpha=0.65``) while doubling RQ-H's ``torch_seeds`` so the Lead verdict
can distinguish robust replication from single-seed luck.
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
)
from experiments.rq_cost_stress.run_cost_stress import (  # noqa: E402
    classical_grid_only,
    mean_val_sharpe,
    pick_champion,
)
from experiments.rq_hard_cvar.run_hard_cvar import train_seed_net as train_lagrange_net
from experiments.rq_risk_matched.run_risk_matched import train_single_net

OUT = pathlib.Path(__file__).resolve().parent / "results"

LIQUID_COST = 0.001

# RQ-H used default ``Hyper().torch_seeds``; we keep those and add three more so
# luck at seed 1 cannot drive the headline alone.
RQ_H_TORCH_SEEDS = (0, 1, 2)
FULL_TORCH_SEEDS = (0, 1, 2, 3, 4, 5)
QUICK_TORCH_SEEDS = (0, 1)

RM_FIXED_ALPHA = 0.05
LAG_FIXED_ALPHA = 0.65


def seed_clears_champion(row: dict, champion: dict) -> bool:
    return (
        row["mean_cost"] < champion["mean_cost"]
        and row["cvar95"] <= champion["cvar95"]
    )


def pooled_robustness_verdict(seed_rows: list[dict], champion: dict) -> tuple[str, dict]:
    """Lead-locked RQ-I labels on all arm×seed OOS rows (prefer pooled)."""
    flags = [seed_clears_champion(r, champion) for r in seed_rows]
    n_clear = sum(flags)
    n_total = len(seed_rows)
    half = n_total / 2.0
    if n_clear == 1:
        verdict = "brittle"
    elif n_clear >= half:
        verdict = "robust"
    else:
        verdict = "fails"
    stats = {
        "n_seed_rows_pooled": n_total,
        "n_clearing_rows": n_clear,
        "fraction_clearing": n_clear / n_total if n_total else 0.0,
        "half_threshold_rows": half,
        "clearing_torch_seeds": sorted(
            {r["torch_seed"] for r, ok in zip(seed_rows, flags) if ok}
        ),
    }
    return verdict, stats


def per_arm_verdict(seed_rows: list[dict], champion: dict) -> str:
    n_clear = sum(seed_clears_champion(r, champion) for r in seed_rows)
    n_total = len(seed_rows)
    if n_clear == 1:
        return "brittle"
    if n_total and n_clear >= n_total / 2.0:
        return "robust"
    return "fails"


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


def run_rm_fixed_arm(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    champion: dict,
) -> dict:
    train_hyper = replace(hyper, alpha=RM_FIXED_ALPHA)
    deep_rows: list[dict] = []
    nets: list = []
    for torch_seed in hyper.torch_seeds:
        net, _ = train_single_net(cfg, train_hyper, int(torch_seed), device)
        nets.append(net)
        stats = pooled_deep_rollout_metrics(cfg, [net], device)
        deep_rows.append(
            {
                "label": f"rm_fixed_seed_{torch_seed}_alpha_{RM_FIXED_ALPHA}",
                "kind": "seed",
                "arm": "risk_matched_fixed_alpha",
                "torch_seed": int(torch_seed),
                "alpha": RM_FIXED_ALPHA,
                "rq_h_recipe": "rm_seed_1_selected used alpha=0.05",
                **stats,
            }
        )
    ensemble_stats = pooled_deep_rollout_metrics(cfg, nets, device)
    deep_rows.append(
        {
            "label": "rm_fixed_ensemble",
            "kind": "ensemble",
            "arm": "risk_matched_fixed_alpha",
            "torch_seeds": list(hyper.torch_seeds),
            "alpha": RM_FIXED_ALPHA,
            **ensemble_stats,
        }
    )
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    return {
        "arm": "risk_matched_fixed_alpha",
        "fixed_alpha": RM_FIXED_ALPHA,
        "verdict": per_arm_verdict(seed_rows, champion),
        "deep": deep_rows,
    }


def run_lag_fixed_arm(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    champion: dict,
) -> dict:
    train_hyper = replace(hyper, alpha=LAG_FIXED_ALPHA)
    target_cvar = champion["cvar95"]
    deep_rows: list[dict] = []
    nets: list = []
    for torch_seed in hyper.torch_seeds:
        net, history, meta = train_lagrange_net(
            cfg, train_hyper, int(torch_seed), target_cvar, device
        )
        nets.append(net)
        stats = pooled_deep_rollout_metrics(cfg, [net], device)
        deep_rows.append(
            {
                "label": f"lag_fixed_seed_{torch_seed}_cvar_capped",
                "kind": "seed",
                "arm": "lagrange_excess_cvar_fixed_alpha",
                "torch_seed": int(torch_seed),
                "training_alpha": LAG_FIXED_ALPHA,
                "train_constraint": meta,
                "final_train_loss": history[-1] if history else None,
                **stats,
            }
        )
    ensemble_stats = pooled_deep_rollout_metrics(cfg, nets, device)
    deep_rows.append(
        {
            "label": "lag_fixed_ensemble",
            "kind": "ensemble",
            "arm": "lagrange_excess_cvar_fixed_alpha",
            "torch_seeds": list(hyper.torch_seeds),
            "training_alpha": LAG_FIXED_ALPHA,
            **ensemble_stats,
        }
    )
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    return {
        "arm": "lagrange_excess_cvar_fixed_alpha",
        "fixed_alpha": LAG_FIXED_ALPHA,
        "verdict": per_arm_verdict(seed_rows, champion),
        "train_constraint": {
            "mechanism": "lagrange_excess_batch_cvar95",
            "target_cvar95_oos_champion": target_cvar,
            "training_alpha_fixed": LAG_FIXED_ALPHA,
        },
        "deep": deep_rows,
    }


def save_plot(
    classical_rows: list[dict],
    champion: dict,
    rm_arm: dict,
    lag_arm: dict,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        [r["mean_cost"] for r in classical_rows],
        [r["cvar95"] for r in classical_rows],
        c="tab:orange",
        s=45,
        label="classical grid (10 bp)",
        zorder=2,
    )
    ax.scatter(
        [champion["mean_cost"]],
        [champion["cvar95"]],
        c="tab:red",
        s=140,
        marker="*",
        label="val-Sharpe champion",
        zorder=3,
    )
    for row in rm_arm["deep"]:
        if row.get("kind") == "seed":
            ok = seed_clears_champion(row, champion)
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:blue" if ok else "tab:gray",
                marker="D",
                s=75,
                zorder=4,
            )
    for row in lag_arm["deep"]:
        if row.get("kind") == "seed":
            ok = seed_clears_champion(row, champion)
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:green" if ok else "tab:cyan",
                marker="^",
                s=75,
                zorder=5,
            )
    ax.scatter([], [], c="tab:blue", marker="D", s=75, label="RM fixed α=0.05")
    ax.scatter([], [], c="tab:cyan", marker="^", s=75, label="Lag fixed α=0.65")
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-I: fixed H recipes — expanded seeds vs 10 bp champion")
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
    base_cfg = replace(B.BenchConfig(), cost=LIQUID_COST)
    hyper = Hyper()
    torch_seeds = QUICK_TORCH_SEEDS if args.quick else FULL_TORCH_SEEDS
    hyper = replace(hyper, torch_seeds=torch_seeds, alpha=LAG_FIXED_ALPHA)
    if args.quick:
        base_cfg, hyper = B._quick_setup(base_cfg, hyper)
        base_cfg = replace(base_cfg, cost=LIQUID_COST)
        hyper = replace(hyper, torch_seeds=QUICK_TORCH_SEEDS, alpha=LAG_FIXED_ALPHA)

    started = time.time()
    classical_rows, champion = sweep_classical_hull(base_cfg)

    frozen_cb = B.load_classical_best()
    frozen_spec = strategy_spec({**frozen_cb, "label": "classical_best"})
    champion_matches_frozen_cb_spec = champion["spec"] == frozen_spec

    rm_arm = run_rm_fixed_arm(base_cfg, hyper, device, champion)
    lag_arm = run_lag_fixed_arm(base_cfg, hyper, device, champion)

    all_seed_rows = [
        r
        for arm in (rm_arm, lag_arm)
        for r in arm["deep"]
        if r.get("kind") == "seed"
    ]
    verdict, pool_stats = pooled_robustness_verdict(all_seed_rows, champion)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "soft_interesting": verdict == "robust",
        "research_question": "rq_i_liquid_10bp_robustness_fixed_h_recipes",
        "bench": {
            "name": base_cfg.name,
            "model": base_cfg.model,
            "cost": base_cfg.cost,
            "cost_bp": int(round(base_cfg.cost * 10_000)),
            "n_steps": base_cfg.n_steps,
            "val_seeds": list(base_cfg.val_seeds),
            "n_val_paths": base_cfg.n_val_paths,
            "quick": args.quick,
        },
        "hyper": asdict(hyper),
        "seed_design": {
            "rq_h_torch_seeds": list(RQ_H_TORCH_SEEDS),
            "rq_i_torch_seeds": list(hyper.torch_seeds),
            "expansion_note": "RQ-H default (0,1,2) plus seeds 3–5; no holdout seeds",
            "rm_fixed_alpha": RM_FIXED_ALPHA,
            "lag_fixed_alpha": LAG_FIXED_ALPHA,
        },
        "classical": classical_rows,
        "champion": champion,
        "arms": {
            "risk_matched_fixed_alpha": rm_arm,
            "lagrange_excess_cvar_fixed_alpha": lag_arm,
        },
        "acceptance": {
            "robust": verdict == "robust",
            "brittle": verdict == "brittle",
            "fails": verdict == "fails",
            "rule_pooled": (
                "Across all arm×seed OOS rows: robust if ≥ half clear "
                "(mean_cost < 10bp champion AND cvar95 <= champion); "
                "brittle if exactly one row clears; else fails"
            ),
            "pool_stats": pool_stats,
            "champion": {
                "mean_cost": champion["mean_cost"],
                "cvar95": champion["cvar95"],
            },
        },
        "audit": {
            "loaded_classical_best_json": False,
            "champion_from_val_sharpe_sweep": True,
            "champion_matches_frozen_cb_spec": champion_matches_frozen_cb_spec,
            "frozen_cb_spec_for_audit_only": frozen_spec,
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("liquid_robust_quick.json" if args.quick else "liquid_robust.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict} pool_clear={pool_stats['n_clearing_rows']}/{pool_stats['n_seed_rows_pooled']}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("liquid_robust_quick.png" if args.quick else "liquid_robust.png")
        save_plot(classical_rows, champion, rm_arm, lag_arm, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
