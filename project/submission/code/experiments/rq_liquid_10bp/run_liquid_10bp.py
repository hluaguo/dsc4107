"""RQ-H: liquid 10 bp — risk-matched α sweep + Lagrange CVaR vs re-swept classical hull.

Sepp's liquid-index **k** level is mirrored numerically as ``BenchConfig.cost = 0.001``
(full proportional rate on ``hedge.rollout``, not Sepp's k/2 ledger). The classical
champion is the val-Sharpe winner on the RQ-A grid at that cost only — never
``bench/classical_best.json`` (20 bp). Two deep arms (RQ-D-style α sweep and
RQ-F-style train-time excess-CVaR at fixed ``Hyper().alpha``) are scored on the
same OOS pool against that champion.
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
from experiments.rq_risk_matched.run_risk_matched import (  # noqa: E402
    FULL_ALPHAS,
    QUICK_ALPHAS,
    select_risk_matched,
    train_single_net,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

LIQUID_COST = 0.001


def clears_champion(seed_rows: list[dict], champion: dict) -> bool:
    """Lead-locked: strict cost improvement at tail no worse than the 10 bp champion."""
    return any(
        row["mean_cost"] < champion["mean_cost"]
        and row["cvar95"] <= champion["cvar95"]
        for row in seed_rows
    )


def arm_verdict(seed_rows: list[dict], champion: dict) -> str:
    return "clears" if clears_champion(seed_rows, champion) else "fails"


def overall_verdict(all_seed_rows: list[dict], champion: dict) -> str:
    return arm_verdict(all_seed_rows, champion)


def soft_interesting(all_seed_rows: list[dict], champion: dict, verdict: str) -> bool:
    if verdict == "clears":
        return True
    cost_ceiling = champion["mean_cost"] * 1.01
    return any(
        row["cvar95"] <= champion["cvar95"] and row["mean_cost"] <= cost_ceiling
        for row in all_seed_rows
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


def run_risk_matched_arm(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    champion: dict,
    alphas: tuple[float, ...],
) -> dict:
    target_cvar = champion["cvar95"]
    target_cost = champion["mean_cost"]
    deep_rows: list[dict] = []
    selected_nets: list = []
    for torch_seed in hyper.torch_seeds:
        sweep_rows: list[dict] = []
        nets_by_alpha: dict[float, object] = {}
        for alpha in alphas:
            run_hyper = replace(hyper, alpha=float(alpha))
            net, _ = train_single_net(cfg, run_hyper, int(torch_seed), device)
            nets_by_alpha[float(alpha)] = net
            stats = pooled_deep_rollout_metrics(cfg, [net], device)
            row = {
                "label": f"rm_seed_{torch_seed}_alpha_{alpha}",
                "kind": "sweep",
                "arm": "risk_matched",
                "torch_seed": int(torch_seed),
                "alpha": float(alpha),
                **stats,
            }
            sweep_rows.append(row)
            deep_rows.append(row)
        chosen = select_risk_matched(sweep_rows, target_cvar, target_cost)
        selected_nets.append(nets_by_alpha[float(chosen["alpha"])])
        deep_rows.append(
            {
                "label": f"rm_seed_{torch_seed}_selected",
                "kind": "seed",
                "arm": "risk_matched",
                "torch_seed": int(torch_seed),
                "alpha": chosen["alpha"],
                "mean_cost": chosen["mean_cost"],
                "cvar95": chosen["cvar95"],
                "pnl_mean": chosen["pnl_mean"],
                "selection": "min cost among cvar<=champion else min cvar",
            }
        )
    ensemble_stats = pooled_deep_rollout_metrics(cfg, selected_nets, device)
    deep_rows.append(
        {
            "label": "rm_ensemble_selected",
            "kind": "ensemble",
            "arm": "risk_matched",
            "torch_seeds": list(hyper.torch_seeds),
            **ensemble_stats,
        }
    )
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    return {
        "arm": "risk_matched",
        "verdict": arm_verdict(seed_rows, champion),
        "alpha_grid": list(alphas),
        "deep": deep_rows,
    }


def run_lagrange_arm(
    cfg: B.BenchConfig,
    hyper: Hyper,
    device: str,
    champion: dict,
) -> dict:
    target_cvar = champion["cvar95"]
    deep_rows: list[dict] = []
    nets: list = []
    for torch_seed in hyper.torch_seeds:
        net, history, meta = train_lagrange_net(
            cfg, hyper, int(torch_seed), target_cvar, device
        )
        nets.append(net)
        stats = pooled_deep_rollout_metrics(cfg, [net], device)
        deep_rows.append(
            {
                "label": f"lag_seed_{torch_seed}_cvar_capped",
                "kind": "seed",
                "arm": "lagrange_excess_cvar",
                "torch_seed": int(torch_seed),
                "training_alpha": float(hyper.alpha),
                "train_constraint": meta,
                "final_train_loss": history[-1] if history else None,
                **stats,
            }
        )
    ensemble_stats = pooled_deep_rollout_metrics(cfg, nets, device)
    deep_rows.append(
        {
            "label": "lag_ensemble_cvar_capped",
            "kind": "ensemble",
            "arm": "lagrange_excess_cvar",
            "torch_seeds": list(hyper.torch_seeds),
            **ensemble_stats,
        }
    )
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    return {
        "arm": "lagrange_excess_cvar",
        "verdict": arm_verdict(seed_rows, champion),
        "train_constraint": {
            "mechanism": "lagrange_excess_batch_cvar95",
            "target_cvar95_oos_champion": target_cvar,
            "training_alpha_fixed": float(hyper.alpha),
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
    rm_sweep = [r for r in rm_arm["deep"] if r.get("kind") == "sweep"]
    if rm_sweep:
        ax.scatter(
            [r["mean_cost"] for r in rm_sweep],
            [r["cvar95"] for r in rm_sweep],
            c="tab:blue",
            alpha=0.25,
            s=28,
            label="RM α sweep",
            zorder=1,
        )
    for row in rm_arm["deep"]:
        if row.get("kind") == "seed":
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:blue",
                marker="D",
                s=75,
                zorder=4,
            )
    for row in lag_arm["deep"]:
        if row.get("kind") == "seed":
            clears = (
                row["mean_cost"] < champion["mean_cost"]
                and row["cvar95"] <= champion["cvar95"]
            )
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:green" if clears else "tab:cyan",
                marker="^",
                s=75,
                zorder=5,
            )
    ax.scatter([], [], c="tab:cyan", marker="^", s=75, label="Lagrange seeds")
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-H: liquid 10 bp — RM + Lagrange vs re-swept hull")
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
    alphas = QUICK_ALPHAS if args.quick else FULL_ALPHAS
    if args.quick:
        base_cfg, hyper = B._quick_setup(base_cfg, hyper)
        base_cfg = replace(base_cfg, cost=LIQUID_COST)

    started = time.time()
    classical_rows, champion = sweep_classical_hull(base_cfg)

    frozen_cb = B.load_classical_best()
    frozen_spec = strategy_spec({**frozen_cb, "label": "classical_best"})
    champion_matches_frozen_cb_spec = champion["spec"] == frozen_spec

    rm_arm = run_risk_matched_arm(base_cfg, hyper, device, champion, alphas)
    lag_arm = run_lagrange_arm(base_cfg, hyper, device, champion)

    all_seed_rows = [
        r
        for arm in (rm_arm, lag_arm)
        for r in arm["deep"]
        if r.get("kind") == "seed"
    ]
    verdict = overall_verdict(all_seed_rows, champion)
    interesting = soft_interesting(all_seed_rows, champion, verdict)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "soft_interesting": interesting,
        "research_question": "rq_h_liquid_10bp_risk_matched_and_lagrange",
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
        "classical": classical_rows,
        "champion": champion,
        "arms": {
            "risk_matched": rm_arm,
            "lagrange_excess_cvar": lag_arm,
        },
        "acceptance": {
            "clears": verdict == "clears",
            "fails": verdict == "fails",
            "rule": (
                "any seed (either arm): OOS mean_cost < 10bp champion "
                "AND OOS cvar95 <= champion"
            ),
            "soft_interesting_rule": (
                "clears OR cvar<=champion with mean_cost within 1% of champion"
            ),
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
    json_path = OUT / ("liquid_10bp_quick.json" if args.quick else "liquid_10bp.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict} interesting={interesting}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("liquid_10bp_quick.png" if args.quick else "liquid_10bp.png")
        save_plot(classical_rows, champion, rm_arm, lag_arm, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
