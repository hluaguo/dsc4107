"""RQ-L: liquid 10 bp — risk-matched training-α grid robustness.

At ``BenchConfig.cost = 0.001`` (Sepp liquid-index **k** under our full-rate
ledger), sweep training ``alpha`` in ``deep.cvar_loss`` on an expanded
``torch_seeds`` pool (same as RQ-I). Per seed, select the OOS rollout point with
minimum mean cost among candidates with CVaR95 ≤ the re-swept val-Sharpe
champion (RQ-D rule). Ask whether α = 0.05 is the only dial that **clears**
that champion or several α values do.
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
from experiments.rq_risk_matched.run_risk_matched import (  # noqa: E402
    QUICK_ALPHAS,
    select_risk_matched,
    train_single_net,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

LIQUID_COST = 0.001

# Shared with RQ-I: keep default (0, 1, 2) for smoke via ``_quick_setup`` only.
EXPANDED_TORCH_SEEDS = (0, 1, 2, 3, 4)

# Neighbors of 0.05 on the RQ-D grid — small enough for a robustness cell, still
# spans seller-style training toward the default recipe.
COMPACT_ALPHAS = (0.05, 0.12, 0.20, 0.30, 0.40)


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


def seed_clears_champion(row: dict, champion: dict) -> bool:
    return (
        row["mean_cost"] < champion["mean_cost"]
        and row["cvar95"] <= champion["cvar95"]
    )


def verdict_alpha_robustness(seed_rows: list[dict], champion: dict) -> str:
    """Lead-locked: ``a05_only`` | ``several_clear`` | ``none`` on selected points."""
    clearing = [r for r in seed_rows if seed_clears_champion(r, champion)]
    if not clearing:
        return "none"
    clearing_alphas = {float(r["alpha"]) for r in clearing}
    if clearing_alphas == {0.05}:
        return "a05_only"
    return "several_clear"


def run_alpha_grid(
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
                "label": f"seed_{torch_seed}_alpha_{alpha}",
                "kind": "sweep",
                "torch_seed": int(torch_seed),
                "alpha": float(alpha),
                **stats,
            }
            sweep_rows.append(row)
            deep_rows.append(row)
        chosen = select_risk_matched(sweep_rows, target_cvar, target_cost)
        selected_nets.append(nets_by_alpha[float(chosen["alpha"])])
        clears = seed_clears_champion(chosen, champion)
        deep_rows.append(
            {
                "label": f"seed_{torch_seed}_selected",
                "kind": "seed",
                "torch_seed": int(torch_seed),
                "alpha": chosen["alpha"],
                "mean_cost": chosen["mean_cost"],
                "cvar95": chosen["cvar95"],
                "pnl_mean": chosen["pnl_mean"],
                "clears_champion": clears,
                "selection": "min cost among cvar<=champion else min cvar",
            }
        )
    ensemble_stats = pooled_deep_rollout_metrics(cfg, selected_nets, device)
    deep_rows.append(
        {
            "label": "ensemble_selected",
            "kind": "ensemble",
            "torch_seeds": list(hyper.torch_seeds),
            **ensemble_stats,
        }
    )
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    sweep_clear_alphas = {
        float(r["alpha"])
        for r in deep_rows
        if r.get("kind") == "sweep" and seed_clears_champion(r, champion)
    }
    return {
        "alpha_grid": list(alphas),
        "deep": deep_rows,
        "seed_rows": seed_rows,
        "sweep_alphas_that_clear": sorted(sweep_clear_alphas),
    }


def save_plot(
    classical_rows: list[dict],
    champion: dict,
    deep_rows: list[dict],
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
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
    sweep = [r for r in deep_rows if r.get("kind") == "sweep"]
    if sweep:
        ax.scatter(
            [r["mean_cost"] for r in sweep],
            [r["cvar95"] for r in sweep],
            c="tab:blue",
            alpha=0.25,
            s=28,
            label="α sweep",
            zorder=1,
        )
    for row in deep_rows:
        if row.get("kind") != "seed":
            continue
        clears = row.get("clears_champion", False)
        ax.scatter(
            [row["mean_cost"]],
            [row["cvar95"]],
            c="tab:green" if clears else "tab:blue",
            marker="D",
            s=80,
            zorder=4,
        )
    ax.scatter([], [], c="tab:green", marker="D", s=80, label="selected clears")
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-L: liquid 10 bp risk-matched α grid")
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
    hyper = replace(Hyper(), torch_seeds=EXPANDED_TORCH_SEEDS)
    alphas = QUICK_ALPHAS if args.quick else COMPACT_ALPHAS
    if args.quick:
        base_cfg, hyper = B._quick_setup(base_cfg, hyper)
        base_cfg = replace(base_cfg, cost=LIQUID_COST)

    started = time.time()
    classical_rows, champion = sweep_classical_hull(base_cfg)

    frozen_cb = B.load_classical_best()
    frozen_spec = strategy_spec({**frozen_cb, "label": "classical_best"})
    champion_matches_frozen_cb_spec = champion["spec"] == frozen_spec

    arm = run_alpha_grid(base_cfg, hyper, device, champion, alphas)
    seed_rows = arm["seed_rows"]
    verdict = verdict_alpha_robustness(seed_rows, champion)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "research_question": "rq_l_liquid_alpha_grid_robustness",
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
        "alpha_grid": arm["alpha_grid"],
        "deep": arm["deep"],
        "acceptance": {
            "verdict_values": ("a05_only", "several_clear", "none"),
            "clear_rule": (
                "per torch seed, selected OOS point: mean_cost < champion.mean_cost "
                "AND cvar95 <= champion.cvar95"
            ),
            "selection_rule": (
                "min mean_cost among sweep points with cvar95 <= champion.cvar95; "
                "else min cvar95"
            ),
            "a05_only_rule": (
                "at least one selected seed clears AND every clearing selected seed "
                "has training alpha == 0.05 (equivalently: distinct clearing alphas "
                "subset is exactly {0.05})"
            ),
            "several_clear_rule": (
                "at least one selected seed clears AND two or more distinct training "
                "alphas appear among clearing selected seeds, OR clearing alphas "
                "include any value other than 0.05"
            ),
            "none_rule": "no selected seed clears the champion",
            "champion": {
                "mean_cost": champion["mean_cost"],
                "cvar95": champion["cvar95"],
            },
            "sweep_alphas_that_clear": arm["sweep_alphas_that_clear"],
        },
        "audit": {
            "loaded_classical_best_json": False,
            "champion_from_val_sharpe_sweep": True,
            "champion_matches_frozen_cb_spec": champion_matches_frozen_cb_spec,
            "frozen_cb_spec_for_audit_only": frozen_spec,
            "expanded_torch_seeds_same_as_rq_i": list(EXPANDED_TORCH_SEEDS),
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("liquid_alpha_quick.json" if args.quick else "liquid_alpha.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("liquid_alpha_quick.png" if args.quick else "liquid_alpha.png")
        save_plot(classical_rows, champion, arm["deep"], plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
