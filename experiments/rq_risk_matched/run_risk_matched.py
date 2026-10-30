"""RQ-D: risk-matched deep hedging vs the classical cost–CVaR hull on the frozen cell.

RQ-A showed the default deep recipe sits inside the hull on cost but above the
pinned baseline on tail risk. This run asks whether an explicit risk-matching
knob — a CVaR-tail ``alpha`` sweep per torch seed — can land at or below
``classical_best`` CVaR95 without paying more mean proportional cost than the
band champion. Scoring still uses the shared validation pool and the same
``hedge.rollout`` ledger as RQ-A; only the training objective grid differs.
"""

from __future__ import annotations

import argparse
import json
import math
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
import torch  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import deep  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import (  # noqa: E402
    classical_candidates,
    pooled_deep_rollout_metrics,
    pooled_rollout_metrics,
    resolve_device,
    strategy_spec,
    weakly_dominated,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

# Lower ``alpha`` in ``deep.cvar_loss`` weights harsher tails; the sweep spans
# from seller-style 5% CVaR training toward the looser mean-heavy default.
FULL_ALPHAS = (0.05, 0.12, 0.20, 0.30, 0.40, 0.50, 0.65)
QUICK_ALPHAS = (0.05, 0.35)


def train_single_net(
    cfg: B.BenchConfig, hyper: Hyper, torch_seed: int, device: str
) -> tuple:
    """One member of the ensemble so each seed can carry its own ``alpha`` choice."""
    paths = B.training_paths(cfg)
    tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
    net, history = deep.train_hedger(
        tensor,
        cfg.K,
        cfg.kind,
        cfg.premium,
        cfg.cost,
        cfg.dt,
        cfg.T,
        epochs=hyper.epochs,
        batch_size=hyper.batch_size,
        lr=hyper.lr,
        hidden=hyper.hidden,
        depth=hyper.depth,
        alpha=hyper.alpha,
        seed=torch_seed,
        device=device,
    )
    return net, history


def select_risk_matched(
    sweep_rows: list[dict], target_cvar: float, target_cost: float
) -> dict:
    """Pick the validation point that best answers the RQ at matched-or-better tail.

    Prefer any point with CVaR95 at or below the classical baseline, breaking ties
    on lower mean cost; if the sweep never reaches the tail target, take the
    lowest CVaR95 so the JSON still records how far risk matching fell short.
    """
    feasible = [r for r in sweep_rows if r["cvar95"] <= target_cvar]
    if feasible:
        return min(feasible, key=lambda r: (r["mean_cost"], r["cvar95"]))
    return min(sweep_rows, key=lambda r: (r["cvar95"], r["mean_cost"]))


def acceptance_verdict(
    deep_rows: list[dict], classical_rows: list[dict], best_row: dict
) -> str:
    front = classical_rows
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    sweep_and_headline = [
        r
        for r in deep_rows
        if r.get("kind") in ("sweep", "seed", "ensemble")
    ]
    n_seeds = len(seed_rows)
    need = math.ceil(2 * n_seeds / 3) if n_seeds else 2
    wins = sum(
        1
        for row in seed_rows
        if row["cvar95"] <= best_row["cvar95"]
        and row["mean_cost"] <= best_row["mean_cost"]
    )
    if wins >= need:
        return "improves"
    if sweep_and_headline and all(
        weakly_dominated(row, front) for row in sweep_and_headline
    ):
        return "nested"
    return "flat"


def save_plot(
    classical_rows: list[dict],
    deep_rows: list[dict],
    best_row: dict,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cx = [r["mean_cost"] for r in classical_rows]
    cy = [r["cvar95"] for r in classical_rows]
    ax.scatter(cx, cy, c="tab:orange", s=55, label="classical grid + best", zorder=2)
    ax.scatter(
        [best_row["mean_cost"]],
        [best_row["cvar95"]],
        c="tab:red",
        s=120,
        marker="*",
        label="classical_best target",
        zorder=3,
    )
    sweep = [r for r in deep_rows if r.get("kind") == "sweep"]
    if sweep:
        ax.scatter(
            [r["mean_cost"] for r in sweep],
            [r["cvar95"] for r in sweep],
            c="tab:blue",
            alpha=0.35,
            s=35,
            label="deep α sweep",
            zorder=1,
        )
    for row in deep_rows:
        if row.get("kind") == "seed":
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:blue",
                marker="D",
                s=90,
                zorder=4,
            )
        elif row.get("kind") == "ensemble":
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:purple",
                marker="s",
                s=100,
                label="risk-matched ensemble",
                zorder=5,
            )
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-D: risk-matched deep vs classical hull")
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
    alphas = QUICK_ALPHAS if args.quick else FULL_ALPHAS
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
    target_cvar = best_row["cvar95"]
    target_cost = best_row["mean_cost"]

    deep_rows: list[dict] = []
    selected_nets: list = []
    for torch_seed in hyper.torch_seeds:
        sweep_rows: list[dict] = []
        nets_by_alpha: dict[float, torch.nn.Module] = {}
        for alpha in alphas:
            run_hyper = replace(hyper, alpha=float(alpha))
            net, _ = train_single_net(cfg, run_hyper, int(torch_seed), device)
            nets_by_alpha[float(alpha)] = net
            stats = pooled_deep_rollout_metrics(cfg, [net], device)
            row = {
                "label": f"deep_seed_{torch_seed}_alpha_{alpha}",
                "kind": "sweep",
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
                "label": f"deep_seed_{torch_seed}_selected",
                "kind": "seed",
                "torch_seed": int(torch_seed),
                "alpha": chosen["alpha"],
                "mean_cost": chosen["mean_cost"],
                "cvar95": chosen["cvar95"],
                "pnl_mean": chosen["pnl_mean"],
                "selection": "min cost among cvar<=baseline else min cvar",
            }
        )

    ensemble_stats = pooled_deep_rollout_metrics(cfg, selected_nets, device)
    deep_rows.append(
        {
            "label": "deep_ensemble_selected",
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
        "research_question": "rq_d_risk_matched",
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
        "classical": classical_rows,
        "deep": deep_rows,
        "acceptance": {
            "improves": verdict == "improves",
            "nested": verdict == "nested",
            "flat": verdict == "flat",
            "classical_best": {
                "mean_cost": target_cost,
                "cvar95": target_cvar,
            },
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("risk_matched_quick.json" if args.quick else "risk_matched.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / (
            "risk_matched_quick.png" if args.quick else "risk_matched.png"
        )
        save_plot(classical_rows, deep_rows, best_row, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
