"""RQ-F: deep hedging with a train-time CVaR95 cap vs ``classical_best``.

RQ-D sweeps the training ``alpha`` dial in ``deep.cvar_loss``; here ``Hyper().alpha``
stays at its default while a **separate** Lagrangian term pushes **in-batch** tail
loss toward the OOS ``classical_best`` CVaR95 level. Scoring still uses only pooled
validation ``hedge.rollout`` metrics (mean cost and ``metrics.cvar95``).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import deep  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import (  # noqa: E402
    classical_candidates,
    pooled_deep_rollout_metrics,
    pooled_rollout_metrics,
    resolve_device,
    strategy_spec,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

# Dual ascent step on the Lagrange multiplier; large enough to react when batches
# breach the classical tail budget, small enough not to oscillate across epochs.
DUAL_LR = 0.75


def batch_cvar95(pnl: torch.Tensor) -> torch.Tensor:
    """5% tail loss on a mini-batch, aligned with ``hedging.metrics.cvar95``."""
    cut = torch.quantile(pnl, 0.05)
    tail = pnl[pnl <= cut]
    if tail.numel() == 0:
        return pnl.new_zeros(())
    return -tail.mean()


def train_cvar_capped_hedger(
    paths,
    K,
    kind,
    premium,
    cost,
    dt,
    T,
    *,
    target_cvar: float,
    epochs: int = 60,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden: int = 64,
    depth: int = 2,
    alpha: float = 0.05,
    seed: int = 0,
    device: str = "cpu",
    verbose: bool = False,
    dual_lr: float = DUAL_LR,
    features=deep.DEFAULT_FEATURES,
) -> tuple[torch.nn.Module, list[float], dict]:
    """Fit with base CVaR objective plus a Lagrange penalty on tail excess vs target.

    The multiplier is updated on epoch-mean constraint slack so training chases the
    same 5% tail statistic used in OOS rollout scoring, without touching ``alpha``.
    """
    torch.manual_seed(seed)
    net = deep.HedgeNet(hidden=hidden, depth=depth, in_dim=len(features)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = paths.shape[0]
    history: list[float] = []
    lam = 0.0
    constraint_log: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        batches = 0
        epoch_excess: list[torch.Tensor] = []
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pnl = deep.hedge_pnl(
                paths[idx], net, K, kind, premium, cost, dt, T, features
            )
            base = deep.cvar_loss(pnl, alpha)
            tail = batch_cvar95(pnl)
            excess = F.relu(tail - target_cvar)
            loss = base + lam * excess
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach())
            batches += 1
            epoch_excess.append(excess.detach())
        history.append(epoch_loss / max(batches, 1))
        mean_excess = torch.stack(epoch_excess).mean().item() if epoch_excess else 0.0
        constraint_log.append(mean_excess)
        lam = max(0.0, lam + dual_lr * mean_excess)
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"epoch {epoch + 1:>3}  loss {history[-1]: .6f}  "
                f"lam {lam: .4f}  mean_excess {mean_excess: .6f}"
            )
    meta = {
        "target_cvar": float(target_cvar),
        "final_lam": float(lam),
        "mean_epoch_excess": float(np.mean(constraint_log)) if constraint_log else 0.0,
        "constraint": "lagrange_excess_cvar95",
        "dual_lr": float(dual_lr),
        "note": "NOT rq_d_alpha_dial; alpha is Hyper default only",
    }
    return net, history, meta


def train_seed_net(
    cfg: B.BenchConfig,
    hyper: Hyper,
    torch_seed: int,
    target_cvar: float,
    device: str,
) -> tuple:
    paths = B.training_paths(cfg)
    tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
    return train_cvar_capped_hedger(
        tensor,
        cfg.K,
        cfg.kind,
        cfg.premium,
        cfg.cost,
        cfg.dt,
        cfg.T,
        target_cvar=target_cvar,
        epochs=hyper.epochs,
        batch_size=hyper.batch_size,
        lr=hyper.lr,
        hidden=hyper.hidden,
        depth=hyper.depth,
        alpha=hyper.alpha,
        seed=torch_seed,
        device=device,
    )


def acceptance_verdict(deep_rows: list[dict], best_row: dict) -> str:
    """``clears`` iff any training seed beats baseline on cost at no worse tail."""
    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    if any(
        row["mean_cost"] < best_row["mean_cost"]
        and row["cvar95"] <= best_row["cvar95"]
        for row in seed_rows
    ):
        return "clears"
    return "fails"


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
    for row in deep_rows:
        if row.get("kind") == "seed":
            clears = (
                row["mean_cost"] < best_row["mean_cost"]
                and row["cvar95"] <= best_row["cvar95"]
            )
            ax.scatter(
                [row["mean_cost"]],
                [row["cvar95"]],
                c="tab:green" if clears else "tab:blue",
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
                label="CVaR-capped ensemble",
                zorder=5,
            )
    ax.set_xlabel("mean path proportional cost")
    ax.set_ylabel("CVaR95 (tail loss)")
    ax.set_title("RQ-F: train-time CVaR cap vs classical_best")
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
    target_cvar = best_row["cvar95"]
    target_cost = best_row["mean_cost"]

    deep_rows: list[dict] = []
    nets: list = []
    train_meta: list[dict] = []
    for torch_seed in hyper.torch_seeds:
        net, history, meta = train_seed_net(
            cfg, hyper, int(torch_seed), target_cvar, device
        )
        nets.append(net)
        train_meta.append(meta)
        stats = pooled_deep_rollout_metrics(cfg, [net], device)
        deep_rows.append(
            {
                "label": f"deep_seed_{torch_seed}_cvar_capped",
                "kind": "seed",
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
            "label": "deep_ensemble_cvar_capped",
            "kind": "ensemble",
            "torch_seeds": list(hyper.torch_seeds),
            **ensemble_stats,
        }
    )

    verdict = acceptance_verdict(deep_rows, best_row)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "research_question": "rq_f_hard_cvar_train_constraint",
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
        "train_constraint": {
            "mechanism": "lagrange_excess_batch_cvar95",
            "target_cvar95_oos_classical_best": target_cvar,
            "not_rq_d_alpha_sweep": True,
            "training_alpha_fixed": float(hyper.alpha),
        },
        "classical": classical_rows,
        "deep": deep_rows,
        "acceptance": {
            "clears": verdict == "clears",
            "fails": verdict == "fails",
            "rule": "any seed: OOS mean_cost < classical_best AND OOS cvar95 <= classical_best",
            "classical_best": {
                "mean_cost": target_cost,
                "cvar95": target_cvar,
            },
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("hard_cvar_quick.json" if args.quick else "hard_cvar.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("hard_cvar_quick.png" if args.quick else "hard_cvar.png")
        save_plot(classical_rows, deep_rows, best_row, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
