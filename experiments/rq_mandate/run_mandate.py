"""RQ-B: mandate / activity floor vs classical_best on the frozen cell.

Deep hedgers are trained with a penalty when in-sample activity (opening |δ| and
path turnover) sits below the OOS levels of the pinned classical band champion.
Before scoring, position paths are rescaled through the same mandate map so the
policy handed to ``hedge.rollout`` cannot trade quieter than that baseline on
either axis — the question is whether any torch seed then clears classical_best
on mean cost and CVaR95, not whether an unconstrained net happens to hedge less.
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
from torch import nn  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import deep, hedge, metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402
from experiments.rq_a.run_pareto import (  # noqa: E402
    _ArrayPositionFn,
    classical_candidates,
    pooled_rollout_metrics,
    resolve_device,
    strategy_spec,
)

OUT = pathlib.Path(__file__).resolve().parent / "results"

# Turnover and tail loss live on different scales; without a heavy weight the
# CVaR term keeps shrinking activity because cheaper quiet policies win in-sample.
MANDATE_ACTIVITY_WEIGHT = 450.0


def turnover_from_position_path(targets: np.ndarray) -> np.ndarray:
    """Per-path turnover in position units, matching ``hedge.rollout``."""
    n_paths, n_steps = targets.shape
    pos = np.zeros(n_paths)
    tot = np.zeros(n_paths)
    for j in range(n_steps):
        new = targets[:, j]
        tot += np.abs(new - pos)
        pos = new
    tot += np.abs(pos)
    return tot


def enforce_mandate_positions(
    targets: np.ndarray, floor_init: float, floor_turn: float
) -> np.ndarray:
    """Rescale trades so opening |δ| and rollout turnover meet the classical floor."""
    n_paths, n_steps = targets.shape
    out = targets.copy()
    signs = np.sign(out[:, 0])
    signs[signs == 0.0] = 1.0
    out[:, 0] = signs * np.maximum(np.abs(out[:, 0]), floor_init)

    if float(turnover_from_position_path(out).mean()) >= floor_turn:
        return out

    lo, hi = 1.0, 10.0
    best = out
    for _ in range(48):
        mid = 0.5 * (lo + hi)
        scaled = np.empty_like(out)
        scaled[:, 0] = out[:, 0]
        for j in range(1, n_steps):
            scaled[:, j] = scaled[:, j - 1] + mid * (out[:, j] - out[:, j - 1])
        if float(turnover_from_position_path(scaled).mean()) >= floor_turn:
            best = scaled
            hi = mid
        else:
            lo = mid
    return best


def pooled_classical_activity(cfg: B.BenchConfig, spec: dict) -> dict:
    """OOS activity stats for one classical spec on the shared validation pool."""
    init_chunks: list[np.ndarray] = []
    turn_chunks: list[np.ndarray] = []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        fn = B.make_classical_position_fn(cfg, spec)
        spot0 = paths[:, 0]
        pos0 = fn(0, spot0, np.zeros(paths.shape[0]))
        init_chunks.append(np.abs(pos0))
        result = hedge.rollout(
            paths, fn, cfg.K, cfg.kind, cfg.premium, cfg.cost, cfg.dt
        )
        turn_chunks.append(result.turnover)
    inits = np.concatenate(init_chunks)
    turns = np.concatenate(turn_chunks)
    return {
        "mean_initial_abs_delta": float(inits.mean()),
        "mean_turnover": float(turns.mean()),
    }


def pooled_mandate_deep_metrics(
    cfg: B.BenchConfig,
    net,
    device: str,
    floor_init: float,
    floor_turn: float,
) -> dict:
    """Score a mandate-rescaled deep policy through the shared rollout ledger."""
    cost_chunks: list[np.ndarray] = []
    pnl_chunks: list[np.ndarray] = []
    init_chunks: list[np.ndarray] = []
    turn_chunks: list[np.ndarray] = []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
        targets = deep.hedge_positions(tensor, net, cfg.K, cfg.dt, cfg.T)
        targets = enforce_mandate_positions(
            targets, floor_init=floor_init, floor_turn=floor_turn
        )
        init_chunks.append(np.abs(targets[:, 0]))
        result = hedge.rollout(
            paths,
            _ArrayPositionFn(targets),
            cfg.K,
            cfg.kind,
            cfg.premium,
            cfg.cost,
            cfg.dt,
        )
        cost_chunks.append(result.costs)
        pnl_chunks.append(result.pnl)
        turn_chunks.append(result.turnover)
    costs = np.concatenate(cost_chunks)
    pnl = np.concatenate(pnl_chunks)
    inits = np.concatenate(init_chunks)
    turns = np.concatenate(turn_chunks)
    floor_met = (
        float(inits.mean()) >= floor_init - 1e-9
        and float(turns.mean()) >= floor_turn - 1e-9
    )
    return {
        "mean_cost": float(costs.mean()),
        "cvar95": metrics.cvar95(pnl),
        "pnl_mean": float(pnl.mean()),
        "mean_initial_abs_delta": float(inits.mean()),
        "mean_turnover": float(turns.mean()),
        "floor_met": floor_met,
    }


def mandate_hedge_pnl(
    paths,
    net,
    K,
    kind,
    premium,
    cost,
    dt,
    T,
    *,
    floor_init: float,
    floor_turn: float,
    features=deep.DEFAULT_FEATURES,
):
    """Differentiable rollout used only inside mandate training."""
    n_steps = paths.shape[1] - 1
    pos = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    cash = torch.full(
        (paths.shape[0],), premium, device=paths.device, dtype=paths.dtype
    )
    turnover = torch.zeros(paths.shape[0], device=paths.device, dtype=paths.dtype)
    initial_abs: torch.Tensor | None = None
    for j in range(n_steps):
        spot = paths[:, j]
        new_pos = net(deep._features(spot, j, dt, T, K, pos, features))
        if j == 0:
            initial_abs = torch.abs(new_pos)
        traded = new_pos - pos
        turnover = turnover + torch.abs(traded)
        cash = cash - traded * spot - cost * torch.abs(traded) * spot
        pos = new_pos

    terminal = paths[:, -1]
    if kind == "call":
        payoff = torch.clamp(terminal - K, min=0.0)
    elif kind == "put":
        payoff = torch.clamp(K - terminal, min=0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    close_cost = cost * torch.abs(pos) * terminal
    turnover = turnover + torch.abs(pos)
    pnl = cash + pos * terminal - close_cost - payoff
    assert initial_abs is not None
    init_gap = torch.relu(floor_init - initial_abs.mean())
    turn_gap = torch.relu(floor_turn - turnover.mean())
    activity_penalty = init_gap.square() + turn_gap.square()
    return pnl, activity_penalty


def train_mandate_hedger(
    paths,
    K,
    kind,
    premium,
    cost,
    dt,
    T,
    *,
    floor_init: float,
    floor_turn: float,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    depth: int,
    alpha: float,
    seed: int,
    device: str,
    activity_weight: float = MANDATE_ACTIVITY_WEIGHT,
    features=deep.DEFAULT_FEATURES,
):
    """Fit a hedger whose in-sample activity is pushed toward the classical OOS floor."""
    torch.manual_seed(seed)
    net = deep.HedgeNet(hidden=hidden, depth=depth, in_dim=len(features)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = paths.shape[0]
    history: list[float] = []
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            pnl, activity_penalty = mandate_hedge_pnl(
                paths[idx],
                net,
                K,
                kind,
                premium,
                cost,
                dt,
                T,
                floor_init=floor_init,
                floor_turn=floor_turn,
                features=features,
            )
            loss = deep.cvar_loss(pnl, alpha) + activity_weight * activity_penalty
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            epoch_loss += float(loss.detach())
            batches += 1
        history.append(epoch_loss / max(batches, 1))
    return net, history


def acceptance_verdict(seed_rows: list[dict], best_row: dict) -> str:
    """Lead-locked readout: any-seed clears, interesting tail-only win, else fail."""
    cb_cost = best_row["mean_cost"]
    cb_cvar = best_row["cvar95"]

    def clears(row: dict) -> bool:
        return (
            row.get("floor_met")
            and row["mean_cost"] < cb_cost
            and row["cvar95"] <= cb_cvar
        )

    if any(clears(r) for r in seed_rows):
        return "clears"
    if any(
        r.get("floor_met") and r["cvar95"] <= cb_cvar and r["mean_cost"] >= cb_cost
        for r in seed_rows
    ):
        return "interesting"
    return "fail"


def save_plot(
    classical_rows: list[dict],
    deep_rows: list[dict],
    best_row: dict,
    floor_row: dict,
    path: pathlib.Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    cx = [r["mean_cost"] for r in classical_rows]
    cy = [r["cvar95"] for r in classical_rows]
    ax.scatter(cx, cy, c="tab:orange", s=55, label="classical grid + best", zorder=2)
    ax.scatter(
        [best_row["mean_cost"]],
        [best_row["cvar95"]],
        c="tab:red",
        s=120,
        marker="*",
        label="classical_best",
        zorder=3,
    )
    for row in deep_rows:
        if row.get("kind") != "seed":
            continue
        color = "tab:green" if row.get("floor_met") else "tab:gray"
        ax.scatter(
            [row["mean_cost"]],
            [row["cvar95"]],
            c=color,
            marker="D",
            s=90,
            zorder=4,
        )
    if deep_rows:
        ax.scatter([], [], c="tab:green", marker="D", label="mandate deep (floor met)")
        ax.scatter([], [], c="tab:gray", marker="D", label="mandate deep (floor miss)")
    ax.axhline(best_row["cvar95"], color="tab:red", ls=":", lw=0.8, alpha=0.6)
    ax.axvline(best_row["mean_cost"], color="tab:red", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("mean path proportional cost (OOS rollout)")
    ax.set_ylabel("CVaR95 tail loss (OOS rollout)")
    ax.set_title(
        "RQ-B: mandate floor (init |δ| & turnover ≥ classical_best)\n"
        f"CB turnover={floor_row['mean_turnover']:.3f}, init|δ|={floor_row['mean_initial_abs_delta']:.3f}"
    )
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
        activity = pooled_classical_activity(cfg, spec)
        classical_rows.append(
            {
                "label": cand["label"],
                "kind": "classical",
                "spec": spec,
                **stats,
                **activity,
            }
        )

    best_row = next(r for r in classical_rows if r["label"] == "classical_best")
    floor_init = best_row["mean_initial_abs_delta"]
    floor_turn = best_row["mean_turnover"]

    paths = B.training_paths(cfg)
    tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)

    deep_rows: list[dict] = []
    mandate_nets: list = []
    for torch_seed in hyper.torch_seeds:
        net, _history = train_mandate_hedger(
            tensor,
            cfg.K,
            cfg.kind,
            cfg.premium,
            cfg.cost,
            cfg.dt,
            cfg.T,
            floor_init=floor_init,
            floor_turn=floor_turn,
            epochs=hyper.epochs,
            batch_size=hyper.batch_size,
            lr=hyper.lr,
            hidden=hyper.hidden,
            depth=hyper.depth,
            alpha=hyper.alpha,
            seed=int(torch_seed),
            device=device,
        )
        mandate_nets.append(net)
        stats = pooled_mandate_deep_metrics(
            cfg, net, device, floor_init=floor_init, floor_turn=floor_turn
        )
        deep_rows.append(
            {
                "label": f"mandate_seed_{torch_seed}",
                "kind": "seed",
                "torch_seed": int(torch_seed),
                **stats,
            }
        )

    # Ensemble: mandate-map each member then average positions pathwise before rollout.
    cost_chunks: list[np.ndarray] = []
    pnl_chunks: list[np.ndarray] = []
    init_chunks: list[np.ndarray] = []
    turn_chunks: list[np.ndarray] = []
    for seed in cfg.val_seeds:
        paths_val = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        member_targets: list[np.ndarray] = []
        for net in mandate_nets:
            tensor_val = torch.as_tensor(
                paths_val, dtype=torch.float32, device=device
            )
            targets = deep.hedge_positions(tensor_val, net, cfg.K, cfg.dt, cfg.T)
            targets = enforce_mandate_positions(
                targets, floor_init=floor_init, floor_turn=floor_turn
            )
            member_targets.append(targets)
        avg_targets = np.mean(member_targets, axis=0)
        avg_targets = enforce_mandate_positions(
            avg_targets, floor_init=floor_init, floor_turn=floor_turn
        )
        init_chunks.append(np.abs(avg_targets[:, 0]))
        result = hedge.rollout(
            paths_val,
            _ArrayPositionFn(avg_targets),
            cfg.K,
            cfg.kind,
            cfg.premium,
            cfg.cost,
            cfg.dt,
        )
        cost_chunks.append(result.costs)
        pnl_chunks.append(result.pnl)
        turn_chunks.append(result.turnover)
    costs = np.concatenate(cost_chunks)
    pnl = np.concatenate(pnl_chunks)
    inits = np.concatenate(init_chunks)
    turns = np.concatenate(turn_chunks)
    ensemble_floor_met = (
        float(inits.mean()) >= floor_init - 1e-9
        and float(turns.mean()) >= floor_turn - 1e-9
    )
    deep_rows.append(
        {
            "label": "mandate_ensemble",
            "kind": "ensemble",
            "torch_seeds": list(hyper.torch_seeds),
            "mean_cost": float(costs.mean()),
            "cvar95": metrics.cvar95(pnl),
            "pnl_mean": float(pnl.mean()),
            "mean_initial_abs_delta": float(inits.mean()),
            "mean_turnover": float(turns.mean()),
            "floor_met": ensemble_floor_met,
        }
    )

    seed_rows = [r for r in deep_rows if r.get("kind") == "seed"]
    verdict = acceptance_verdict(seed_rows, best_row)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {
        "timestamp": stamp,
        "verdict": verdict,
        "research_question": "rq_b_mandate_activity_floor",
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
        "mandate": {
            "activity_weight": MANDATE_ACTIVITY_WEIGHT,
            "floor_source": "classical_best OOS on shared val pool",
            "mean_initial_abs_delta_floor": floor_init,
            "mean_turnover_floor": floor_turn,
            "definitions": {
                "mean_initial_abs_delta": (
                    "Mean over all validation paths of |δ| after the first "
                    "rebalance (j=0), before costs."
                ),
                "mean_turnover": (
                    "Mean path turnover from hedge.rollout: Σ|Δδ| over "
                    "decision steps plus terminal |δ| (position units)."
                ),
                "floor_met": (
                    "OOS pooled mean_initial_abs_delta ≥ floor and "
                    "mean_turnover ≥ floor after mandate position map."
                ),
            },
        },
        "classical": classical_rows,
        "deep_mandate": deep_rows,
        "acceptance": {
            "clears": verdict == "clears",
            "interesting": verdict == "interesting",
            "fail": verdict == "fail",
            "classical_best": {
                "mean_cost": best_row["mean_cost"],
                "cvar95": best_row["cvar95"],
                "mean_initial_abs_delta": floor_init,
                "mean_turnover": floor_turn,
            },
            "rules": {
                "clears": (
                    "Any seed: floor_met and mean_cost < CB and cvar95 ≤ CB"
                ),
                "interesting": (
                    "Floor met, CVaR95 ≤ CB, but mean_cost still ≥ CB (no seed clears)"
                ),
            },
        },
        "wall_time_s": time.time() - started,
    }
    json_path = OUT / ("mandate_quick.json" if args.quick else "mandate.json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"verdict={verdict}")
    print(f"wrote {json_path.relative_to(ROOT)}")

    if not args.no_plot:
        plot_path = OUT / ("mandate_quick.png" if args.quick else "mandate.png")
        save_plot(classical_rows, deep_rows, best_row, best_row, plot_path)
        print(f"wrote {plot_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
