"""Exploration: the training objective chooses between hedging and harvesting.

The comparison's headline is that the learned hedger's high-cost edge is really
premium harvesting. If that is the mechanism, it should be tunable through the
one dial that decides which part of the P&L distribution the network cares
about: the CVaR tail fraction ``alpha``.

``alpha -> 0`` trains on the worst tail, which is the seller's true concern and
should produce a real hedge. ``alpha -> 1`` trains on the mean, which should let
the network hold a small position and collect the volatility risk premium. This
script sweeps ``alpha`` on two markets and two cost levels and reports the
Sharpe, the P&L mean and standard deviation, and the turnover, so the shift from
hedging to harvesting is visible in the positions rather than inferred.

The side run never touches the holdout pool; it selects nothing and simply
tabulates, so ``train``/``val`` seeds are enough.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import deep, metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "exploration"
ALPHAS = (0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
CELLS = (("heston", 0.002), ("heston", 0.01), ("jd", 0.002), ("jd", 0.01))


def ensemble_turnover(nets, paths, cfg, device="cpu"):
    """Mean per-path traded notional for the averaged ensemble.

    Turnover is the mechanism variable the P&L hides: two strategies can post
    the same Sharpe while one trades ten times as much. Averaging the positions
    across nets mirrors ``NetworkStrategy`` averaging the P&L, so the number
    describes the deployed ensemble rather than a single lucky member.
    """
    tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
    positions = [
        deep.hedge_positions(tensor, net, cfg.K, cfg.dt, cfg.T) for net in nets
    ]
    pos = np.mean(positions, axis=0)  # (n_paths, n_steps)
    traded = np.diff(pos, axis=1, prepend=np.zeros((pos.shape[0], 1)))
    notional = np.abs(traded) * paths[:, :-1]
    turnover = notional.sum(axis=1) + np.abs(pos[:, -1]) * paths[:, -1]
    return float(turnover.mean())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = B.BenchConfig()
    rows = []
    for model, cost in CELLS:
        cfg = replace(base, model=model, cost=float(cost))
        classical = B.ClassicalStrategy(cfg, {"strategy": "time", "frequency": 1,
                                              "delta_band": None, "price_band": None})
        # Cache the val paths and the classical P&L once per cell; the sweep
        # then only varies the network.
        val = [cfg.simulate(model, s, cfg.n_val_paths) for s in cfg.val_seeds]
        class_sharpes = [metrics.sharpe(classical.pnl(p)) for p in val]
        for alpha in ALPHAS:
            hyper = Hyper(
                epochs=60, lr=1e-3, hidden=64, depth=3, alpha=alpha,
                torch_seeds=(0, 1, 2),
            )
            nets, _ = B.train_ensemble(cfg, hyper, device="cuda")
            learned = B.NetworkStrategy(cfg, nets, device="cuda")
            sharpes, means, stds, turns = [], [], [], []
            for p in val:
                pnl = learned.pnl(p)
                s = metrics.summarize(pnl)
                sharpes.append(s["sharpe"])
                means.append(s["mean"])
                stds.append(s["std"])
                turns.append(ensemble_turnover(nets, p, cfg, device="cuda"))
            rows.append(
                {
                    "model": model,
                    "cost": float(cost),
                    "alpha": alpha,
                    "deep_sharpe": float(np.mean(sharpes)),
                    "deep_mean": float(np.mean(means)),
                    "deep_std": float(np.mean(stds)),
                    "deep_turnover": float(np.mean(turns)),
                    "classical_sharpe": float(np.mean(class_sharpes)),
                }
            )
            print(
                f"{model:>6} {cost:<6} alpha={alpha:<5} "
                f"sharpe={rows[-1]['deep_sharpe']:+.3f} "
                f"mean={rows[-1]['deep_mean']:+.4f} std={rows[-1]['deep_std']:.3f} "
                f"turn={rows[-1]['deep_turnover']:.3f}"
            )

    (OUT / "objective_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, (model, cost) in zip(axes.ravel(), CELLS):
        sub = [r for r in rows if r["model"] == model and r["cost"] == cost]
        ax.plot([r["alpha"] for r in sub], [r["deep_sharpe"] for r in sub],
                marker="o", label="learned Sharpe")
        ax.axhline(sub[0]["classical_sharpe"], ls="--", color="black",
                   label="BS delta every step")
        ax.set_title(f"{model} @ {cost * 1e4:.0f} bp")
        ax.set_xlabel("CVaR alpha (1 = mean objective)")
        ax.set_ylabel("Sharpe")
        ax.legend(fontsize=7)
    fig.suptitle("Objective sweep: tail objective versus mean objective")
    fig.tight_layout()
    path = OUT / "objective_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {path.relative_to(ROOT)} and objective_sweep.json")


if __name__ == "__main__":
    main()
