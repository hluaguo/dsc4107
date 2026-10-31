"""Exploration: which observables do the learned hedge and the band behaviour need?

The report argues the network can express band-like behaviour only because the
current position is part of its state. That is testable by retraining with one
observable removed at a time. Dropping ``pos`` should force a fresh target every
step (no band), raising turnover; dropping ``time`` removes the schedule;
dropping ``log_moneyness`` leaves the policy unable to see the moneyness it is
supposed to track.

The ablation uses the additive ``features`` argument added to ``hedging/deep.py``;
the frozen benchmark and the default columns are untouched.
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

OUT = ROOT / "results" / "exploration"
FEATURE_SETS = {
    "full": ("log_moneyness", "time", "pos"),
    "no_pos": ("log_moneyness", "time"),
    "no_time": ("log_moneyness", "pos"),
    "no_moneyness": ("time", "pos"),
}
CELLS = (("heston", 0.002), ("heston", 0.01))
EPOCHS = 60


def train_local(paths, cfg, features, seed):
    """Train one net with an explicit feature set.

    ``bench.train_ensemble`` does not take a feature set, and threading one
    through the frozen benchmark is off limits, so the ablation trains through
    ``deep.train_hedger`` directly. The training setup is otherwise identical.
    """
    tensor = torch.as_tensor(paths, dtype=torch.float32, device="cuda")
    return deep.train_hedger(
        tensor,
        cfg.K,
        cfg.kind,
        cfg.premium,
        cfg.cost,
        cfg.dt,
        cfg.T,
        epochs=EPOCHS,
        batch_size=1024,
        lr=1e-3,
        hidden=64,
        depth=3,
        alpha=0.65,
        seed=seed,
        device="cuda",
        features=features,
    )[0]


def turnover(nets, paths, cfg, features):
    tensor = torch.as_tensor(paths, dtype=torch.float32, device="cuda")
    positions = [
        deep.hedge_positions(tensor, net, cfg.K, cfg.dt, cfg.T, features)
        for net in nets
    ]
    pos = np.mean(positions, axis=0)
    traded = np.diff(pos, axis=1, prepend=np.zeros((pos.shape[0], 1)))
    return float(
        (np.abs(traded) * paths[:, :-1]).sum(axis=1).mean()
        + (np.abs(pos[:, -1]) * paths[:, -1]).mean()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = B.BenchConfig()
    rows = []
    for model, cost in CELLS:
        cfg = replace(base, model=model, cost=float(cost))
        train = B.training_paths(cfg)
        val = [cfg.simulate(model, s, cfg.n_val_paths) for s in cfg.val_seeds]
        for name, features in FEATURE_SETS.items():
            nets = [train_local(train, cfg, features, s) for s in (0, 1, 2)]
            sharpes, turns = [], []
            for p in val:
                tensor = torch.as_tensor(p, dtype=torch.float32, device="cuda")
                pnl = np.mean(
                    [
                        deep.evaluate_network(
                            tensor, net, cfg.K, cfg.kind, cfg.premium, cfg.cost,
                            cfg.dt, cfg.T, features=features,
                        )
                        for net in nets
                    ],
                    axis=0,
                )
                sharpes.append(metrics.sharpe(pnl))
                turns.append(turnover(nets, p, cfg, features))
            rows.append(
                {
                    "model": model,
                    "cost": float(cost),
                    "features": name,
                    "deep_sharpe": float(np.mean(sharpes)),
                    "turnover": float(np.mean(turns)),
                }
            )
            print(
                f"{model:>6} {cost:<6} {name:>12} "
                f"sharpe={rows[-1]['deep_sharpe']:+.3f} "
                f"turnover={rows[-1]['turnover']:.3f}"
            )

    (OUT / "feature_ablation.json").write_text(json.dumps(rows, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (model, cost) in zip(axes, CELLS):
        sub = [r for r in rows if r["model"] == model and r["cost"] == cost]
        ax.bar([r["features"] for r in sub], [r["deep_sharpe"] for r in sub])
        ax.set_title(f"{model} @ {cost * 1e4:.0f} bp")
        ax.set_ylabel("Sharpe (val)")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Feature ablation of the learned hedger")
    fig.tight_layout()
    path = OUT / "feature_ablation.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {path.relative_to(ROOT)} and feature_ablation.json")


if __name__ == "__main__":
    main()
