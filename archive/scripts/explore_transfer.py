"""Exploration: does a learned hedge transfer across market models?

The comparison trains and tests inside one market. A strategy that only works on
the market it was trained on is a weaker result than one that shares a single
policy across the four processes, so this asks the broader question directly:
train the hedger on each model, then score the *same* network on all four.

The network never observes the volatility state or the model, so there is a
theoretical reason to expect decent transfer: its input is (moneyness, time,
position). The counter-hypothesis is that it learns model-specific
under-hedging (for example ignoring jump risk) that travels badly. The diagonal
of the matrix is the from-scratch result already reported; off-diagonal entries
below the diagonal measure how much of the edge is market-specific.

Only ``train`` and ``val`` seeds are used; the holdout pool is never touched.
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

from hedging import bench as B  # noqa: E402
from hedging import metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "exploration"
MODELS = ("gbm", "jd", "heston", "svj")
COST = 0.002


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = replace(B.BenchConfig(), cost=COST)
    hyper = Hyper(epochs=80, hidden=64, depth=3, alpha=0.65, torch_seeds=(0, 1, 2))

    # Pre-simulate every target's val paths once; the 16 cells then share them,
    # which is what makes the matrix comparable across rows.
    target_paths = {
        target: [base.simulate(target, s, base.n_val_paths) for s in base.val_seeds]
        for target in MODELS
    }
    target_delta = {}
    for target in MODELS:
        cfg = replace(base, model=target)
        delta = B.ClassicalStrategy(
            cfg, {"strategy": "time", "frequency": 1, "delta_band": None,
                  "price_band": None}
        )
        target_delta[target] = float(
            np.mean([metrics.sharpe(delta.pnl(p)) for p in target_paths[target]])
        )

    matrix: dict[str, dict[str, float]] = {}
    for source in MODELS:
        cfg_src = replace(base, model=source)
        nets, _ = B.train_ensemble(cfg_src, hyper, device="cuda")
        row = {}
        for target in MODELS:
            cfg_tgt = replace(base, model=target)
            learned = B.NetworkStrategy(cfg_tgt, nets, device="cuda")
            row[target] = float(
                np.mean(
                    [metrics.sharpe(learned.pnl(p)) for p in target_paths[target]]
                )
            )
        matrix[source] = row
        print(
            f"trained on {source:>6}: "
            + "  ".join(f"{t}={row[t]:+.3f}" for t in MODELS)
        )

    print("\nplain BS delta reference:")
    print("  " + "  ".join(f"{t}={target_delta[t]:+.3f}" for t in MODELS))

    (OUT / "transfer_matrix.json").write_text(
        json.dumps({"matrix": matrix, "bs_delta": target_delta}, indent=2) + "\n"
    )

    grid = np.array([[matrix[s][t] for t in MODELS] for s in MODELS])
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(grid, cmap="RdYlGn")
    ax.set_xticks(range(len(MODELS)), MODELS)
    ax.set_yticks(range(len(MODELS)), MODELS)
    ax.set_xlabel("test market")
    ax.set_ylabel("train market")
    for i in range(len(MODELS)):
        for j in range(len(MODELS)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Sharpe (val, 20 bp)")
    ax.set_title("Cross-model transfer of the learned hedger")
    fig.tight_layout()
    path = OUT / "transfer_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {path.relative_to(ROOT)} and transfer_matrix.json")


if __name__ == "__main__":
    main()
