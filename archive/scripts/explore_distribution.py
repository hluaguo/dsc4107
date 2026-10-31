"""Exploration: the terminal P&L distribution behind the Sharpe comparison.

The plan asks for the loss distribution alongside the mean, volatility and
Sharpe, because two strategies can post similar risk-adjusted numbers with very
different tails. This trains the frozen benchmark (Heston, 20 bp, default
budget), pools the validation-path P&L for the learned hedger and the classical
band, and plots both distributions with their 5% CVaR marks.

The plot is the visual counterpart to the standard-deviation guard in
``comparison_study.py``: a learned curve that is shifted right but much wider is
harvesting the premium, not hedging better.
"""

from __future__ import annotations

import json
import pathlib
import sys

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


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = B.BenchConfig()
    classical = B.ClassicalStrategy(cfg, B.load_classical_best())
    nets, _ = B.train_ensemble(cfg, Hyper(), device="cuda")
    learned = B.NetworkStrategy(cfg, nets, device="cuda")

    deep, clas = [], []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        deep.append(learned.pnl(paths))
        clas.append(classical.pnl(paths))
    deep = np.concatenate(deep)
    clas = np.concatenate(clas)

    summary = {}
    for name, pnl in (("learned", deep), ("band", clas)):
        stats = metrics.summarize(pnl)
        stats["cvar95_quantile"] = float(np.quantile(pnl, 0.05))
        summary[name] = stats
    (OUT / "distribution.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lo = float(min(deep.min(), clas.min()))
    hi = float(max(deep.max(), clas.max()))
    bins = np.linspace(lo, hi, 160)
    for ax, (name, pnl), color in (
        (axes[0], ("learned", deep), "tab:blue"),
        (axes[1], ("band", clas), "tab:orange"),
    ):
        ax.hist(pnl, bins=bins, density=True, color=color, alpha=0.75)
        q = np.quantile(pnl, 0.05)
        ax.axvline(q, color="black", ls="--", lw=1.0)
        ax.axvline(pnl.mean(), color="black", lw=1.0)
        ax.set_title(f"{name}: mean {pnl.mean():.3f}, std {pnl.std():.3f}, "
                     f"CVaR {metrics.cvar95(pnl):.3f}")
        ax.set_xlabel("terminal P&L")
        ax.set_ylabel("density")
    fig.suptitle("Terminal P&L distribution, Heston 20 bp (dashed = 5% quantile)")
    fig.tight_layout()
    path = OUT / "distribution.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"learned: {summary['learned']}")
    print(f"band:    {summary['band']}")
    print(f"wrote {path.relative_to(ROOT)} and distribution.json")


if __name__ == "__main__":
    main()
