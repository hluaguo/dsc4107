"""Exploration: how much of the headline edge is estimation noise?

The plan asks every Sharpe ratio to carry a bootstrap confidence interval, and
the comparison so far reports only a per-seed spread. Two sources of noise are
separated here for the frozen benchmark cell (Heston, 20 bp, default budget):

- *training-seed* noise, by bootstrapping the eight per-seed edges, which is
  what an independent re-run of the whole pipeline would move; and
- *path* noise, by paired-resampling the pooled validation paths and recomputing
  the Sharpe difference, which is the error of the estimate conditional on the
  trained nets.

A path-count sweep checks that the reported edge is stable rather than an
artefact of the 40,000-path evaluation. The holdout pool is never touched.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "exploration"
N_BOOT = 4000
RNG = np.random.default_rng(7)


def bootstrap_over_seeds(edges, n_boot=N_BOOT) -> dict:
    edges = np.asarray(edges, dtype=float)
    draws = RNG.choice(edges, size=(n_boot, edges.size), replace=True).mean(axis=1)
    return {
        "mean": float(edges.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def bootstrap_paths(deep, classical, n_boot=N_BOOT) -> dict:
    """Paired path bootstrap of the Sharpe difference.

    Resampling the same path indices for both strategies preserves the
    path-level pairing, which is what makes the interval the sampling error of
    the comparison rather than the sum of two independent errors.
    """
    n = deep.shape[0]
    base = metrics.sharpe(deep) - metrics.sharpe(classical)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        out[b] = metrics.sharpe(deep[idx]) - metrics.sharpe(classical[idx])
    return {
        "point": float(base),
        "ci_low": float(np.quantile(out, 0.025)),
        "ci_high": float(np.quantile(out, 0.975)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = B.BenchConfig()
    classical = B.ClassicalStrategy(base, B.load_classical_best())
    hyper = Hyper()  # the frozen default: 200 epochs, hidden 128, depth 3
    nets, _ = B.train_ensemble(base, hyper, device="cuda")
    learned = B.NetworkStrategy(base, nets, device="cuda")

    edges, deep_all, class_all = [], [], []
    for seed in base.val_seeds:
        paths = base.simulate(base.model, seed, base.n_val_paths)
        pnl_c = classical.pnl(paths)
        pnl_d = learned.pnl(paths)
        edges.append(metrics.sharpe(pnl_d) - metrics.sharpe(pnl_c))
        deep_all.append(pnl_d)
        class_all.append(pnl_c)

    seed_ci = bootstrap_over_seeds(edges)
    path_ci = bootstrap_paths(np.concatenate(deep_all), np.concatenate(class_all))

    print("training-seed bootstrap (n=8 edges):")
    print(f"  mean={seed_ci['mean']:+.4f}  95% CI [{seed_ci['ci_low']:+.4f}, {seed_ci['ci_high']:+.4f}]")
    print("paired path bootstrap (320k paths):")
    print(f"  mean={path_ci['point']:+.4f}  95% CI [{path_ci['ci_low']:+.4f}, {path_ci['ci_high']:+.4f}]")

    # Path-count sensitivity: the same nets, more evaluation paths. If the edge
    # drifts with the count the original 40k estimate was under-powered.
    sensitivity = []
    for n_paths in (5_000, 20_000, 40_000, 80_000):
        vals = []
        for seed in base.val_seeds:
            paths = base.simulate(base.model, seed, n_paths)
            vals.append(
                metrics.sharpe(learned.pnl(paths)) - metrics.sharpe(classical.pnl(paths))
            )
        sensitivity.append({"n_paths": n_paths, "edge_mean": float(np.mean(vals)),
                            "edge_std": float(np.std(vals, ddof=1))})
        print(f"  n_paths={n_paths:>6}: edge={sensitivity[-1]['edge_mean']:+.4f} "
              f"+/- {sensitivity[-1]['edge_std']:.4f}")

    (OUT / "robustness.json").write_text(
        json.dumps(
            {"seed_bootstrap": seed_ci, "path_bootstrap": path_ci,
             "path_sensitivity": sensitivity},
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {OUT.relative_to(ROOT)}/robustness.json")


if __name__ == "__main__":
    main()
