"""Choose the classical baseline once, then freeze it.

The baseline is tuned on its own seed pool, disjoint from the validation and
holdout pools. Re-tuning it inside the optimisation loop would let the bar move
under the comparison, so the chosen configuration is written out and treated as
fixed from that point on. The remaining details are reported so the frozen
choice can be audited rather than trusted.
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import hedge, metrics  # noqa: E402


def candidate_configs():
    configs = [
        {"strategy": "time", "frequency": f, "delta_band": None, "price_band": None}
        for f in B.FREQUENCIES
    ]
    configs += [
        {"strategy": "band", "frequency": 1, "delta_band": d, "price_band": None}
        for d in B.DELTA_BANDS
    ]
    configs += [
        {"strategy": "band", "frequency": 1, "delta_band": None, "price_band": p}
        for p in B.PRICE_BANDS
    ]
    configs += [
        {"strategy": "band", "frequency": 1, "delta_band": d, "price_band": p}
        for d in B.DELTA_BANDS
        for p in B.PRICE_BANDS
    ]
    return configs


def mean_sharpe(cfg: B.BenchConfig, candidate: dict, seeds) -> float:
    sharpes = []
    for seed in seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        fn = B.make_classical_position_fn(cfg, candidate)
        result = hedge.rollout(
            paths, fn, cfg.K, cfg.kind, cfg.premium, cfg.cost, cfg.dt
        )
        sharpes.append(metrics.sharpe(result.pnl))
    return float(np.mean(sharpes))


def main() -> None:
    cfg = B.BenchConfig()
    scored = [(mean_sharpe(cfg, c, cfg.tune_seeds), c) for c in candidate_configs()]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    print("top candidates on the tuning pool (higher Sharpe is better):")
    for sharpe, candidate in scored[:8]:
        print(f"  {sharpe: .4f}  {json.dumps(candidate)}")

    best = scored[0][1]
    B.CLASSICAL_BEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    B.CLASSICAL_BEST_PATH.write_text(json.dumps(best, indent=2) + "\n")
    print(f"\nfrozen baseline -> {B.CLASSICAL_BEST_PATH}")


if __name__ == "__main__":
    main()
