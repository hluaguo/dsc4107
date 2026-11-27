"""Final evaluation on the locked holdout paths.

This script is run once, after the optimisation loop has stopped. The holdout
seeds never appear in the loop, so the number printed here is the only
validation-free estimate of the edge and the one that belongs in the report. A
holdout edge well below the validation edge means the loop was fitting its own
noise, not the problem.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import metrics  # noqa: E402

RESULTS = ROOT / "results"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="holdout")
    args = parser.parse_args()

    cfg = B.BenchConfig()
    best = B.load_classical_best()
    classical = B.ClassicalStrategy(cfg, best)
    nets, _ = B.train_ensemble(cfg, device=args.device)
    learned = B.NetworkStrategy(cfg, nets, device=args.device)

    rows = []
    edges = []
    for seed in cfg.holdout_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        pnl_c = classical.pnl(paths)
        pnl_d = learned.pnl(paths)
        sc = metrics.summarize(pnl_c)
        sd = metrics.summarize(pnl_d)
        edges.append(sd["sharpe"] - sc["sharpe"])
        rows.append(
            {
                "seed": int(seed),
                "classical_sharpe": sc["sharpe"],
                "deep_sharpe": sd["sharpe"],
                "edge_sharpe": sd["sharpe"] - sc["sharpe"],
                "classical_cvar95": sc["cvar95"],
                "deep_cvar95": sd["cvar95"],
                "paired_tstat": metrics.paired_tstat(pnl_d, pnl_c),
            }
        )

    edges = np.asarray(edges)
    summary = {
        "edge_mean": float(edges.mean()),
        "edge_std": float(edges.std(ddof=1)),
        "edge_min": float(edges.min()),
        "win_rate": float((edges > 0).mean()),
        "n_seeds": int(edges.size),
    }
    for key, value in summary.items():
        print(f"METRIC holdout_{key}={value:.6f}")

    RESULTS.mkdir(exist_ok=True)
    frame = pl.DataFrame(rows)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frame.write_parquet(RESULTS / f"holdout_{stamp}_{args.tag}.parquet")
    with (RESULTS / "holdout_history.jsonl").open("a") as handle:
        handle.write(json.dumps({**summary, "timestamp": stamp}, default=str) + "\n")


if __name__ == "__main__":
    main()
