"""One benchmark iteration, reported as machine-readable metrics.

Every number is computed on the validation seed pool. The holdout pool is
deliberately absent, because reading it inside the loop would turn the final
test into just another training signal. Per-seed rows are written out so the
spread behind the headline edge stays inspectable after a context reset.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime, timezone

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import polars as pl  # noqa: E402

from hedging import bench as B  # noqa: E402

PRIMARY = "edge_mean"
RESULTS = ROOT / "results"


def emit(name: str, value) -> None:
    if isinstance(value, float):
        print(f"METRIC {name}={value:.6f}")
    else:
        print(f"METRIC {name}={value}")


def record(summary: dict, elapsed: float, tag: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    frame = pl.DataFrame(summary["val_records"])
    frame.write_parquet(RESULTS / f"val_{stamp}_{tag}.parquet")

    flat = {
        key: value
        for key, value in summary.items()
        if key not in ("val_records", "train_histories")
    }
    flat["wall_time_s"] = elapsed
    flat["timestamp"] = stamp
    flat["tag"] = tag
    with (RESULTS / "history.jsonl").open("a") as handle:
        handle.write(json.dumps(flat, default=str) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="run")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    started = time.time()
    summary = B.run_benchmark(device=args.device, quick=args.quick)
    elapsed = time.time() - started

    emit("edge_mean", summary["edge_mean"])
    emit("edge_std", summary["edge_std"])
    emit("edge_min", summary["edge_min"])
    emit("edge_median", summary["edge_median"])
    emit("edge_tstat", summary["edge_tstat"])
    emit("win_rate", summary["win_rate"])
    emit("cvar_edge_mean", summary["cvar_edge_mean"])
    emit("val_deep_sharpe_mean", summary["val_deep_sharpe_mean"])
    emit("val_classical_sharpe_mean", summary["val_classical_sharpe_mean"])
    emit("train_edge", summary["train_edge"])
    emit("generation_gap", summary["generation_gap"])
    for model, value in summary["robust_edges"].items():
        emit(f"robust_edge_{model}", value)
    emit("wall_time_s", elapsed)

    print("# per-seed edge (deep minus classical)")
    for row in summary["val_records"]:
        print(
            f"#   seed {row['seed']}: edge={row['edge_sharpe']:+.4f} "
            f"deep={row['deep_sharpe']:.4f} classical={row['classical_sharpe']:.4f} "
            f"t={row['paired_tstat']:.2f}"
        )

    if not args.no_record:
        record(summary, elapsed, args.tag)


if __name__ == "__main__":
    main()
