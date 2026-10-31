"""Reproduce the Heston experiment of Buehler et al. (2019), plus its check.

Two things are asked of this script:

1. With no transaction costs, a correctly implemented hedger should learn a
   policy close to the Black-Scholes delta. If it does not, nothing else it
   reports is trustworthy, so this is the first check and it is reported first.
2. With proportional costs, the learned hedger should beat the textbook delta
   hedge on risk-adjusted P&L in the Heston market the paper uses.

The comparator here is the *plain* Black-Scholes delta at every step, which is
what the paper benchmarks against. The frozen band baseline that ``bench``
carries is a harder, cost-aware comparator and is reported separately so the
two comparisons are not confused.
"""

from __future__ import annotations

import argparse
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
import polars as pl  # noqa: E402
import torch  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import bs, deep, hedge, metrics  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "deep_hedging"


def bs_delta_positions(cfg, paths) -> np.ndarray:
    """Black-Scholes delta at every decision point, the frictionless target."""
    n_steps = paths.shape[1] - 1
    out = np.empty((paths.shape[0], n_steps))
    for j in range(n_steps):
        tau = max(cfg.T - j * cfg.dt, 0.0)
        out[:, j] = bs.bs_delta(paths[:, j], cfg.K, tau, cfg.r, cfg.sigma_impl, cfg.kind)
    return out


def resolve_device(requested: str) -> str:
    """Use CUDA when asked for ``auto`` and it is present.

    The rollout is a stack of tiny per-step matmuls, which a GPU still handles
    far faster than the CPU; keeping the CPU fallback means the script is not
    tied to an accelerator being present.
    """
    if requested != "auto":
        return requested
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def frictionless_check(cfg, hyper, n_paths, seed, device="cpu"):
    """Train without cost and compare the learned policy with BS delta.

    Correlation is reported alongside the RMSE because a policy can sit on a
    different *scale* of delta (for example half the position) yet still be
    monotonically right; the correlation separates a scaling error from a
    policy that learned the wrong shape.
    """
    paths = cfg.simulate(cfg.model, seed, n_paths)
    tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
    nets, _ = B.train_ensemble(cfg, hyper, device=device)
    net = nets[0]
    learned = deep.hedge_positions(tensor, net, cfg.K, cfg.dt, cfg.T)
    target = bs_delta_positions(cfg, paths)
    x = learned.reshape(-1)
    y = target.reshape(-1)
    rmse = float(np.sqrt(np.mean((x - y) ** 2)))
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.polyfit(y, x, 1)[0])
    return {
        "rmse": rmse,
        "correlation": corr,
        "slope": slope,
        "mean_learned": float(x.mean()),
        "mean_target": float(y.mean()),
    }


def evaluate(cfg, hyper, device="cpu"):
    """Deep versus plain BS delta versus the frozen band, on val seeds."""
    nets, _ = B.train_ensemble(cfg, hyper, device=device)
    learned = B.NetworkStrategy(cfg, nets, device=device)
    delta = B.ClassicalStrategy(
        cfg, {"strategy": "time", "frequency": 1, "delta_band": None, "price_band": None}
    )
    band = B.ClassicalStrategy(cfg, B.load_classical_best())
    rows = []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        pnl_d = learned.pnl(paths)
        pnl_delta = delta.pnl(paths)
        pnl_band = band.pnl(paths)
        ones = np.zeros_like(pnl_d)
        rows.append(
            {
                "seed": int(seed),
                "deep_sharpe": metrics.sharpe(pnl_d),
                "bs_delta_sharpe": metrics.sharpe(pnl_delta),
                "band_sharpe": metrics.sharpe(pnl_band),
                "edge_vs_delta": metrics.sharpe(pnl_d) - metrics.sharpe(pnl_delta),
                "edge_vs_band": metrics.sharpe(pnl_d) - metrics.sharpe(pnl_band),
                "deep_std": float(pnl_d.std()),
                "delta_std": float(pnl_delta.std()),
                "paired_tstat": metrics.paired_tstat(pnl_d, pnl_delta),
                "placeholder": float(ones.sum()),
            }
        )
    return rows


def summarize(rows, keys):
    out = {}
    for key in keys:
        values = np.array([r[key] for r in rows])
        out[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "min": float(values.min()),
        }
    return out


def figure_sharpe_vs_cost(results: list[dict]) -> pathlib.Path:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    costs = [r["cost"] * 1e4 for r in results]
    for key, label in (
        ("deep_sharpe", "learned"),
        ("bs_delta_sharpe", "BS delta (every step)"),
        ("band_sharpe", "frozen band baseline"),
    ):
        ax.plot(costs, [r["summary"][key]["mean"] for r in results], marker="o", label=label)
    ax.set_xlabel("proportional transaction cost (bp)")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Heston experiment: learned hedger against classical hedges")
    ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "sharpe_vs_cost.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--costs", nargs="*", type=float, default=[0.0, 0.002, 0.005, 0.01])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.65)
    parser.add_argument("--frictionless-paths", type=int, default=20_000)
    parser.add_argument("--frictionless-seed", type=int, default=2001)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for training ('auto', 'cpu', 'cuda')",
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    print(f"device: {args.device}")
    if args.quick:
        args.costs = [0.0]
        args.epochs = 10
        args.hidden = 32
        args.frictionless_paths = 2000

    OUT.mkdir(parents=True, exist_ok=True)
    base = B.BenchConfig()

    # Frictionless check first: it is the paper's own correctness test and the
    # only case where we know the right answer in advance.
    frictionless_cfg = replace(base, cost=0.0, n_val_paths=args.frictionless_paths)
    check_hyper = Hyper(
        epochs=args.epochs,
        hidden=args.hidden,
        depth=args.depth,
        alpha=args.alpha,
        torch_seeds=(0,),
    )
    check = frictionless_check(
        frictionless_cfg,
        check_hyper,
        args.frictionless_paths,
        args.frictionless_seed,
        device=args.device,
    )
    print("frictionless delta check (cost = 0):")
    print(
        f"  correlation={check['correlation']:.4f}  slope={check['slope']:.3f}  "
        f"rmse={check['rmse']:.4f}  mean learned={check['mean_learned']:.3f} "
        f"mean delta={check['mean_target']:.3f}"
    )

    results = []
    for cost in args.costs:
        cfg = replace(base, cost=float(cost))
        hyper = Hyper(
            epochs=args.epochs,
            hidden=args.hidden,
            depth=args.depth,
            alpha=args.alpha,
            torch_seeds=(0, 1, 2),
        )
        rows = evaluate(cfg, hyper, device=args.device)
        summary = summarize(
            rows, ["deep_sharpe", "bs_delta_sharpe", "band_sharpe", "edge_vs_delta", "edge_vs_band"]
        )
        results.append({"cost": float(cost), "summary": summary, "rows": rows})
        print(
            f"cost={cost:<6} deep={summary['deep_sharpe']['mean']:.3f} "
            f"delta={summary['bs_delta_sharpe']['mean']:.3f} "
            f"band={summary['band_sharpe']['mean']:.3f}  "
            f"edge_vs_delta={summary['edge_vs_delta']['mean']:+.4f}"
        )

    (OUT / "results.json").write_text(
        json.dumps(
            {"frictionless": check, "costs": results, "device": args.device},
            indent=2,
        )
        + "\n"
    )
    pl.DataFrame(results[0]["rows"]).write_csv(OUT / "val_records_cost0.csv")
    path = figure_sharpe_vs_cost(results)
    print(f"\nwrote {path.relative_to(ROOT)} and {OUT.relative_to(ROOT)}/results.json")


if __name__ == "__main__":
    main()
