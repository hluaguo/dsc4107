"""The original question: does the learned hedger beat the tuned classical one?

Sepp (2013) optimises a hand-built family; Buehler et al. (2019) search a much
larger family with a network. Neither runs the head-to-head comparison across
market models and cost levels, which is what this script does.

Design decisions that matter for the honesty of the result:

- The classical baseline is re-tuned for every ``(model, cost)`` cell on its own
  ``tune_seeds`` pool, exactly as ``scripts/tune_classical.py`` freezes it for
  the main benchmark. Comparing a tuned network against a stale classical
  setting would manufacture a win.
- The Optuna search optimises on ``SEARCH_SEEDS``, a pool disjoint from both the
  training seeds and the benchmark's validation seeds. Selecting the network
  configuration on the same seeds that produce the headline edge is the obvious
  way to overfit, so a fourth pool is carved out for the search.
- The reported number is still the mean per-seed edge on the benchmark's
  ``val_seeds``, averaged over ``Hyper.torch_seeds``. The search pool only
  decides *which* configuration is reported.

The run is deliberately small: this is a course-scale testbed, not a paper's
compute. ``--quick`` shrinks paths and epochs further for a smoke test.
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
import optuna  # noqa: E402
import polars as pl  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import deep, hedge, metrics, market  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "comparison"

MODELS = (market.GBM, market.JUMP, market.HESTON, market.SVJ)
DEFAULT_COSTS = (0.0, 0.002, 0.005, 0.01, 0.02)

# A fourth seed pool, disjoint from train/val/tune/holdout. Its only job is to
# rank Optuna trials, so it never contributes to a reported edge.
SEARCH_SEEDS = (4001, 4002, 4003)

SEARCH_SPACE = {
    "hidden": (32, 256),
    "depth": (2, 4),
    "lr": (1e-4, 3e-3),
    "alpha": (0.2, 0.9),
}


def candidate_classical_configs():
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


def classical_sharpe(cfg, candidate, seeds) -> float:
    sharpe = 0.0
    for seed in seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        fn = B.make_classical_position_fn(cfg, candidate)
        result = hedge.rollout(
            paths, fn, cfg.K, cfg.kind, cfg.premium, cfg.cost, cfg.dt
        )
        sharpe += metrics.sharpe(result.pnl)
    return sharpe / len(seeds)


def tune_classical(cfg) -> dict:
    scored = [(classical_sharpe(cfg, c, cfg.tune_seeds), c) for c in candidate_classical_configs()]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def resolve_device(requested: str) -> str:
    """Pick the accelerator when asked for ``auto``.

    Training here is a rollout of many small steps, so the win is the throughput
    of the many tiny matmuls rather than raw FLOPs; an RTX 4060 still finishes a
    cell in a fraction of the CPU time. Falling back silently keeps the harness
    runnable on a machine without CUDA.
    """
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def deep_edge(cfg, classical: B.ClassicalStrategy, nets, seeds, device: str) -> float:
    """Mean per-seed Sharpe edge of the network over the classical strategy."""
    learned = B.NetworkStrategy(cfg, nets, device=device)
    edges = []
    for seed in seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        pnl_c = classical.pnl(paths)
        pnl_d = learned.pnl(paths)
        edges.append(metrics.sharpe(pnl_d) - metrics.sharpe(pnl_c))
    return sum(edges) / len(edges)


def report(cfg, classical: B.ClassicalStrategy, nets, seeds, device: str) -> dict:
    """The numbers that go in the table, plus the guard rails."""
    learned = B.NetworkStrategy(cfg, nets, device=device)
    edges, deep, clas, cvar_edges = [], [], [], []
    deep_std, class_std, deep_mean, class_mean = [], [], [], []
    for seed in seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        pnl_c = classical.pnl(paths)
        pnl_d = learned.pnl(paths)
        sc, sd = metrics.summarize(pnl_c), metrics.summarize(pnl_d)
        edges.append(sd["sharpe"] - sc["sharpe"])
        cvar_edges.append(sc["cvar95"] - sd["cvar95"])
        deep.append(sd["sharpe"])
        clas.append(sc["sharpe"])
        deep_std.append(sd["std"])
        class_std.append(sc["std"])
        deep_mean.append(sd["mean"])
        class_mean.append(sc["mean"])
    n = len(edges)
    mean = sum(edges) / n
    sd_edge = (sum((e - mean) ** 2 for e in edges) / (n - 1)) ** 0.5
    # The spread columns are the premium-harvesting guard: when the deep P&L
    # standard deviation diverges from the classical one while its Sharpe rises,
    # the network has changed its risk profile (typically by under-hedging and
    # collecting the volatility risk premium) rather than hedged better. Without
    # these a high-cost "win" would be uninterpretable.
    return {
        "edge_mean": mean,
        "edge_std": sd_edge,
        "edge_min": min(edges),
        "win_rate": sum(e > 0 for e in edges) / n,
        "cvar_edge_mean": sum(cvar_edges) / n,
        "deep_sharpe": sum(deep) / n,
        "classical_sharpe": sum(clas) / n,
        "deep_std": sum(deep_std) / n,
        "classical_std": sum(class_std) / n,
        "deep_mean": sum(deep_mean) / n,
        "classical_mean": sum(class_mean) / n,
    }


def run_cell(model: str, cost: float, args) -> dict:
    cfg = replace(
        B.BenchConfig(),
        model=model,
        cost=float(cost),
        n_train_paths=args.train_paths,
        n_val_paths=args.val_paths,
    )
    classical_best = tune_classical(cfg)
    classical = B.ClassicalStrategy(cfg, classical_best)

    if args.fixed_hyper:
        # Skipping the search is the honest way to produce the edge map when the
        # compute budget only allows one configuration per cell: the same explicit
        # hyperparameters are used everywhere, so a cell cannot win because it
        # got luckier in the search than its neighbours.
        final_hyper = Hyper(
            epochs=args.epochs,
            lr=args.fixed_lr,
            hidden=args.fixed_hidden,
            depth=args.fixed_depth,
            alpha=args.fixed_alpha,
            torch_seeds=(0, 1, 2),
        )
        nets, _ = B.train_ensemble(cfg, final_hyper, device=args.device)
        result = report(cfg, classical, nets, cfg.val_seeds, args.device)
        result.update(
            {
                "model": model,
                "cost": float(cost),
                "classical": classical_best,
                "hyper": {
                    "lr": args.fixed_lr,
                    "hidden": args.fixed_hidden,
                    "depth": args.fixed_depth,
                    "alpha": args.fixed_alpha,
                    "epochs": args.epochs,
                    "searched": False,
                },
                "search_edge": None,
                "n_trials": 0,
                "device": args.device,
            }
        )
        print(
            f"{model:>6} cost={cost:<6} edge={result['edge_mean']:+.4f}  "
            f"win={result['win_rate']:.2f} deep={result['deep_sharpe']:.3f} "
            f"class={result['classical_sharpe']:.3f} "
            f"(std {result['deep_std']:.3f}/{result['classical_std']:.3f})"
        )
        return result

    def objective(trial: optuna.Trial) -> float:
        lo, hi = SEARCH_SPACE["hidden"]
        lo_d, hi_d = SEARCH_SPACE["depth"]
        lo_lr, hi_lr = SEARCH_SPACE["lr"]
        lo_a, hi_a = SEARCH_SPACE["alpha"]
        hyper = Hyper(
            epochs=args.epochs,
            batch_size=1024,
            lr=float(trial.suggest_float("lr", lo_lr, hi_lr, log=True)),
            hidden=int(trial.suggest_int("hidden", lo, hi, step=32)),
            depth=int(trial.suggest_int("depth", lo_d, hi_d)),
            alpha=float(trial.suggest_float("alpha", lo_a, hi_a)),
            torch_seeds=(0,),
        )
        nets, _ = B.train_ensemble(cfg, hyper, device=args.device)
        return deep_edge(cfg, classical, nets, SEARCH_SEEDS, args.device)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
    best = study.best_params

    final_hyper = Hyper(
        epochs=args.epochs,
        lr=float(best["lr"]),
        hidden=int(best["hidden"]),
        depth=int(best["depth"]),
        alpha=float(best["alpha"]),
        torch_seeds=(0, 1, 2),
    )
    nets, _ = B.train_ensemble(cfg, final_hyper, device=args.device)
    result = report(cfg, classical, nets, cfg.val_seeds, args.device)
    result.update(
        {
            "model": model,
            "cost": float(cost),
            "classical": classical_best,
            "hyper": {
                "lr": best["lr"],
                "hidden": best["hidden"],
                "depth": best["depth"],
                "alpha": best["alpha"],
                "epochs": args.epochs,
            },
            "search_seeds": list(SEARCH_SEEDS),
            "search_edge": study.best_value,
            "n_trials": args.trials,
            "device": args.device,
        }
    )
    print(
        f"{model:>6} cost={cost:<6} edge={result['edge_mean']:+.4f} "
        f"(search {study.best_value:+.4f})  win={result['win_rate']:.2f} "
        f"deep={result['deep_sharpe']:.3f} class={result['classical_sharpe']:.3f}  "
        f"hyper={result['hyper']}"
    )
    return result


def figure_edge_map(rows: list[dict]) -> pathlib.Path:
    """Edge against cost, one line per model, with the zero line marked.

    A single picture of where the learned hedger stops paying for itself: the
    crossing point is the object the comparison is trying to locate, and the
    zero line makes a negative edge unmistakable.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for model in MODELS:
        subset = sorted(
            [r for r in rows if r["model"] == model], key=lambda r: r["cost"]
        )
        if not subset:
            continue
        ax.plot(
            [r["cost"] * 1e4 for r in subset],
            [r["edge_mean"] for r in subset],
            marker="o",
            label=model,
        )
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("proportional transaction cost (bp)")
    ax.set_ylabel("Sharpe edge: learned - classical")
    ax.set_title("Where the learned hedger wins")
    ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "edge_map.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--costs", nargs="*", type=float, default=list(DEFAULT_COSTS))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-paths", type=int, default=2048)
    parser.add_argument("--val-paths", type=int, default=40_000)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--fixed-hyper",
        action="store_true",
        help="skip Optuna and use one explicit configuration in every cell",
    )
    parser.add_argument("--fixed-hidden", type=int, default=64)
    parser.add_argument("--fixed-depth", type=int, default=3)
    parser.add_argument("--fixed-lr", type=float, default=1e-3)
    parser.add_argument("--fixed-alpha", type=float, default=0.65)
    parser.add_argument(
        "--device",
        default="auto",
        help="torch device for training ('auto', 'cpu', 'cuda')",
    )
    args = parser.parse_args()
    args.device = resolve_device(args.device)
    print(f"device: {args.device}")
    if args.quick:
        args.trials = min(args.trials, 2)
        args.epochs = min(args.epochs, 10)
        args.train_paths = min(args.train_paths, 256)
        args.val_paths = min(args.val_paths, 2000)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        for cost in args.costs:
            rows.append(run_cell(model, cost, args))

    frame = pl.DataFrame(
        [
            {
                k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                for k, v in r.items()
            }
            for r in rows
        ]
    )
    frame.write_csv(OUT / "edge_map.csv")
    (OUT / "edge_map.json").write_text(json.dumps(rows, indent=2) + "\n")
    path = figure_edge_map(rows)
    print(f"\nwrote {path.relative_to(ROOT)} and {OUT.relative_to(ROOT)}/edge_map.csv")


if __name__ == "__main__":
    main()
