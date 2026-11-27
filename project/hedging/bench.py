"""The frozen benchmark used by every comparison in this project.

Nothing here is re-tuned per experiment. The market, the seeds, the cost level,
the contract and the classical baseline are fixed, so a change in the reported
edge reflects a better hedge rather than an easier test. Three disjoint seed
pools carry the honesty:

- ``train`` seeds generate the paths the network learns on;
- ``val`` seeds score the running experiments;
- ``holdout`` seeds are never touched by the loop and are read once at the end.

Keeping ``val`` separate from ``holdout`` is what stops the optimisation loop
from quietly becoming a search over the test set. The generator gap, reported
next to the edge, is the second guard: it exposes a change that helps only on
the paths the network has already seen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import bs as _bs
from . import deep, hedge, market, metrics
from .hyper import Hyper

CLASSICAL_BEST_PATH = Path("bench/classical_best.json")

# Frequencies and bands the classical tune searches over. Kept here rather than
# in the loop so the baseline cannot drift while the network is being changed.
FREQUENCIES = (1, 2, 3, 5, 7, 10, 21)
DELTA_BANDS = (0.05, 0.10, 0.20)
PRICE_BANDS = (0.01, 0.02, 0.05)

DEFAULT_CLASSICAL = {
    "strategy": "time",
    "frequency": 2,
    "delta_band": None,
    "price_band": None,
}


@dataclass(frozen=True)
class BenchConfig:
    """The fixed description of the experiment.

    Pool sizes are large enough to give a stable Sharpe estimate and small
    enough that a full iteration finishes in minutes, because a benchmark nobody
    runs is worse than a slightly noisy one.
    """

    name: str = "heston_atm_1m"
    model: str = market.HESTON
    S0: float = 100.0
    K: float = 100.0
    T: float = 1.0 / 12.0
    r: float = 0.0
    # Implied volatility sits above the Heston long-run level of 0.20 on purpose.
    # Without a volatility risk premium, selling the option and doing nothing
    # dominates any hedge, so a Sharpe-based comparison collapses into "trade as
    # little as possible" and the network learns to skip hedging altogether. The
    # premium makes variance reduction worth paying for, which is the setting the
    # two papers actually study.
    sigma_impl: float = 0.25
    kind: str = "call"
    cost: float = 0.002
    n_steps: int = 21

    n_train_paths: int = 2048
    n_val_paths: int = 40_000

    train_seeds: tuple[int, ...] = (1001, 1002, 1003)
    val_seeds: tuple[int, ...] = (2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008)
    tune_seeds: tuple[int, ...] = (3001, 3002, 3003, 3004)
    holdout_seeds: tuple[int, ...] = (9001, 9002, 9003, 9004, 9005)

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def premium(self) -> float:
        return float(
            _bs.bs_price(self.S0, self.K, self.T, self.r, self.sigma_impl, self.kind)
        )

    def market_params(self, model: str | None = None) -> market.MarketParams:
        return market.MarketParams(model=model or self.model)

    def simulate(self, model: str, seed: int, n_paths: int) -> np.ndarray:
        return market.simulate_paths(
            self.market_params(model), self.S0, self.T, self.n_steps, n_paths, seed
        )


def load_classical_best(path: Path = CLASSICAL_BEST_PATH) -> dict:
    if not path.exists():
        return dict(DEFAULT_CLASSICAL)
    return json.loads(path.read_text())


def make_classical_position_fn(cfg: BenchConfig, best: dict):
    if best["strategy"] == "time":
        return hedge.TimeBasedDelta(
            cfg.K, cfg.T, cfg.r, cfg.sigma_impl, cfg.kind, cfg.dt, best["frequency"]
        )
    if best["strategy"] == "band":
        return hedge.BandDelta(
            cfg.K,
            cfg.T,
            cfg.r,
            cfg.sigma_impl,
            cfg.kind,
            cfg.dt,
            delta_band=best.get("delta_band"),
            price_band=best.get("price_band"),
        )
    raise ValueError(f"unknown classical strategy {best['strategy']!r}")


class ClassicalStrategy:
    """The frozen baseline: textbook delta on a fixed schedule or band."""

    def __init__(self, cfg: BenchConfig, best: dict):
        self.cfg = cfg
        self.best = best

    def pnl(self, paths: np.ndarray) -> np.ndarray:
        fn = make_classical_position_fn(self.cfg, self.best)
        return hedge.rollout(
            paths,
            fn,
            self.cfg.K,
            self.cfg.kind,
            self.cfg.premium,
            self.cfg.cost,
            self.cfg.dt,
        ).pnl


class NetworkStrategy:
    """An ensemble of trained hedgers, combined by averaging their cash flows.

    Averaging several training runs rather than reporting the best one keeps
    seed luck out of the headline number, and averaging the P&L itself means the
    reported strategy is one a desk could actually deploy.
    """

    def __init__(self, cfg: BenchConfig, nets, device: str = "cpu"):
        self.cfg = cfg
        self.nets = list(nets)
        self.device = device

    def pnl(self, paths: np.ndarray) -> np.ndarray:
        tensor = torch.as_tensor(paths, dtype=torch.float32, device=self.device)
        members = [
            deep.evaluate_network(
                tensor,
                net,
                self.cfg.K,
                self.cfg.kind,
                self.cfg.premium,
                self.cfg.cost,
                self.cfg.dt,
                self.cfg.T,
            )
            for net in self.nets
        ]
        return np.mean(members, axis=0)


def training_paths(cfg: BenchConfig) -> np.ndarray:
    return np.concatenate(
        [cfg.simulate(cfg.model, s, cfg.n_train_paths) for s in cfg.train_seeds],
        axis=0,
    )


def train_ensemble(cfg: BenchConfig, hyper: Hyper | None = None, device: str = "cpu"):
    hyper = hyper or Hyper()
    paths = training_paths(cfg)
    tensor = torch.as_tensor(paths, dtype=torch.float32, device=device)
    nets = []
    histories = []
    for seed in hyper.torch_seeds:
        net, history = deep.train_hedger(
            tensor,
            cfg.K,
            cfg.kind,
            cfg.premium,
            cfg.cost,
            cfg.dt,
            cfg.T,
            epochs=hyper.epochs,
            batch_size=hyper.batch_size,
            lr=hyper.lr,
            hidden=hyper.hidden,
            depth=hyper.depth,
            alpha=hyper.alpha,
            seed=seed,
            device=device,
        )
        nets.append(net)
        histories.append(history)
    return nets, histories


def evaluate_on_seeds(cfg: BenchConfig, strategy, model: str, seeds, n_paths: int):
    records, pnls = [], []
    for seed in seeds:
        paths = cfg.simulate(model, seed, n_paths)
        pnl = strategy.pnl(paths)
        stats = metrics.summarize(pnl)
        stats.update({"seed": int(seed), "model": model})
        records.append(stats)
        pnls.append(pnl)
    return records, pnls


def _aggregate(edges, cvar_edges, deep_sharpes, classical_sharpes) -> dict:
    edges = np.asarray(edges, dtype=float)
    cvar_edges = np.asarray(cvar_edges, dtype=float)
    n = edges.size
    sd = edges.std(ddof=1) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else 0.0
    return {
        "edge_mean": float(edges.mean()),
        "edge_std": float(sd),
        "edge_min": float(edges.min()),
        "edge_median": float(np.median(edges)),
        "edge_se": float(se),
        "edge_tstat": float(edges.mean() / se) if se > 0 else 0.0,
        "win_rate": float((edges > 0).mean()),
        "cvar_edge_mean": float(cvar_edges.mean()),
        "val_deep_sharpe_mean": float(np.mean(deep_sharpes)),
        "val_classical_sharpe_mean": float(np.mean(classical_sharpes)),
    }


def run_benchmark(
    cfg: BenchConfig | None = None,
    hyper: Hyper | None = None,
    device: str = "cpu",
    robust_models: tuple[str, ...] = (market.GBM, market.JUMP),
    quick: bool = False,
    nets=None,
) -> dict:
    """Train the learned hedger and score it against the frozen baseline.

    Both strategies see the same paths and the same premium, so the difference
    between them is the difference between their hedges. The edge is averaged
    over independent validation seeds and the per-seed spread is reported with
    it, so a mean that hides disagreement between seeds cannot pass as a result.
    """
    cfg = cfg or BenchConfig()
    if quick:
        cfg, hyper = _quick_setup(cfg, hyper)
    hyper = hyper or Hyper()

    best = load_classical_best()
    classical = ClassicalStrategy(cfg, best)
    if nets is None:
        nets, histories = train_ensemble(cfg, hyper, device=device)
    else:
        histories = []
    learned = NetworkStrategy(cfg, nets, device=device)

    val_records = []
    edges, cvar_edges, deep_sharpes, classical_sharpes = [], [], [], []
    for seed in cfg.val_seeds:
        paths = cfg.simulate(cfg.model, seed, cfg.n_val_paths)
        pnl_c = classical.pnl(paths)
        pnl_d = learned.pnl(paths)
        sc = metrics.summarize(pnl_c)
        sd = metrics.summarize(pnl_d)
        val_records.append(
            {
                "seed": int(seed),
                "model": cfg.model,
                "classical_sharpe": sc["sharpe"],
                "deep_sharpe": sd["sharpe"],
                "classical_cvar95": sc["cvar95"],
                "deep_cvar95": sd["cvar95"],
                "classical_mean": sc["mean"],
                "deep_mean": sd["mean"],
                "classical_std": sc["std"],
                "deep_std": sd["std"],
                "edge_sharpe": sd["sharpe"] - sc["sharpe"],
                "edge_cvar": sc["cvar95"] - sd["cvar95"],
                "paired_tstat": metrics.paired_tstat(pnl_d, pnl_c),
            }
        )
        edges.append(sd["sharpe"] - sc["sharpe"])
        cvar_edges.append(sc["cvar95"] - sd["cvar95"])
        deep_sharpes.append(sd["sharpe"])
        classical_sharpes.append(sc["sharpe"])

    summary = _aggregate(edges, cvar_edges, deep_sharpes, classical_sharpes)
    summary.update(
        {
            "config": cfg.name,
            "model": cfg.model,
            "cost": cfg.cost,
            "n_val_seeds": len(cfg.val_seeds),
            "n_ensemble": len(nets),
            "classical_best": best,
            "premium": cfg.premium,
            "hyper": {
                "epochs": hyper.epochs,
                "batch_size": hyper.batch_size,
                "lr": hyper.lr,
                "hidden": hyper.hidden,
                "depth": hyper.depth,
                "alpha": hyper.alpha,
                "torch_seeds": list(hyper.torch_seeds),
            },
        }
    )

    # In-sample edge. A large gap between this and the validation edge is the
    # signature of a hedger that memorised the training paths.
    train_paths = training_paths(cfg)
    pnl_c_train = classical.pnl(train_paths)
    pnl_d_train = learned.pnl(train_paths)
    train_edge = metrics.sharpe(pnl_d_train) - metrics.sharpe(pnl_c_train)
    summary["train_edge"] = float(train_edge)
    summary["generation_gap"] = float(train_edge - summary["edge_mean"])

    summary["val_records"] = val_records
    summary["train_histories"] = histories

    robust = {}
    for model in robust_models:
        learned_records, _ = evaluate_on_seeds(
            cfg, learned, model, cfg.val_seeds[:2], cfg.n_val_paths
        )
        classical_records, _ = evaluate_on_seeds(
            cfg, classical, model, cfg.val_seeds[:2], cfg.n_val_paths
        )
        robust[model] = float(
            np.mean(
                [
                    r["sharpe"] - c["sharpe"]
                    for r, c in zip(learned_records, classical_records)
                ]
            )
        )
    summary["robust_edges"] = robust
    return summary


def _quick_setup(cfg: BenchConfig, hyper: Hyper | None):
    """Shrink a run for smoke tests without disturbing the real pool sizes."""
    quick_cfg = BenchConfig(
        **{
            **cfg.__dict__,
            "n_train_paths": 512,
            "n_val_paths": 2000,
            "val_seeds": cfg.val_seeds[:2],
        }
    )
    quick_hyper = Hyper(
        **{**(hyper or Hyper()).__dict__, "epochs": 5, "torch_seeds": (0,)}
    )
    return quick_cfg, quick_hyper
