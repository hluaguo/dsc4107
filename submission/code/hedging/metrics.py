"""P&L statistics used by the benchmark.

Every strategy is summarised by the same set of numbers so that no metric is
special-cased for one approach. The Sharpe ratio is the arbitrating score
(Sepp maximises it); CVaR is reported alongside so a strategy cannot win the
mean/variance race by quietly offloading risk into the tail.
"""

from __future__ import annotations

import numpy as np


def sharpe(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    sd = pnl.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return 0.0
    return float(pnl.mean() / sd)


def cvar95(pnl: np.ndarray) -> float:
    """Mean tail loss: average of the worst 5% of P&L, reported as a loss."""
    pnl = np.asarray(pnl, dtype=float)
    cut = np.quantile(pnl, 0.05)
    tail = pnl[pnl <= cut]
    if tail.size == 0:
        return 0.0
    return float(-tail.mean())


def summarize(pnl: np.ndarray) -> dict[str, float]:
    pnl = np.asarray(pnl, dtype=float)
    return {
        "sharpe": sharpe(pnl),
        "cvar95": cvar95(pnl),
        "mean": float(pnl.mean()),
        "std": float(pnl.std(ddof=1)),
        "median": float(np.median(pnl)),
        "skew": float(_safe_skew(pnl)),
    }


def _safe_skew(pnl: np.ndarray) -> float:
    pnl = np.asarray(pnl, dtype=float)
    sd = pnl.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.mean(((pnl - pnl.mean()) / sd) ** 3))


def paired_tstat(a: np.ndarray, b: np.ndarray) -> float:
    """t-statistic of the mean paired difference a - b.

    Both strategies are scored on the same simulated paths, so differencing
    removes the common market move and leaves mostly the strategy effect.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = d.size
    if n < 2:
        return 0.0
    sd = d.std(ddof=1)
    if sd <= 0:
        return 0.0
    return float(d.mean() / (sd / np.sqrt(n)))
