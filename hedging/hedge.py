"""The hedging rollout shared by the classical and learned strategies.

Both approaches are scored through one accounting routine. Giving the network
its own P&L formula would make the comparison depend on the bookkeeping rather
than on the hedging, so this engine is deliberately the only route from
positions to profit and loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from . import bs

PositionFn = Callable[[int, np.ndarray, np.ndarray], np.ndarray]


@dataclass
class RolloutResult:
    pnl: np.ndarray
    costs: np.ndarray
    turnover: np.ndarray


def rollout(
    paths: np.ndarray,
    position_fn: PositionFn,
    K: float,
    kind: str,
    premium: float,
    cost: float,
    dt: float,
) -> RolloutResult:
    """Run one strategy over a batch of paths and return its cash P&L.

    Cash is tracked instead of accumulating a closed-form P&L expression, so the
    premium, the trades and the terminal liquidation net out by construction
    with no discount factor to get wrong. The residual hedge is sold at
    maturity, which penalises a strategy that leaves exposure open to the end.
    """
    n_paths, n_cols = paths.shape
    n_steps = n_cols - 1
    pos = np.zeros(n_paths)
    cash = np.full(n_paths, float(premium))
    costs = np.zeros(n_paths)
    turnover = np.zeros(n_paths)

    for j in range(n_steps):
        spot = paths[:, j]
        new_pos = position_fn(j, spot, pos)
        traded = new_pos - pos
        step_cost = cost * np.abs(traded) * spot
        cash -= traded * spot + step_cost
        costs += step_cost
        turnover += np.abs(traded)
        pos = new_pos

    terminal = paths[:, -1]
    close_cost = cost * np.abs(pos) * terminal
    costs += close_cost
    turnover += np.abs(pos)
    cash += pos * terminal - close_cost - _payoff(terminal, K, kind)
    return RolloutResult(cash, costs, turnover)


def _payoff(terminal: np.ndarray, K: float, kind: str) -> np.ndarray:
    if kind == "call":
        return np.maximum(terminal - K, 0.0)
    if kind == "put":
        return np.maximum(K - terminal, 0.0)
    raise ValueError(f"unknown kind {kind!r}")


class TimeBasedDelta:
    """Rebalance to the Black-Scholes delta every ``frequency`` steps.

    This is Sepp's baseline shape: the strategy is fixed and only the clock
    varies, so the experiment isolates hedging frequency instead of mixing it
    with strategy design.
    """

    def __init__(self, K, T, r, sigma_impl, kind, dt, frequency):
        if frequency < 1:
            raise ValueError("frequency must be at least 1")
        self.K = K
        self.T = T
        self.r = r
        self.sigma_impl = sigma_impl
        self.kind = kind
        self.dt = dt
        self.frequency = int(frequency)

    def __call__(self, j, spot, pos):
        if j % self.frequency:
            return pos
        tau = max(self.T - j * self.dt, 0.0)
        return bs.bs_delta(spot, self.K, tau, self.r, self.sigma_impl, self.kind)


class BandDelta:
    """Rebalance only when price or delta drifts past a threshold.

    The anchor is the spot at the previous trade, which is what makes a price
    band adaptive: a quiet market trades rarely and a fast one trades often, so
    the fee bill tracks realised volatility rather than the calendar.
    """

    def __init__(
        self,
        K,
        T,
        r,
        sigma_impl,
        kind,
        dt,
        delta_band: Optional[float] = None,
        price_band: Optional[float] = None,
    ):
        if delta_band is None and price_band is None:
            raise ValueError("at least one band is required")
        self.K = K
        self.T = T
        self.r = r
        self.sigma_impl = sigma_impl
        self.kind = kind
        self.dt = dt
        self.delta_band = delta_band
        self.price_band = price_band
        self.anchor: Optional[np.ndarray] = None

    def __call__(self, j, spot, pos):
        tau = max(self.T - j * self.dt, 0.0)
        target = bs.bs_delta(spot, self.K, tau, self.r, self.sigma_impl, self.kind)
        if self.anchor is None:
            self.anchor = np.asarray(spot, dtype=float).copy()
            return target
        trigger = np.zeros(np.shape(spot), dtype=bool)
        if self.delta_band is not None:
            trigger |= np.abs(target - pos) > self.delta_band
        if self.price_band is not None:
            trigger |= np.abs(spot / self.anchor - 1.0) > self.price_band
        new_pos = np.where(trigger, target, pos)
        self.anchor = np.where(trigger, spot, self.anchor)
        return new_pos
