"""Market simulators for the four processes used by Sepp (2013).

Each simulator returns a price path array of shape ``(n_paths, n_steps + 1)``
including the initial spot in column 0. The models are simulated under the
statistical measure, matching the paper's distinction between the process that
moves the underlying and the Black-Scholes recipe used to hedge.

Heston uses the full truncation Euler scheme, which keeps the variance in the
recursion (possibly negative) and uses its positive part in the drift and
diffusion. That is accurate enough at the step sizes used here and avoids the
positivity hacks that bias the variance upward.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

GBM = "gbm"
JUMP = "jd"
HESTON = "heston"
SVJ = "svj"


@dataclass(frozen=True)
class MarketParams:
    """Frozen so a variant can be derived with ``replace`` rather than copied,
    and so parallel evaluations cannot mutate shared state."""

    model: str = HESTON
    mu: float = 0.0
    sigma: float = 0.20
    # Merton jump component (intensity per year)
    lam: float = 1.0
    mu_j: float = -0.05
    sigma_j: float = 0.15
    # Heston variance component
    v0: float = 0.04
    kappa: float = 2.0
    theta: float = 0.04
    xi: float = 0.30
    rho: float = -0.70

    def with_model(self, model: str) -> "MarketParams":
        return replace(self, model=model)


def default_params() -> MarketParams:
    return MarketParams()


def _gbm_increments(mp, dt, n_paths, n_steps, rng):
    z = rng.standard_normal((n_paths, n_steps))
    drift = (mp.mu - 0.5 * mp.sigma**2) * dt
    return drift + mp.sigma * np.sqrt(dt) * z


def _jump_increments(mp, dt, n_paths, n_steps, rng):
    """Compound-Poisson jump increments with the compensator removed.

    ``reduceat`` sums each path-step's jump sizes in one pass; a Python loop
    over the jump counts would cost more than the rest of the simulator.
    """
    counts = rng.poisson(mp.lam * dt, size=(n_paths, n_steps))
    jumps = np.zeros((n_paths, n_steps), dtype=float)
    mask = counts > 0
    if mask.any():
        sizes = counts[mask].astype(np.int64)
        total = int(sizes.sum())
        draws = rng.normal(mp.mu_j, mp.sigma_j, size=total)
        starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
        jumps[mask] = np.add.reduceat(draws, starts)
    compensator = np.exp(mp.mu_j + 0.5 * mp.sigma_j**2) - 1.0
    return jumps - mp.lam * compensator * dt


def _heston_log_increments(mp, dt, n_paths, n_steps, rng):
    """Full truncation Euler for the Heston log-price and variance."""
    z1 = rng.standard_normal((n_paths, n_steps))
    z2 = rng.standard_normal((n_paths, n_steps))
    z_v = mp.rho * z1 + np.sqrt(1.0 - mp.rho**2) * z2
    d_w1 = np.sqrt(dt) * z1
    d_w2 = np.sqrt(dt) * z_v

    v = np.full(n_paths, mp.v0, dtype=float)
    log_inc = np.empty((n_paths, n_steps), dtype=float)
    for j in range(n_steps):
        v_pos = np.maximum(v, 0.0)
        log_inc[:, j] = (mp.mu - 0.5 * v_pos) * dt + np.sqrt(v_pos) * d_w1[:, j]
        v = v + mp.kappa * (mp.theta - v_pos) * dt + mp.xi * np.sqrt(v_pos) * d_w2[:, j]
    return log_inc


def simulate_paths(
    mp: MarketParams,
    S0: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: int,
) -> np.ndarray:
    """Return paths with the initial spot in column 0.

    The rollout reads the spot at every grid point including t=0, so the
    starting value belongs in the array rather than being prepended by each
    caller.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    if mp.model == GBM:
        log_inc = _gbm_increments(mp, dt, n_paths, n_steps, rng)
    elif mp.model == JUMP:
        log_inc = _gbm_increments(mp, dt, n_paths, n_steps, rng)
        log_inc = log_inc + _jump_increments(mp, dt, n_paths, n_steps, rng)
    elif mp.model == HESTON:
        log_inc = _heston_log_increments(mp, dt, n_paths, n_steps, rng)
    elif mp.model == SVJ:
        log_inc = _heston_log_increments(mp, dt, n_paths, n_steps, rng)
        log_inc = log_inc + _jump_increments(mp, dt, n_paths, n_steps, rng)
    else:
        raise ValueError(f"unknown model {mp.model!r}")

    log_paths = np.concatenate(
        [np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1
    )
    return S0 * np.exp(log_paths)
