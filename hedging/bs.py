"""Black-Scholes pricing and greeks.

The classical hedger in this project uses the textbook recipe: hold the
Black-Scholes delta computed with a constant implied volatility, regardless of
the process that actually drives the underlying. Keeping these functions exact
matters because the frictionless deep hedger is checked against them.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


def _d1_d2(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol = sigma * np.sqrt(T)
        d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol
    return d1, d1 - vol


def bs_price(S, K, T, r, sigma, kind="call"):
    """Price a European option, falling back to intrinsic value at expiry.

    The fallback matters inside the hedging rollout: the final decision point
    sits one step before maturity, and a degenerate time or volatility would
    otherwise put NaNs into the recursion and poison every later step.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if kind == "call":
        price = call
        intrinsic = np.maximum(S - K, 0.0)
    elif kind == "put":
        price = call - S + K * np.exp(-r * T)
        intrinsic = np.maximum(K - S, 0.0)
    else:
        raise ValueError(f"unknown kind {kind!r}")
    return np.where((T > 0) & (sigma > 0), price, intrinsic)


def bs_delta(S, K, T, r, sigma, kind="call"):
    d1, _ = _d1_d2(S, K, T, r, sigma)
    base = norm.cdf(d1)
    if kind == "call":
        delta = base
        limit = (np.asarray(S) > np.asarray(K)).astype(float)
    else:
        delta = base - 1.0
        limit = -(np.asarray(S) < np.asarray(K)).astype(float)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return np.where((T > 0) & (sigma > 0), delta, limit)


def bs_gamma(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = np.asarray(T, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return np.where((T > 0) & (sigma > 0), g, 0.0)


def bs_vega(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    d1, _ = _d1_d2(S, K, T, r, sigma)
    T = np.asarray(T, dtype=float)
    return S * norm.pdf(d1) * np.sqrt(T)


def implied_vol(price, S, K, T, r, kind="call"):
    """Invert Black-Scholes for the implied volatility.

    Brent's method is used instead of Newton because the price is monotone in
    volatility under the flat rate used here, whereas Newton divides by vega,
    which collapses for deep out-of-the-money options and stalls. Inputs are
    scalars because calibration happens once per contract, not per path.
    """
    if T <= 0 or price <= 0:
        return np.nan

    def objective(sigma):
        return float(bs_price(S, K, T, r, sigma, kind)) - float(price)

    try:
        return brentq(objective, 1e-6, 5.0, xtol=1e-12, maxiter=200)
    except ValueError:
        return np.nan
