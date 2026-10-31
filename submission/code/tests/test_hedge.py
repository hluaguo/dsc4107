import numpy as np
import pytest

from hedging import bs, hedge, market


def _paths(model=market.GBM, n=20_000, steps=21, seed=5, T=1 / 12):
    return market.simulate_paths(
        market.MarketParams(model=model), 100.0, T, steps, n, seed=seed
    )


def test_rollout_without_a_hedge_returns_premium_minus_payoff():
    paths = _paths()
    dt = (1 / 12) / 21
    premium = float(bs.bs_price(100.0, 100.0, 1 / 12, 0.0, 0.2, "call"))
    result = hedge.rollout(
        paths,
        lambda j, spot, pos: np.zeros_like(spot),
        100.0,
        "call",
        premium,
        0.0,
        dt,
    )
    payoff = np.maximum(paths[:, -1] - 100.0, 0.0)
    assert np.allclose(result.pnl, premium - payoff)
    assert np.all(result.turnover == 0)


def test_finer_hedging_reduces_error_variance_in_gbm():
    paths = _paths(n=40_000)
    dt = (1 / 12) / 21
    premium = float(bs.bs_price(100.0, 100.0, 1 / 12, 0.0, 0.2, "call"))

    def run(freq):
        fn = hedge.TimeBasedDelta(
            100.0, 1 / 12, 0.0, 0.2, "call", dt, frequency=freq
        )
        return hedge.rollout(paths, fn, 100.0, "call", premium, 0.0, dt).pnl

    coarse = run(7)
    fine = run(1)
    assert fine.std() < coarse.std()


def test_transaction_costs_lower_the_mean_pnl():
    paths = _paths()
    dt = (1 / 12) / 21
    premium = float(bs.bs_price(100.0, 100.0, 1 / 12, 0.0, 0.2, "call"))
    fn = lambda j, spot, pos: hedge.TimeBasedDelta(
        100.0, 1 / 12, 0.0, 0.2, "call", dt, 1
    )(j, spot, pos)
    free = hedge.rollout(paths, fn, 100.0, "call", premium, 0.0, dt)
    costly = hedge.rollout(paths, fn, 100.0, "call", premium, 0.005, dt)
    assert costly.pnl.mean() < free.pnl.mean()
    assert costly.costs.mean() > 0


def test_price_band_ignores_small_moves():
    # A gentle drift below the band threshold is where the two strategies should
    # diverge: the band holds its opening hedge, while the clock chases every
    # small change in spot and time.
    ramp = np.linspace(100.0, 100.8, 22)
    paths = np.tile(ramp, (4, 1))
    dt = (1 / 12) / 21
    premium = float(bs.bs_price(100.0, 100.0, 1 / 12, 0.0, 0.2, "call"))

    band = hedge.BandDelta(100.0, 1 / 12, 0.0, 0.2, "call", dt, price_band=0.02)
    band_result = hedge.rollout(paths, band, 100.0, "call", premium, 0.0, dt)

    time = hedge.TimeBasedDelta(100.0, 1 / 12, 0.0, 0.2, "call", dt, 1)
    time_result = hedge.rollout(paths, time, 100.0, "call", premium, 0.0, dt)
    assert band_result.turnover.mean() < time_result.turnover.mean()
