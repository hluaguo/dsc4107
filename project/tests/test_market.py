import numpy as np
import pytest

from hedging import market


def test_shapes_and_initial_spot():
    for model in (market.GBM, market.JUMP, market.HESTON, market.SVJ):
        paths = market.simulate_paths(
            market.MarketParams(model=model), 100.0, 1.0, 12, 500, seed=7
        )
        assert paths.shape == (500, 13)
        assert np.all(paths[:, 0] == 100.0)
        assert np.all(np.isfinite(paths))
        assert np.all(paths > 0)


def test_gbm_matches_analytic_mean_and_variance():
    mp = market.MarketParams(model=market.GBM, mu=0.05, sigma=0.2)
    S0, T, n = 100.0, 1.0, 400_000
    paths = market.simulate_paths(mp, S0, T, 1, n, seed=1)
    terminal = paths[:, -1]
    expected_mean = S0 * np.exp(mp.mu * T)
    expected_var = (
        S0**2 * np.exp(2 * mp.mu * T) * (np.exp(mp.sigma**2 * T) - 1.0)
    )
    assert terminal.mean() == pytest.approx(expected_mean, rel=0.01)
    assert terminal.var() == pytest.approx(expected_var, rel=0.03)


def test_jump_model_has_heavier_tail_than_diffusion():
    base = market.simulate_paths(
        market.MarketParams(model=market.GBM, mu=0.0, sigma=0.2, lam=0.0),
        100.0,
        1.0,
        1,
        200_000,
        seed=3,
    )[:, -1]
    jump = market.simulate_paths(
        market.MarketParams(model=market.JUMP, mu=0.0, sigma=0.2, lam=1.0),
        100.0,
        1.0,
        1,
        200_000,
        seed=3,
    )[:, -1]
    assert jump.std() > base.std()


def test_heston_variance_stays_positive_in_distribution():
    paths = market.simulate_paths(
        market.MarketParams(model=market.HESTON), 100.0, 1.0, 50, 20_000, seed=11
    )
    assert np.all(paths > 0)
    assert np.quantile(paths[:, -1], 0.01) > 0
