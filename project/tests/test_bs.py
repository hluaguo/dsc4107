import numpy as np
import pytest

from hedging import bs


def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 105.0, 0.75, 0.03, 0.2
    call = bs.bs_price(S, K, T, r, sigma, "call")
    put = bs.bs_price(S, K, T, r, sigma, "put")
    assert call - put == pytest.approx(S - K * np.exp(-r * T), abs=1e-10)


def test_delta_matches_finite_difference():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.02, 0.25
    eps = 1e-4
    analytic = bs.bs_delta(S, K, T, r, sigma, "call")
    up = bs.bs_price(S + eps, K, T, r, sigma, "call")
    down = bs.bs_price(S - eps, K, T, r, sigma, "call")
    assert analytic == pytest.approx((up - down) / (2 * eps), abs=1e-6)


def test_gamma_matches_finite_difference_of_delta():
    S, K, T, r, sigma = 100.0, 100.0, 0.5, 0.02, 0.25
    eps = 1e-3
    analytic = bs.bs_gamma(S, K, T, r, sigma)
    up = bs.bs_delta(S + eps, K, T, r, sigma, "call")
    down = bs.bs_delta(S - eps, K, T, r, sigma, "call")
    assert analytic == pytest.approx((up - down) / (2 * eps), abs=1e-5)


def test_implied_vol_round_trip():
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.01, 0.35
    price = bs.bs_price(S, K, T, r, sigma, "call")
    assert bs.implied_vol(price, S, K, T, r, "call") == pytest.approx(sigma, abs=1e-8)


def test_price_falls_back_to_intrinsic_at_expiry():
    assert bs.bs_price(120.0, 100.0, 0.0, 0.0, 0.2, "call") == pytest.approx(20.0)
    assert bs.bs_price(80.0, 100.0, 0.0, 0.0, 0.2, "put") == pytest.approx(20.0)
    assert np.isfinite(bs.bs_delta(120.0, 100.0, 0.0, 0.0, 0.2, "call"))
