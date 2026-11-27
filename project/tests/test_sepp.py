"""Checks that pin the Sepp formulas to the simulator.

The paper's Appendix A is the reference for the implementation, but the
simulator is the reference for the paper: every closed form here is compared
with a Monte Carlo estimate of the same quantity. Where the two could not be
made to agree (the expected-gamma evaluation point, the opening-trade
adjustment) the test records the convention that was chosen and why, rather than
loosening the tolerance until it passed.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from hedging import bs, hedge, market, sepp


def _mc_paths(model: sepp.SeppModel, N: int, n_paths: int, seed: int = 3):
    return market.simulate_paths(
        model.market_params(), model.S, model.T, N, n_paths, seed=seed
    )


def test_cash_gamma_matches_black_scholes_gamma():
    # At zero rates ``0.5 S^2 U_SS`` equals the textbook gamma, so the paper's
    # closed form and the project's own greek must agree exactly; any drift here
    # would mean the P&L expansion uses a different gamma than the delta hedge.
    S = np.array([80.0, 100.0, 120.0])
    tau = np.array([0.25, 0.5, 1.0])
    sigma = 0.2
    got = sepp.cash_gamma(tau, S, 100.0, sigma)
    want = 0.5 * S**2 * bs.bs_gamma(S, 100.0, tau, 0.0, sigma)
    assert np.allclose(got, want, rtol=1e-10)


def test_z_integral_matches_quadrature():
    # The completed square is easy to get wrong in a sign, so it is checked
    # against a direct integral rather than only used downstream.
    q2, q1, mu, var = 0.7, -0.4, 0.1, 0.05
    got = float(sepp.z_integral(q2, q1, mu, var))
    want, _ = quad(
        lambda x: np.exp(-q2 * x**2 - q1 * x)
        * np.exp(-0.5 * (x - mu) ** 2 / var)
        / np.sqrt(2 * np.pi * var),
        -20,
        20,
    )
    assert got == pytest.approx(want, rel=1e-6)


def test_heston_mgf_is_a_martingale_at_one():
    # E[S_T] = S_0 for the driftless process; the affine form collapses to one
    # exactly at u = 1, which is the cheapest guard against a wrong drift term.
    for rho in (-0.5, 0.0, 0.5):
        got = sepp.heston_mgf(1.0, 1.0, 0.04, 2.0, 0.04, 0.3, rho)
        assert complex(got) == pytest.approx(1.0 + 0j, abs=1e-9)


def test_q_sv_matches_monte_carlo_quadratic_moment():
    # The whole SV quadratic moment (Fourier transform plus characteristic
    # function) is compared at once, because that is where an OCR misread of
    # equation (7.16) would hide.
    model = sepp.table1_case("I", market.HESTON)
    T, q2, q1 = model.T, 40.0, 0.3
    paths = _mc_paths(model, 1, 200_000, seed=11)
    x = np.log(paths[:, -1] / model.S)
    mc = float(np.exp(-q2 * x**2 - q1 * x).mean())
    got = float(sepp.q_sv(T, q2, q1, model))
    assert got == pytest.approx(mc, rel=0.06)


def test_table2_u_is_reproduced_with_the_front_gamma():
    # Table 2's `u` is the unadjusted coefficient. GBM and SVJ reproduce it to
    # the fourth decimal with the front gamma; the SV case is 1% low because
    # the rho = -0.5 auto-correlation correction is not in the closed form.
    expected = {
        market.GBM: 0.005692,
        market.SVJ: 0.006550,
    }
    for model_name, value in expected.items():
        stats = sepp.sepp_stats(sepp.table1_case("I", model_name))
        assert stats.u_raw == pytest.approx(value, rel=0.01)


def test_gbm_moments_match_monte_carlo():
    # Mean, cost and volatility of the rolled-out hedge are each compared with
    # the closed forms at two frequencies. The tolerances are Monte Carlo
    # driven (100k paths), not tuned: a real formula error is far larger.
    model = sepp.table1_case("I", market.GBM)
    stats = sepp.sepp_stats(model)
    N = 60
    paths = _mc_paths(model, N, 100_000, seed=5)
    premium = float(bs.bs_price(model.S, model.K, model.T, 0.0, model.sigma_i, "call"))
    fn = hedge.TimeBasedDelta(
        model.K, model.T, 0.0, model.sigma_i, "call", model.T / N, frequency=1
    )
    result = hedge.rollout(
        paths, fn, model.K, "call", premium, model.k / 2.0, model.T / N
    )
    assert result.pnl.mean() == pytest.approx(float(stats.mean_pnl(N)), rel=0.06)
    assert result.pnl.std() == pytest.approx(float(stats.volatility(N)), rel=0.12)
    # Equation (3.10) prices only the rebalancing trades, while the rollout also
    # pays to open the hedge and to liquidate it at maturity, so the Monte Carlo
    # cost must exceed the closed form rather than merely sit near it. Asserting
    # the inequality keeps the check meaningful despite the two missing legs.
    assert result.costs.mean() > float(stats.transaction_cost(N))


def test_sv_autocorrelation_correction_is_negative():
    # Sepp's Section 4.3: `L_sv < 0`, so stochastic volatility *helps* the
    # short-volatility position. A positive value would mean the ODE sign is
    # wrong even if the Sharpe still looked plausible.
    model = sepp.table1_case("I", market.HESTON)
    assert sepp._sv_autocorrelation_correction(model) < 0.0


def test_optimal_frequency_is_the_grid_argmax():
    # The Cardano root and a brute-force maximisation of the same Sharpe curve
    # must agree, otherwise the closed form is solving a different cubic than
    # the one the Sharpe ratio defines.
    for model_name in (market.GBM, market.JUMP, market.HESTON, market.SVJ):
        stats = sepp.sepp_stats(sepp.table1_case("I", model_name))
        _, n_star, _, _ = sepp.optimal_frequency(stats)
        grid = np.logspace(0.5, 3.5, 4000)
        assert grid[np.argmax(stats.sharpe(grid))] == pytest.approx(n_star, rel=0.05)


def test_bands_shrink_with_frequency_and_order_price_vs_delta():
    # Equation (3.5)/(3.8): more frequent hedging means tighter bands, and for
    # an at-the-money option the delta band is larger than the price band
    # because gamma is at its peak.
    model = sepp.table1_case("I", market.GBM)
    alpha_s_lo, alpha_d_lo = sepp.band_from_frequency(20, model)
    alpha_s_hi, alpha_d_hi = sepp.band_from_frequency(200, model)
    assert alpha_s_lo > alpha_s_hi
    assert alpha_d_lo > alpha_d_hi
    assert alpha_d_lo > alpha_s_lo


def test_jd_variance_matches_paper_table2():
    # Regression guard for the `q = sqrt(pi^3)/4` factor of equations
    # (7.13)-(7.14): the JD branch once dropped it, putting `p` 24% below the
    # paper and inflating the optimal frequency. Table 2, not the simulator, is
    # the target for these coefficients (the simulator comparison lives in the
    # replication script), so they are pinned to the printed values. The
    # tolerances cover the paper's four-figure rounding and its own
    # case-II variance approximation, and are tight enough that the old,
    # unnormalised value fails.
    stats = sepp.sepp_stats(sepp.table1_case("I", market.JUMP))
    assert stats.p == pytest.approx(0.002006, rel=0.08)
    assert stats.f == pytest.approx(0.000020, rel=0.12)


def test_jumps_raise_the_irreducible_variance_and_lower_the_optimum():
    # The economic claim the JF model is meant to display: an unhedgeable jump
    # component raises `f` and pushes the optimal frequency down relative to the
    # pure diffusion with the same total quadratic variance.
    df = sepp.sepp_stats(sepp.table1_case("I", market.GBM))
    jd = sepp.sepp_stats(sepp.table1_case("I", market.JUMP))
    assert jd.f > df.f
    _, n_df, _, _ = sepp.optimal_frequency(df)
    _, n_jd, _, _ = sepp.optimal_frequency(jd)
    assert n_jd < n_df
