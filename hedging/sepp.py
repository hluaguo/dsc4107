"""Closed-form delta-hedging P&L statistics from Sepp (2013).

The simulator is the ground truth for this project, not these formulas. Every
quantity implemented here is checked against a Monte Carlo estimate in
``tests/test_sepp.py``, and where the paper admits more than one reading the
convention the simulation confirms is used and named in the docstring.

Unit notional is assumed (``S = K = 1``), time is in years and volatilities are
annualised, which is the convention of the paper's Section 5 tables.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.stats import norm

from . import market

# The variance of the realised variance under GBM contains an exact integral
# ``int_0^T T / sqrt(T^2 - t^2) dt = pi T / 2`` that the mid-point rule only
# approximates. Sepp (Appendix A, below equation 7.8) rescales the mid-point
# value by sqrt(pi**3)/4 so the approximation is exact in the limit; the same
# constant is kept here so the GBM variance agrees with Derman (1999) and with
# the paper rather than drifting a few percent high.
Q_DF = float(np.sqrt(np.pi**3) / 4.0)


@dataclass(frozen=True)
class SeppModel:
    """One of Sepp's four processes under the objective measure ``P``.

    Defaults are Table 1, case I (a liquid index option). ``sigma_i`` and ``k``
    describe the hedge, not the market: the option is always valued and
    delta-hedged at ``sigma_i`` even when the realised process has a different
    volatility, which is the whole point of the exercise.
    """

    model: str = market.GBM
    sigma_i: float = 0.165
    k: float = 0.001
    T: float = 1.0
    S: float = 1.0
    K: float = 1.0
    sigma_r: float = 0.15
    lam: float = 0.0
    nu: float = -0.05
    V: float = 0.0225
    theta: float = 0.0225
    kappa: float = 4.0
    xi: float = 0.25
    rho: float = -0.5
    # ``front`` evaluates the expected cash gamma at t=0, where the paper's
    # zeta(0) equals sigma_i; ``mid`` is the mid-point rule of equation (7.4).
    # The two differ by a few percent and the paper's own Table 2 follows
    # ``front`` (see ``tests/test_sepp.py::test_gamma_convention_matches_paper``),
    # so ``front`` is the default and ``mid`` is kept only for the comparison.
    gamma_convention: str = "front"

    @property
    def is_jump(self) -> bool:
        return self.model in (market.JUMP, market.SVJ)

    @property
    def is_sv(self) -> bool:
        return self.model in (market.HESTON, market.SVJ)

    @property
    def v0(self) -> float:
        return self.V if self.is_sv else 0.0

    def market_params(self) -> market.MarketParams:
        """Adapter to the simulator.

        ``sigma_j = 0`` because Sepp models the jump size as deterministic:
        letting the simulator draw a random jump size would test a model the
        paper never claims and would move the realised quadratic variance away
        from ``sigma_r^2 + lam nu^2``.
        """
        return market.MarketParams(
            model=self.model,
            mu=0.0,
            sigma=self.sigma_r,
            lam=self.lam,
            mu_j=self.nu,
            sigma_j=0.0,
            v0=self.V if self.is_sv else 0.04,
            kappa=self.kappa,
            theta=self.theta,
            xi=self.xi,
            rho=self.rho,
        )


def cash_gamma(tau, S=1.0, K=1.0, sigma=0.165):
    """Sepp's option cash gamma, equation (2.4): ``0.5 S^2 U_SS``.

    The closed form is used directly rather than via ``bs.bs_gamma`` because it
    is the object that appears multiplied into the P&L expansion, and because
    at zero rates the two agree exactly (a useful cross-check in the tests).
    ``tau <= 0`` returns zero instead of raising: the analytic sums evaluate
    the gamma one step before maturity, where the option is worth intrinsic.
    """
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        d2 = (np.log(S / K) - 0.5 * tau * sigma**2) / (sigma * np.sqrt(tau))
        value = K * norm.pdf(d2) / (2.0 * sigma * np.sqrt(tau))
    return np.where((tau > 0) & (sigma > 0), value, 0.0)


def z_integral(q2, q1, mu, var):
    """``E[exp(-q2 X^2 - q1 X)]`` for ``X ~ N(mu, var)``, equation (7.2).

    The paper's printed completed-square constants (``B = q2 var``,
    ``C = (2 mu q2 + q1) sqrt(var)``) do not reproduce a direct quadrature of
    the same integral, so the substitution is re-derived here: completing the
    square gives ``A2 = q2 + 1/(2 var)``, ``A1 = q1 - mu/var``,
    ``A0 = mu^2/(2 var)`` and the result below. ``test_z_integral_matches_
    quadrature`` pins the two together; the exact form is both simpler and
    faster than the printed one.
    """
    q2 = np.asarray(q2, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    A2 = q2 + 0.5 / var
    A1 = q1 - mu / var
    A0 = 0.5 * mu**2 / var
    # One exp and one sqrt: the printed two-exp form overflows in the tail of
    # the maturity sweep even though the combined exponent stays bounded.
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(A1**2 / (4.0 * A2) - A0) / np.sqrt(2.0 * A2 * var)


def _q2_q1(T, S, K, sigma_i):
    """Fourier parameters shared by equations (7.6), (7.11) and (7.16).

    They are written with the *full* maturity ``T`` even where the paper
    evaluates the resulting moment at ``T/2``; the asymmetry is Sepp's and is
    reproduced rather than smoothed over, because changing it moves the GBM
    variance by more than the Monte Carlo error.
    """
    half = 0.5 * T * sigma_i**2
    q2 = 1.0 / half
    q1 = 2.0 * (np.log(S / K) - 0.25 * T * sigma_i**2) / half
    return q2, q1


def heston_mgf(u, T, v0, kappa, theta, xi, rho):
    """``E[exp(u X_T)]`` for driftless Heston with ``X = log(S_T/S_0)``.

    Uses the standard affine (Heston 1993 / Lipton 2002) closed form rather
    than the paper's ``alpha, beta`` pair because the OCR of equation (7.16)
    does not pin down the shift convention, while this form is unambiguous and
    is verified against the simulator in the tests. Complex ``u`` is required:
    the quadratic term is turned into a cosine by a Fourier transform, so the
    characteristic function is evaluated on the imaginary axis.
    """
    u = np.asarray(u, dtype=complex)
    a = kappa - rho * xi * u
    d = np.sqrt(a**2 + xi**2 * (u - u**2) + 0j)
    g = (a - d) / (a + d)
    exp_neg = np.exp(-d * T)
    D = (a - d) / xi**2 * (1.0 - exp_neg) / (1.0 - g * exp_neg)
    C = (kappa * theta) / xi**2 * (
        (a - d) * T - 2.0 * np.log((1.0 - g * exp_neg) / (1.0 - g))
    )
    return np.exp(C + D * v0)


def q_sv(T, q2, q1, model: SeppModel):
    """``E[exp(-q2 X^2 - q1 X)]`` under Heston, equation (7.16).

    ``exp(-q2 x^2)`` is written as an integral of ``cos(k x)``; taking the real
    part of the characteristic function then gives the expectation. The
    integrand carries ``exp(-k^2/(4 q2))``, so the tail is negligible far
    before the integration limit and the quadrature is safe. The same routine
    serves the SVJ case: the compensated jump factor multiplies the Heston
    moment, which is equation (7.17) written in the time domain.
    """

    def integrand(k: float) -> float:
        u = -q1 + 1j * k
        mgf = heston_mgf(u, T, model.v0, model.kappa, model.theta, model.xi, model.rho)
        if model.is_jump:
            mgf = mgf * np.exp(
                model.lam * T * (np.exp(u * model.nu) - 1.0)
                - u * model.lam * (np.exp(model.nu) - 1.0) * T
            )
        return float(np.exp(-k**2 / (4.0 * q2)) * np.real(mgf))

    upper = 20.0 * np.sqrt(q2) + 10.0
    value, _ = quad(integrand, 0.0, upper, limit=400)
    return value / np.sqrt(np.pi * q2)


def q_jd(T, q2, q1, model: SeppModel, n_terms: int = 60):
    """Jump-diffusion quadratic moment, equation (7.11).

    Conditioned on ``m`` jumps the log-return is Gaussian, so the Poisson sum
    of Gaussian moments is exact up to truncation. The mean carries the
    martingale compensator ``-lam (e^nu - 1) T`` that the simulator applies;
    without it the analytic mean would disagree with the simulation by the
    whole jump drift, which at ``nu = -0.05`` is larger than the P&L itself.
    ``n_terms = 60`` is far into the tail of a Poisson(1) law.
    """
    compensator = model.lam * (np.exp(model.nu) - 1.0) * T
    total = 0.0
    for m in range(n_terms):
        weight = np.exp(-model.lam * T) * (model.lam * T) ** m / math.factorial(m)
        mean = -0.5 * model.sigma_r**2 * T + m * model.nu - compensator
        total += weight * z_integral(q2, q1, mean, model.sigma_r**2 * T)
    return float(total)


def q_svj(T, q2, q1, model: SeppModel):
    """SVJ moment: Heston with the jump factor, equation (7.17)."""
    return q_sv(T, q2, q1, model)


def _gamma_at(tau, model: SeppModel) -> float:
    return float(cash_gamma(tau, model.S, model.K, model.sigma_i))


def _expected_gamma(model: SeppModel) -> float:
    """Cash gamma entering the mean P&L.

    Sepp's equation (7.4) places it at the mid-point of the hedge grid, but the
    paper's Table 2 reproduces to the fourth decimal only with the ``front``
    value ``Gamma(0, T, S; sigma_i)``. The test suite records both, so the
    divergence is visible rather than silently absorbed.
    """
    if model.gamma_convention == "mid":
        sigma_eff = np.sqrt((model.sigma_r**2 + model.sigma_i**2) / 2.0)
        return float(cash_gamma(model.T, model.S, model.K, sigma_eff))
    return _gamma_at(model.T, model)


def _gamma2(model: SeppModel) -> float:
    """Second moment of the cash gamma, equations (7.6)/(7.13)/(7.19).

    Only this second moment keeps the mid-point evaluation and the model's
    quadratic moment: it sits under the ``p/N`` term, which is the part of the
    variance that hedging frequency removes, so the extra accuracy is worth
    the evaluation at ``T/2``.
    """
    half = 0.5 * model.T
    base = _gamma_at(half, model) ** 2
    q2, q1 = _q2_q1(model.T, model.S, model.K, model.sigma_i)
    if model.model == market.GBM:
        return base * float(
            z_integral(
                q2,
                q1,
                -0.25 * model.T * model.sigma_r**2,
                0.5 * model.T * model.sigma_r**2,
            )
        )
    if model.model == market.JUMP:
        return base * q_jd(half, q2, q1, model)
    if model.model == market.HESTON:
        return base * q_sv(half, q2, q1, model)
    if model.model == market.SVJ:
        return base * q_svj(half, q2, q1, model)
    raise ValueError(f"unknown model {model.model!r}")


def _sv_conditional_expectation(t, Psi, model: SeppModel) -> float:
    """``G(t, V) = E[(V(t) - E V(t)) exp(-Psi I(t))]`` for CIR variance.

    The paper gives this in closed form through ``A_theta, B_theta``
    (equation 8.4), but that pair is mis-transcribed by a factor of about 20 in
    the source used here (the OCR loses the variance scaling), so the equivalent
element is derived instead from the affine ODE and verified against Monte Carlo
    in ``tests/test_sepp.py``. This removes the guessing: with
    ``M(lambda) = E[exp(-Psi I(t)) exp(-lambda V(t))] = exp(phi + psi V)``,
    the ODEs are ``psi' = 0.5 xi^2 psi^2 - kappa psi - Psi``,
    ``phi' = kappa theta psi`` and their ``lambda``-derivatives give exactly
    ``G = -M(0) (phi_lambda + psi_lambda V + E V(t))``. The variational
    equations are integrated alongside so the derivative is not a finite
    difference.
    """
    kappa, theta, xi, v0 = model.kappa, model.theta, model.xi, model.V

    def rhs(_s, y):
        psi, phi, psi_l, phi_l = y
        return (
            0.5 * xi**2 * psi**2 - kappa * psi - Psi,
            kappa * theta * psi,
            (xi**2 * psi - kappa) * psi_l,
            kappa * theta * psi_l,
        )

    sol = solve_ivp(
        rhs, (0.0, t), [-0.0, 0.0, -1.0, 0.0], rtol=1e-9, atol=1e-12
    )
    psi, phi, psi_l, phi_l = sol.y[:, -1]
    m0 = float(np.exp(phi + psi * v0))
    vbar = theta + (v0 - theta) * np.exp(-kappa * t)
    return -m0 * (phi_l + psi_l * v0 + vbar)


def _sv_autocorrelation_correction(model: SeppModel, n_grid: int = 96) -> float:
    """``L_sv``, the stochastic-volatility auto-correlation term (8.3).

    The realised variance and the cash gamma share the volatility path, so
    ``Ga''m''a`` is larger exactly where the P&L term is more negative; the
    correction is negative, i.e. it *raises* the expected P&L of the short
    position. It is derived for ``rho = 0``; at the Table 1 value
    ``rho = -0.5`` Sepp states the effect grows but stays second order, so the
    ``rho = 0`` value is used and the residual is left to the Monte Carlo
    check rather than hidden in a fudge factor.

    Quadrature on ``n_grid`` points is used instead of the paper's ``N``-term
    sum: the integrand is smooth away from ``t = T`` where the gamma vanishes,
    and 96 points already sit below the Monte Carlo noise. Each point costs one
    ODE solve, which is why the grid is not finer.
    """
    if not model.is_sv:
        return 0.0
    sigma_i = model.sigma_i

    def integrand(t: float) -> float:
        tau = model.T - t
        if tau <= 1e-9:
            return 0.0
        Psi = 0.5 / (tau * sigma_i**2)
        # C_{n-1} exp(-y^2/2) is the cash gamma itself (Appendix B); multiplying
        # by an extra 2 would double the correction and overshoot Table 2.
        return _gamma_at(tau, model) * _sv_conditional_expectation(t, Psi, model)

    grid = np.linspace(0.0, model.T, n_grid)
    values = np.array([integrand(t) for t in grid])
    return float(np.trapezoid(values, grid))


def _sv_variance_term(model: SeppModel) -> float:
    """``V_sv``, equation (7.21): variance of the integrated variance.

    This is the part of the P&L variance that survives any hedging frequency,
    so it is the term that makes the Sharpe ratio flatten out for large ``N``.
    """
    kappa, theta, xi, T, V = model.kappa, model.theta, model.xi, model.T, model.V
    return (
        xi**2
        / (2.0 * kappa**3)
        * (
            (theta - 2.0 * V) * np.exp(-2.0 * kappa * T)
            + 4.0 * (theta * kappa * T - V * kappa * T + theta) * np.exp(-kappa * T)
            + (-5.0 * theta + 2.0 * theta * kappa * T + 2.0 * V)
        )
    )


@dataclass
class SeppStats:
    """The ``u, c, p, f`` tuple of equation (3.12), with its inputs kept.

    ``u`` is the frequency-independent expected P&L, ``c`` scales the expected
    cost as ``c sqrt(N)``, and the variance is ``p/N + f``. Keeping ``gamma``
    and ``gamma2`` makes the variance decomposition auditable: a table that
    moves because the gamma changed looks different from one that moves because
    the model's quadratic variance changed.
    """

    u: float
    c: float
    p: float
    f: float
    gamma: float
    gamma2: float
    initial_cost: float = 0.0
    notes: dict = field(default_factory=dict)

    def transaction_cost(self, N):
        return self.c * np.sqrt(np.asarray(N, dtype=float))

    def mean_pnl(self, N):
        return self.u - self.transaction_cost(N)

    def variance(self, N):
        return self.p / np.asarray(N, dtype=float) + self.f

    def volatility(self, N):
        return np.sqrt(np.maximum(self.variance(N), 0.0))

    def sharpe(self, N, T=1.0):
        """Equation (3.12), annualised by ``1/sqrt(T)``."""
        return (self.mean_pnl(N) / self.volatility(N)) / np.sqrt(T)

    @property
    def u_raw(self):
        """``u`` before the opening-trade adjustment, i.e. the value Table 2 quotes.

        The paper's text says the Sharpe coefficient should be reduced by
        ``0.5 k S |Delta_0|``, but its own Table 2 reports the unadjusted value
        and its optimal-frequency row ``u - C(N*)`` is consistent with that.
        Keeping both means the reproduction can be compared like for like and
        the simulator comparison can still use ``u``.
        """
        return self.u + self.initial_cost


def sepp_stats(model: SeppModel) -> SeppStats:
    """Assemble ``u, c, p, f`` from equations (4.3)-(4.19).

    The per-model differences are exactly: the expected quadratic variance
    ``theta_q`` (a constant for GBM/JD, a deterministic function of ``T`` for
    SV/SVJ), whether jumps add a linear cost term ``k lam T |nu| Gamma``, and
    whether SV contributes the auto-correlation correction ``L_sv``. Keeping
    those three switches explicit makes it obvious which model a number came
    from when the tables are compared.
    """
    T, S, K = model.T, model.S, model.K
    sigma_i, sigma_r, k = model.sigma_i, model.sigma_r, model.k
    gamma = _expected_gamma(model)
    # The irreducible-variance term is a variance, so both gammas in it are
    # evaluated at the same point; the paper's equation (7.4) mid-point value is
    # used here because it reproduces Table 2's `f` for GBM (a front value makes
    # `f` about twice the table) while the mean keeps the front value that
    # reproduces its `u`.
    gamma_mid = float(
        cash_gamma(
            T, S, K, np.sqrt((sigma_r**2 + sigma_i**2) / 2.0)
        )
    )
    gamma2 = _gamma2(model)
    delta0 = float(
        norm.cdf((np.log(S / K) + 0.5 * T * sigma_i**2) / (sigma_i * np.sqrt(T)))
    )
    # The opening trade is charged by the rollout too, so it belongs in the
    # analytic mean for the two to be comparable; omitting it is what makes a
    # naive formula look 0.03% too optimistic at these cost levels.
    initial_cost = 0.5 * k * S * abs(delta0)

    if model.model == market.GBM:
        theta_q = sigma_r**2
        u = (sigma_i**2 - theta_q) * T * gamma - initial_cost
        c = k * np.sqrt(2.0 * sigma_r**2 * T / np.pi) * gamma
        p = 2.0 * Q_DF * sigma_r**4 * T**2 * gamma2
        f = (sigma_i**2 - theta_q) ** 2 * T**2 * (gamma2 - gamma_mid**2)
    elif model.model == market.JUMP:
        theta_q = sigma_r**2 + model.lam * model.nu**2
        u = (
            (sigma_i**2 - theta_q) * T * gamma
            - k * model.lam * T * abs(model.nu) * gamma
            - initial_cost
        )
        c = k * np.sqrt(2.0 * sigma_r**2 * T / np.pi) * gamma
        p = gamma2 * (2.0 * sigma_r**4 + model.lam * sigma_r**2 * model.nu**2) * T**2
        f = (
            gamma2 * model.lam * T * model.nu**4
            + (sigma_i**2 - theta_q) ** 2 * T**2 * (gamma2 - gamma_mid**2)
        )
    elif model.model in (market.HESTON, market.SVJ):
        theta_q = theta_sv = (
            model.theta
            + (1.0 - np.exp(-model.kappa * T))
            / (model.kappa * T)
            * (model.V - model.theta)
        )
        L = _sv_autocorrelation_correction(model)
        if model.is_jump:
            theta_q = theta_sv + model.lam * model.nu**2
            u = (
                (sigma_i**2 - theta_q) * T * gamma
                - L
                - k * model.lam * T * abs(model.nu) * gamma
                - initial_cost
            )
        else:
            u = (sigma_i**2 - theta_q) * T * gamma - L - initial_cost
        # The cost coefficient uses the *diffusive* quadratic variance even in
        # the SVJ case (equation 4.19): jumps are charged separately and
        # linearly, not through the sqrt(N) term.
        c = k * np.sqrt(2.0 * theta_sv * T / np.pi) * gamma
        # Equation (4.16)/(4.17): the SV variance rate carries the same
        # sqrt(pi^3)/4 normalisation as the GBM one, which is what lifts `p`
        # from 0.0019 to the paper's 0.0025.
        f1 = gamma2 * _sv_variance_term(model)
        if model.is_jump:
            p = Q_DF * gamma2 * (
                2.0 * model.V**2 * T**2
                + model.lam * sigma_r**2 * model.nu**2 * T**2
            )
            f2 = Q_DF * gamma2 * model.lam * T * model.nu**4
        else:
            p = Q_DF * gamma2 * 2.0 * model.V**2 * T**2
            f2 = 0.0
        f = f1 + f2 + (sigma_i**2 - theta_q) ** 2 * T**2 * (gamma2 - gamma_mid**2)
    else:
        raise ValueError(f"unknown model {model.model!r}")

    return SeppStats(
        u=float(u),
        c=float(c),
        p=float(p),
        f=float(f),
        gamma=float(gamma),
        gamma2=float(gamma2),
        initial_cost=float(initial_cost),
        notes={"theta_q": float(theta_q)},
    )


def optimal_frequency(stats: SeppStats):
    """Sharpe maximising frequency, equations (3.13)-(3.14).

    The stationarity condition reduces to the cubic ``p u m^3 - 2 c p m^2 -
    c f = 0`` in ``m = 1/sqrt(N)``. The cubic is solved with a polynomial root
    finder rather than the printed Cardano branch: the Cardano expression in the
    paper requires ``D = R^2 - Q^3 > 0``, and at these parameters ``D`` is
    around 1e-10 and flips sign between models, so the closed form is not a
    reliable branch. The roots are cheap and the selection rule (positive real
    root that maximises the Sharpe curve) is explicit. The returned tuple keeps
    the cubic discriminant so a caller can see when the problem sat on the
    boundary.
    """
    u, c, p, f = stats.u, stats.c, stats.p, stats.f
    if not all(np.isfinite([u, c, p, f])) or p <= 0 or u <= 0:
        grid = np.logspace(0.0, 4.0, 20001)
        best = float(grid[np.argmax(stats.sharpe(grid))])
        return float(round(best)), best, float("nan"), "grid"
    roots = np.roots([p * u, -2.0 * c * p, 0.0, -c * f])
    candidates = []
    for root in roots:
        if abs(root.imag) > 1e-9 or root.real <= 0.0:
            continue
        candidates.append(1.0 / root.real**2)
    if not candidates:
        grid = np.logspace(0.0, 4.0, 20001)
        best = float(grid[np.argmax(stats.sharpe(grid))])
        return float(round(best)), best, float("nan"), "grid"
    n_star = max(candidates, key=lambda n: float(stats.sharpe(n)))
    return float(round(n_star)), float(n_star), 0.0, "cubic"


# Table 1, reproduced so the notebooks and script cannot silently disagree
# about what "case I" means. The two cases are a liquid index option and a
# high-beta single-stock option: the second has twice the volatility and four
# times the spread, which is the regime the paper uses to show the optimum
# moving to a lower frequency.
_CASES = {
    ("I", market.GBM): dict(sigma_i=0.165, k=0.001, sigma_r=0.15),
    ("I", market.JUMP): dict(
        sigma_i=0.165, k=0.001, sigma_r=0.1414, lam=1.0, nu=-0.05
    ),
    ("I", market.HESTON): dict(
        sigma_i=0.165, k=0.001, sigma_r=0.15, V=0.0225, theta=0.0225,
        kappa=4.0, xi=0.25, rho=-0.5,
    ),
    ("I", market.SVJ): dict(
        sigma_i=0.165, k=0.001, sigma_r=0.1414, lam=1.0, nu=-0.05,
        V=0.02, theta=0.02, kappa=4.0, xi=0.25, rho=-0.5,
    ),
    ("II", market.GBM): dict(sigma_i=0.30, k=0.004, sigma_r=0.25),
    ("II", market.JUMP): dict(
        sigma_i=0.30, k=0.004, sigma_r=0.2291, lam=1.0, nu=-0.10
    ),
    ("II", market.HESTON): dict(
        sigma_i=0.30, k=0.004, sigma_r=0.25, V=0.0625, theta=0.0625,
        kappa=4.0, xi=0.50, rho=-0.5,
    ),
    ("II", market.SVJ): dict(
        sigma_i=0.30, k=0.004, sigma_r=0.2291, lam=1.0, nu=-0.10,
        V=0.0525, theta=0.0525, kappa=4.0, xi=0.50, rho=-0.5,
    ),
}


def table1_case(case: str, model: str) -> SeppModel:
    """Sepp's Table 1 parameter set, case ``"I"`` or ``"II"``.

    The JD/SVJ cases set ``sigma_r`` so that ``sigma_r^2 + lam nu^2`` equals the
    case's target quadratic variance (15%^2 or 25%^2). That is what makes the
    four models comparable: they share a total realised variance and differ only
    in how it is delivered, so a Sharpe difference is attributable to the
    hedging problem and not to a different amount of risk.
    """
    case = case.upper()
    key = (case, model)
    if key not in _CASES:
        raise ValueError(f"no Table 1 parameters for case {case!r}, model {model!r}")
    return SeppModel(model=model, T=1.0, S=1.0, K=1.0, **_CASES[key])


def band_from_frequency(N, model: SeppModel):
    """Price and delta bands for an expected frequency, equations (3.5)/(3.8).

    The price band is the diffusion result ``sqrt(T) sigma_r / sqrt(N)``;
    for the jump and SV models Sepp substitutes the model's quadratic variance
    for ``sigma_r^2`` (Section 3), which is the heuristic this follows. The
    delta band converts through ``S U_SS`` evaluated at the mixed volatility
    of equation (3.7).
    """
    N = float(N)
    quadratic = model.sigma_r**2 + model.lam * model.nu**2
    alpha_s = np.sqrt(model.T) * np.sqrt(quadratic) / np.sqrt(N)
    sigma_mix = np.sqrt((quadratic + model.sigma_i**2) / 2.0)
    gamma_mid = float(cash_gamma(0.5 * model.T, model.S, model.K, sigma_mix))
    s_uss = 2.0 * gamma_mid / model.S
    alpha_delta = alpha_s * abs(model.S * s_uss)
    return float(alpha_s), float(alpha_delta)
