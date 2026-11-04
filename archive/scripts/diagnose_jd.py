"""Diagnose the Sepp jump-diffusion variance gap against the simulator.

``REPORT.md`` records that the JD variance coefficients (``p`` and ``f``) sit
below the paper's Table 2 while the Monte Carlo volatility is well above the
analytic one. That is a real disagreement, and the plan says to locate the
faulty side rather than paper over it. This script does three things:

1. Fits ``p`` and ``f`` from the simulator by regressing the rolled-out P&L
   variance on ``1/N`` across hedging frequencies. The simulator is the ground
   truth in this project, so this is the reference.
2. Estimates ``E[Gamma(T/2)]`` and ``E[Gamma(T/2)^2]`` directly from simulated
   paths and compares them with the analytic ``gamma`` and ``gamma2``.
3. Prints the same three quantities for GBM as a control, where the formula and
   the paper already agree. A fault isolated to JD but absent in GBM points at
   the jump moment rather than the shared machinery.

Nothing here changes the frozen modules; it only measures them.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from hedging import bs, hedge, market, sepp  # noqa: E402

NS = (16, 32, 64, 128, 256, 512)
N_PATHS = 200_000
SEED = 4242


def fit_pf(model: sepp.SeppModel):
    inv, var = [], []
    for N in NS:
        paths = market.simulate_paths(
            model.market_params(), model.S, model.T, N, N_PATHS, seed=SEED + N
        )
        premium = float(bs.bs_price(model.S, model.K, model.T, 0.0, model.sigma_i, "call"))
        fn = hedge.TimeBasedDelta(
            model.K, model.T, 0.0, model.sigma_i, "call", model.T / N, frequency=1
        )
        result = hedge.rollout(paths, fn, model.K, "call", premium, 0.0, model.T / N)
        inv.append(1.0 / N)
        var.append(result.pnl.var(ddof=1))
    A = np.vstack([inv, np.ones_like(inv)]).T
    p, f = np.linalg.lstsq(A, np.asarray(var), rcond=None)[0]
    return float(p), float(f)


def mc_gammas(model: sepp.SeppModel, n_steps: int = 64):
    """E[Gamma(T/2)] and E[Gamma(T/2)^2] from simulated spot paths.

    ``n_steps`` even so one grid point falls exactly on ``T/2``; the gamma is
    evaluated at the implied Sigma_i, which is the object the analytic
    ``gamma2`` integrates.
    """
    paths = market.simulate_paths(
        model.market_params(), model.S, model.T, n_steps, N_PATHS, seed=SEED + 7
    )
    mid = paths[:, n_steps // 2]
    gamma = sepp.cash_gamma(0.5 * model.T, mid, model.K, model.sigma_i)
    return float(gamma.mean()), float((gamma**2).mean())


def main() -> None:
    for name, model in (
        ("gbm", sepp.table1_case("I", market.GBM)),
        ("jd", sepp.table1_case("I", market.JUMP)),
    ):
        st = sepp.sepp_stats(model)
        p_mc, f_mc = fit_pf(model)
        g_mc, g2_mc = mc_gammas(model)
        print(f"=== {name} (case I) ===")
        print(f"  analytic   p={st.p:.6e}  f={st.f:.6e}  gamma={st.gamma:.4f}  gamma2={st.gamma2:.4f}")
        print(f"  simulator  p={p_mc:.6e}  f={f_mc:.6e}  gamma={g_mc:.4f}  gamma2={g2_mc:.4f}")
        if name == "jd":
            print("  paper     p=2.006e-03  f=2.000e-05")
        print(
            f"  ratio p mc/analytic={p_mc / st.p:.3f}  gamma2 mc/analytic={g2_mc / st.gamma2:.3f}"
        )


if __name__ == "__main__":
    main()
