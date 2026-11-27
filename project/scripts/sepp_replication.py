"""Reproduce Sepp (2013), Section 5: the Sharpe-optimal hedging frequency.

Runs the closed forms in ``hedging.sepp`` for both of the paper's parameter
cases and writes the tables and figures the paper reports. Monte Carlo is
optional (``--mc``) and slow; the default run is analytic only, because the
analytic tables are what the replication is compared against and the simulator
agreement is already locked down in ``tests/test_sepp.py``.

Outputs land in ``results/sepp/`` (tables as JSON and markdown) and
``results/figures/`` (PNG). Nothing here is used by the deep-hedging benchmark;
it exists so the classical side of the project is a checked replication rather
than an assertion that the baseline is good.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from hedging import bs, hedge, market, sepp  # noqa: E402

OUT_TABLES = ROOT / "results" / "sepp"
OUT_FIGURES = ROOT / "results" / "figures"

# The paper's Table 2, transcribed so the script prints the comparison next to
# its own output rather than the reader having to hold two documents open.
TABLE2 = {
    ("I", market.GBM): (0.5692, 0.0151, 0.2461, 0.0004, 239, 0.89),
    ("I", market.JUMP): (0.5510, 0.0139, 0.2006, 0.0020, 138, 0.66),
    ("I", market.HESTON): (0.6703, 0.0153, 0.2524, 0.0107, 74, 0.45),
    ("I", market.SVJ): (0.6550, 0.0142, 0.2085, 0.0110, 68, 0.45),
    ("II", market.GBM): (1.8558, 0.0571, 0.5998, 0.0040, 129, 1.30),
    ("II", market.JUMP): (1.7628, 0.0506, 0.4423, 0.0119, 75, 0.99),
    ("II", market.HESTON): (2.0876, 0.0581, 0.6214, 0.0403, 49, 0.73),
    ("II", market.SVJ): (1.9995, 0.0518, 0.4634, 0.0418, 43, 0.72),
}

MODELS = (market.GBM, market.JUMP, market.HESTON, market.SVJ)


def stats_grid(case: str, model: str):
    """Analytic ``u, c, p, f`` and the Sharpe-optimal frequency for one row."""
    sm = sepp.table1_case(case, model)
    st = sepp.sepp_stats(sm)
    n_round, n_star, _, how = sepp.optimal_frequency(st)
    return sm, st, n_round, n_star, how


def table2_rows() -> list[dict]:
    rows = []
    for case in ("I", "II"):
        for model in MODELS:
            _, st, n_round, n_star, how = stats_grid(case, model)
            pu, pc, pp, pf, pn, ps = TABLE2[(case, model)]
            rows.append(
                {
                    "case": case,
                    "model": model,
                    "u_pct": st.u_raw * 100,
                    "u_paper_pct": pu,
                    "c_pct": st.c * 100,
                    "c_paper_pct": pc,
                    "p_pct": st.p * 100,
                    "p_paper_pct": pp,
                    "f_pct": st.f * 100,
                    "f_paper_pct": pf,
                    "N_round": n_round,
                    "N_paper": pn,
                    "sharpe": st.sharpe(n_round),
                    "sharpe_paper": ps,
                    "solver": how,
                }
            )
    return rows


def print_table2(rows: list[dict]) -> None:
    header = f"{'case':>4} {'model':>7} | {'u ours':>7} {'u paper':>7} | {'c ours':>7} {'c paper':>7} | {'p ours':>7} {'p paper':>7} | {'f ours':>7} {'f paper':>7} | {'N*':>4} {'Np':>4} | {'S':>5} {'Sp':>5}"
    print("Table 2 reproduction (all figures in percent)")
    print(header)
    for r in rows:
        print(
            f"{r['case']:>4} {r['model']:>7} | {r['u_pct']:>7.4f} {r['u_paper_pct']:>7.4f} | "
            f"{r['c_pct']:>7.4f} {r['c_paper_pct']:>7.4f} | {r['p_pct']:>7.4f} {r['p_paper_pct']:>7.4f} | "
            f"{r['f_pct']:>7.5f} {r['f_paper_pct']:>7.5f} | {r['N_round']:>4} {r['N_paper']:>4} | "
            f"{r['sharpe']:>5.2f} {r['sharpe_paper']:>5.2f}"
        )


def optimal_table(case: str) -> list[dict]:
    """Time-based optimal frequency and the equivalent price/delta bands."""
    rows = []
    for model in MODELS:
        sm, st, n_round, n_star, _ = stats_grid(case, model)
        alpha_s, alpha_d = sepp.band_from_frequency(n_star, sm)
        rows.append(
            {
                "case": case,
                "model": model,
                "N_star": n_star,
                "N_round": n_round,
                "sharpe": st.sharpe(n_round),
                "expected_pnl": st.mean_pnl(n_round),
                "costs": st.transaction_cost(n_round),
                "volatility": st.volatility(n_round),
                "price_band": alpha_s,
                "delta_band": alpha_d,
            }
        )
    return rows


def figure_pnl_and_sharpe(case: str) -> pathlib.Path:
    """Figure 1 shape: expected P&L and volatility against Sharpe on N."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grid = np.logspace(0.7, 3.4, 200)
    for model in MODELS:
        sm = sepp.table1_case(case, model)
        st = sepp.sepp_stats(sm)
        ax.plot(grid, st.mean_pnl(grid) * 100, label=f"{model} P&L")
    ax.set_xscale("log")
    ax.set_xlabel("hedging frequency N")
    ax.set_ylabel("expected P&L (percent of notional)")
    ax.set_title(f"Sepp case {case}: expected delta-hedging P&L")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    path = OUT_FIGURES / f"sepp_pnl_case{case}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_sharpe(case: str) -> pathlib.Path:
    """Figure 2 shape: Sharpe against N for the four processes."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grid = np.logspace(0.7, 3.6, 400)
    for model in MODELS:
        sm = sepp.table1_case(case, model)
        st = sepp.sepp_stats(sm)
        ax.plot(grid, st.sharpe(grid), label=model)
        n_round, n_star, _, _ = sepp.optimal_frequency(st)
        ax.plot([n_star], [st.sharpe(n_star)], "o", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("hedging frequency N")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(f"Sepp case {case}: Sharpe ratio against hedging frequency")
    ax.legend()
    fig.tight_layout()
    path = OUT_FIGURES / f"sepp_sharpe_case{case}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def figure_term_structure() -> pathlib.Path:
    """Figure 3 shape: the optimal Sharpe ratio as a function of maturity."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    maturities = np.linspace(1.0 / 12.0, 2.0, 60)
    for model in MODELS:
        base = sepp.table1_case("II", model)
        sharpes = []
        for T in maturities:
            sm = sepp.SeppModel(**{**base.__dict__, "T": float(T)})
            st = sepp.sepp_stats(sm)
            _, n_star, _, _ = sepp.optimal_frequency(st)
            sharpes.append(st.sharpe(n_star))
        ax.plot(maturities, sharpes, label=model)
    ax.set_xlabel("maturity T (years)")
    ax.set_ylabel("optimal Sharpe ratio")
    ax.set_title("Sepp case II: term structure of the optimal Sharpe ratio")
    ax.legend()
    fig.tight_layout()
    path = OUT_FIGURES / "sepp_term_structure.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def mc_check(case: str, n_paths: int, seed: int) -> list[dict]:
    """Optional simulator cross-check at the analytic optimal frequency.

    Costs are charged at ``k/2`` per trade (Sepp's equation 2.5 treats ``k`` as
    the full bid-ask spread), and the rollout pays to open and close, so its
    mean should sit a little below the analytic ``u - c sqrt(N)``.
    """
    rows = []
    for model in MODELS:
        sm, st, n_round, n_star, _ = stats_grid(case, model)
        N = max(4, int(n_round))
        paths = market.simulate_paths(
            sm.market_params(), sm.S, sm.T, N, n_paths, seed=seed
        )
        premium = float(bs.bs_price(sm.S, sm.K, sm.T, 0.0, sm.sigma_i, "call"))
        fn = hedge.TimeBasedDelta(
            sm.K, sm.T, 0.0, sm.sigma_i, "call", sm.T / N, frequency=1
        )
        result = hedge.rollout(
            paths, fn, sm.K, "call", premium, sm.k / 2.0, sm.T / N
        )
        mc_sharpe = float(result.pnl.mean() / result.pnl.std())
        rows.append(
            {
                "case": case,
                "model": model,
                "N": N,
                "analytic_pnl": float(st.mean_pnl(N)),
                "mc_pnl": float(result.pnl.mean()),
                "analytic_vol": float(st.volatility(N)),
                "mc_vol": float(result.pnl.std()),
                "analytic_sharpe": float(st.sharpe(N)),
                "mc_sharpe": mc_sharpe,
            }
        )
    return rows


def print_mc(rows: list[dict]) -> None:
    print("\nMonte Carlo cross-check at the analytic optimum")
    print(f"{'model':>7} {'N':>5} | {'P&L a':>8} {'P&L mc':>8} | {'vol a':>8} {'vol mc':>8} | {'S a':>5} {'S mc':>5}")
    for r in rows:
        print(
            f"{r['model']:>7} {r['N']:>5} | {r['analytic_pnl']:>8.5f} {r['mc_pnl']:>8.5f} | "
            f"{r['analytic_vol']:>8.5f} {r['mc_vol']:>8.5f} | {r['analytic_sharpe']:>5.2f} {r['mc_sharpe']:>5.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mc", action="store_true", help="also run the slow MC check")
    parser.add_argument("--mc-paths", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    OUT_TABLES.mkdir(parents=True, exist_ok=True)

    rows = table2_rows()
    print_table2(rows)
    pl.DataFrame(rows).write_json(OUT_TABLES / "table2.json")

    optima = []
    for case in ("I", "II"):
        optima.extend(optimal_table(case))
    pl.DataFrame(optima).write_json(OUT_TABLES / "optima.json")
    print("\nOptimal time-based frequency and equivalent bands")
    for r in optima:
        print(
            f"  case {r['case']} {r['model']:>7}: N*={r['N_star']:7.1f}  S={r['sharpe']:.3f}  "
            f"price band={r['price_band']:.4f}  delta band={r['delta_band']:.4f}"
        )

    figures = []
    for case in ("I", "II"):
        figures.append(figure_pnl_and_sharpe(case))
        figures.append(figure_sharpe(case))
    figures.append(figure_term_structure())
    print("\nfigures:")
    for path in figures:
        print(f"  {path.relative_to(ROOT)}")

    if args.mc:
        rows_mc = []
        for case in ("I", "II"):
            rows_mc.extend(mc_check(case, args.mc_paths, args.seed))
        print_mc(rows_mc)
        pl.DataFrame(rows_mc).write_json(OUT_TABLES / "monte_carlo_check.json")

    summary = {
        "u_relative_error": {
            f"{r['case']}-{r['model']}": abs(r["u_pct"] - r["u_paper_pct"])
            / r["u_paper_pct"]
            for r in rows
        },
        "n_optimal_relative_error": {
            f"{r['case']}-{r['model']}": abs(r["N_round"] - r["N_paper"])
            / r["N_paper"]
            for r in rows
        },
    }
    (OUT_TABLES / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "\nmax relative error in u: "
        f"{max(summary['u_relative_error'].values()):.1%}"
    )


if __name__ == "__main__":
    main()
