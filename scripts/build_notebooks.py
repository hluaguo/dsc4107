"""Build the four grader-facing notebooks from readable cell sources.

Hand-writing ``.ipynb`` JSON is brittle; this file is the reviewable source.
Run ``uv run python scripts/build_notebooks.py`` after editing cells. Notebooks
default to loading committed artifacts under ``submission/data/`` and
``experiments/*/results/`` so demos do not re-run multi-hour training.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def _notebook(cells: list[tuple[str, str]]) -> dict:
    return {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for kind, source in cells
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


SEPP_CELLS = [
    (
        "markdown",
        """# Replication of Sepp (2013), *When You Hedge Discretely*

Sepp derives closed forms for the mean, variance and expected cost of a
discrete delta-hedged short option position, then maximises the Sharpe ratio
over the hedging frequency and the equivalent price/delta bands. This notebook
reproduces Section 5: the four processes (DF, JD, SV, SVJ), the two parameter
cases, and the Sharpe-optimal frequency.

The implementation lives in `hedging/sepp.py` and is checked against the
simulator in `tests/test_sepp.py`. Where the paper's printed constants could
not be made to agree with a direct Monte Carlo estimate, the notebook says so
rather than smoothing it over.

Regenerate the full run (tables, figures, optional Monte Carlo):
`uv run python scripts/sepp_replication.py`. The Monte Carlo cell below uses
`hedge.rollout(..., sm.k / 2.0, ...)` because Sepp eq. (2.5) charges **half**
the quoted spread on |Δδ|·S; the frozen `BenchConfig` ledger uses **full-rate**
`cost` instead (see root README liquid footnote).
""",
    ),
    (
        "code",
        """import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))

from hedging import bs, market, sepp

MODELS = (market.GBM, market.JUMP, market.HESTON, market.SVJ)
print("project root:", ROOT)""",
    ),
    (
        "markdown",
        """## 1. The cash gamma

Sepp's equation (2.4) defines the option cash gamma `0.5 S^2 U_SS`. At zero
rates it must equal the textbook Black-Scholes gamma times `0.5 S^2`; if it did
not, the P&L expansion and the delta hedge would be built from different
objects.
""",
    ),
    (
        "code",
        """S = np.linspace(80, 120, 200)
tau = 0.5
mine = sepp.cash_gamma(tau, S, 100.0, 0.165)
theirs = 0.5 * S**2 * bs.bs_gamma(S, 100.0, tau, 0.0, 0.165)

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(S, mine, label="Sepp (2.4)")
ax.plot(S, theirs, "--", label="0.5 S^2 BS gamma")
ax.set_xlabel("spot")
ax.set_ylabel("cash gamma")
ax.legend()
plt.show()

print("max abs difference:", np.abs(mine - theirs).max())""",
    ),
    (
        "markdown",
        """## 2. Table 2 reproduction

`u, c, p, f` are the coefficients of the Sharpe ratio
`S(N) = (u - c sqrt(N)) / (sqrt(T) sqrt(p/N + f))`. The table below prints our
value next to the paper's for every model and both parameter cases. The `u`
column uses the paper's convention, which does **not** subtract the cost of the
opening trade; the simulator comparison later does.
""",
    ),
    (
        "code",
        """import json

TABLE2 = {
    ("I", "gbm"): (0.5692, 0.0151, 0.2461, 0.0004, 239, 0.89),
    ("I", "jd"): (0.5510, 0.0139, 0.2006, 0.0020, 138, 0.66),
    ("I", "heston"): (0.6703, 0.0153, 0.2524, 0.0107, 74, 0.45),
    ("I", "svj"): (0.6550, 0.0142, 0.2085, 0.0110, 68, 0.45),
    ("II", "gbm"): (1.8558, 0.0571, 0.5998, 0.0040, 129, 1.30),
    ("II", "jd"): (1.7628, 0.0506, 0.4423, 0.0119, 75, 0.99),
    ("II", "heston"): (2.0876, 0.0581, 0.6214, 0.0403, 49, 0.73),
    ("II", "svj"): (1.9995, 0.0518, 0.4634, 0.0418, 43, 0.72),
}

rows = []
for case in ("I", "II"):
    for model in MODELS:
        st = sepp.sepp_stats(sepp.table1_case(case, model))
        n_round, n_star, _, _ = sepp.optimal_frequency(st)
        pu, pc, pp, pf, pn, ps = TABLE2[(case, model)]
        rows.append({
            "case": case, "model": model,
            "u": st.u_raw * 100, "u_paper": pu,
            "c": st.c * 100, "c_paper": pc,
            "p": st.p * 100, "p_paper": pp,
            "f": st.f * 100, "f_paper": pf,
            "N": n_round, "N_paper": pn,
            "S": st.sharpe(n_round), "S_paper": ps,
        })
table = pl.DataFrame(rows)
table.select(["case", "model", "u", "u_paper", "c", "c_paper", "p", "p_paper", "f", "f_paper", "N", "N_paper", "S", "S_paper"])""",
    ),
    (
        "code",
        """# The `u` column is the paper's own coefficient (before the opening-trade
# adjustment) and reproduces Table 2 to about 4% worst case; the optimal
# frequency is within about 13% and the Sharpe ordering is preserved.
table.select(["case", "model", "u", "u_paper", "N", "N_paper", "S", "S_paper"])""",
    ),
    (
        "markdown",
        """## 3. Sharpe ratio against hedging frequency

The shape is the paper's Figure 2: each process has a unique interior optimum,
and the models with unhedgeable risk (jumps, stochastic volatility) peak at a
lower frequency and a lower Sharpe ratio. Jump and SV risk raise the
irreducible variance `f`, which is the part of the risk no frequency removes.
""",
    ),
    (
        "code",
        """grid = np.logspace(0.7, 3.6, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for ax, case in zip(axes, ("I", "II")):
    for model in MODELS:
        st = sepp.sepp_stats(sepp.table1_case(case, model))
        ax.plot(grid, st.sharpe(grid), label=model)
        _, n_star, _, _ = sepp.optimal_frequency(st)
        ax.plot([n_star], [st.sharpe(n_star)], "o", ms=4)
    ax.set_xscale("log")
    ax.set_xlabel("hedging frequency N")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title(f"case {case}")
    ax.legend()
fig.tight_layout()
plt.show()""",
    ),
    (
        "markdown",
        """## 4. Optimal frequency and equivalent bands

Sepp converts the optimal frequency into a price band `sqrt(T) sigma_r /
sqrt(N)` and a delta band through the option's `S U_SS`. These are the dials
the classical baseline in the main benchmark actually trades, so this table is
the bridge between the replication and the comparison.
""",
    ),
    (
        "code",
        """optima = []
for case in ("I", "II"):
    for model in MODELS:
        sm = sepp.table1_case(case, model)
        st = sepp.sepp_stats(sm)
        _, n_star, _, _ = sepp.optimal_frequency(st)
        alpha_s, alpha_d = sepp.band_from_frequency(n_star, sm)
        optima.append({
            "case": case, "model": model, "N_star": round(n_star, 1),
            "sharpe": st.sharpe(n_star),
            "price_band": alpha_s, "delta_band": alpha_d,
        })
pl.DataFrame(optima)""",
    ),
    (
        "markdown",
        """## 5. Simulator cross-check

The paper's formulas are only useful if they describe the process the
simulator actually generates. This cell rolls a plain delta hedge over
simulated Heston paths and compares the realised mean, volatility and Sharpe
with the closed forms at the analytic optimum.
""",
    ),
    (
        "code",
        """from hedging import hedge

case = "I"
N_MC, N_PATHS = 60, 40_000
premium = None
rows = []
for model in MODELS:
    sm = sepp.table1_case(case, model)
    st = sepp.sepp_stats(sm)
    paths = market.simulate_paths(sm.market_params(), sm.S, sm.T, N_MC, N_PATHS, seed=23)
    premium = float(bs.bs_price(sm.S, sm.K, sm.T, 0.0, sm.sigma_i, "call"))
    fn = hedge.TimeBasedDelta(sm.K, sm.T, 0.0, sm.sigma_i, "call", sm.T / N_MC, frequency=1)
    # (2.5) treats k as the full spread, so each trade pays k/2.
    result = hedge.rollout(paths, fn, sm.K, "call", premium, sm.k / 2.0, sm.T / N_MC)
    rows.append({
        "model": model, "N": N_MC,
        "P&L_analytic": st.mean_pnl(N_MC), "P&L_mc": result.pnl.mean(),
        "vol_analytic": st.volatility(N_MC), "vol_mc": result.pnl.std(),
        "S_analytic": st.sharpe(N_MC), "S_mc": result.pnl.mean() / result.pnl.std(),
    })
pl.DataFrame(rows)""",
    ),
    (
        "markdown",
        """### What the MC check shows

- The mean P&L matches to a few percent for all four processes.
- The P&L volatility matches for DF, SV and SVJ.
- The jump-diffusion case initially disagreed because the JD branch had dropped
  the `q = sqrt(pi^3)/4` midpoint factor that equations (7.13)-(7.14) attach to
  the `p` bracket and the jump part of `f`, and that the GBM and SVJ branches
  already carry. With the factor restored the JD `p` sits within 6% of Table 2,
  the analytic and simulated JD Sharpe ratios agree, and the residual case-II
  volatility gap (about 12%) is the paper's own variance approximation rather
  than a transcription error.

The notebook therefore supports the paper's optimisation result. The one
remaining approximation is Sepp's variance formula for jumps at the longer
maturity of case II, and it is left visible rather than tuned away.
""",
    ),
]


DEEP_CELLS = [
    (
        "markdown",
        """# Replication of Buehler et al. (2019), *Deep Hedging* (Heston case)

Buehler, Gonon, Teichmann and Wood replace the analytic hedging rule with a
network trained to minimise a convex risk measure of the terminal P&L. The
committed numbers live in `submission/data/deep_hedging/results.json`, produced
by `scripts/deep_hedging_replication.py` (same entry point as `REPORT.md` §6).

Default below: **load artifacts** (fast). Set `RUN_TRAINING = True` only for a
small smoke re-run; full reproduction uses the script with `--epochs 100`.
""",
    ),
    (
        "code",
        """import json
import pathlib
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))

from hedging import bench as B

RUN_TRAINING = False
DATA = ROOT / "submission" / "data" / "deep_hedging" / "results.json"
FIG = ROOT / "submission" / "data" / "deep_hedging" / "sharpe_vs_cost.png"

BASE = B.BenchConfig()
print("frozen bench (for context):", BASE.model, "cost=", BASE.cost)
print("artifact:", DATA.relative_to(ROOT))""",
    ),
    (
        "markdown",
        """## 1. Frictionless delta check (committed run)

With zero costs the network should track Black-Scholes delta. `REPORT.md` §4
records correlation **0.9907** on the full script run; we display the committed
JSON rather than re-train here.
""",
    ),
    (
        "code",
        """payload = json.loads(DATA.read_text())
fr = payload["frictionless"]
pl.DataFrame([{
    "correlation": fr["correlation"],
    "slope": fr["slope"],
    "rmse": fr["rmse"],
}])""",
    ),
    (
        "markdown",
        """## 2. Heston with proportional costs (committed run)

Sharpe ratios vs cost for learned hedger, every-step BS delta (Buehler baseline),
and the frozen band (`load_classical_best()`). Numbers match `REPORT.md` §2 table.
""",
    ),
    (
        "code",
        """rows = []
for block in payload["costs"]:
    s = block["summary"]
    rows.append({
        "cost_bp": block["cost"] * 1e4,
        "learned_sharpe": s["deep_sharpe"]["mean"],
        "bs_delta_sharpe": s["bs_delta_sharpe"]["mean"],
        "band_sharpe": s["band_sharpe"]["mean"],
        "edge_vs_band": s["edge_vs_band"]["mean"],
    })
table = pl.DataFrame(rows)
table""",
    ),
    (
        "code",
        """if FIG.is_file():
    img = plt.imread(FIG)
    plt.figure(figsize=(7, 4))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Committed figure (deep_hedging_replication.py)")
    plt.show()
else:
    fig, ax = plt.subplots(figsize=(7, 4))
    for col, label in (
        ("learned_sharpe", "learned"),
        ("bs_delta_sharpe", "BS delta"),
        ("band_sharpe", "frozen band"),
    ):
        ax.plot(table["cost_bp"], table[col], marker="o", label=label)
    ax.set_xlabel("transaction cost (bp)")
    ax.set_ylabel("Sharpe ratio")
    ax.legend()
    plt.show()""",
    ),
    (
        "markdown",
        """### Optional smoke re-run (off by default)

Mirrors the frictionless block in `scripts/deep_hedging_replication.py` with a
tiny budget. Do not cite smoke numbers as report figures.
""",
    ),
    (
        "code",
        """if not RUN_TRAINING:
    print("Set RUN_TRAINING = True to execute the smoke training cell.")
else:
    import torch
    from dataclasses import replace

    from hedging import bs, deep
    from hedging.hyper import Hyper

    cfg0 = replace(BASE, cost=0.0, n_val_paths=2_000)
    hyper = Hyper(epochs=20, hidden=64, depth=3, alpha=0.05, torch_seeds=(0,))
    paths = cfg0.simulate(cfg0.model, 2001, 2_000)
    tensor = torch.as_tensor(paths, dtype=torch.float32)
    nets, _ = B.train_ensemble(cfg0, hyper)
    learned = deep.hedge_positions(tensor, nets[0], cfg0.K, cfg0.dt, cfg0.T)
    n_steps = paths.shape[1] - 1
    target = np.stack([
        bs.bs_delta(paths[:, j], cfg0.K, max(cfg0.T - j * cfg0.dt, 0.0), 0.0, cfg0.sigma_impl, "call")
        for j in range(n_steps)
    ], axis=1)
    x, y = learned.reshape(-1), target.reshape(-1)
    print("smoke correlation:", round(float(np.corrcoef(x, y)[0, 1]), 4))""",
    ),
]


CLASSICAL_CELLS = [
    (
        "markdown",
        """# Classical versus learned hedging (display from committed artifacts)

Headline cell: frozen `BenchConfig` (Heston ATM 1M, **20 bp full-rate** cost) and
`bench/classical_best.json`. Historical 4×5 comparison and the locked holdout
are already in `REPORT.md` §3; this notebook **loads** the JSON mirror under
`submission/data/` — it does not re-score paths or invent Sharpe edges.

Re-run writers (optional, slow): `archive/scripts/comparison_study.py`,
`scripts/holdout_eval.py` (holdout only, once).
""",
    ),
    (
        "code",
        """import json
import pathlib
import sys

import matplotlib.pyplot as plt
import polars as pl

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))

from hedging import bench as B

cfg = B.BenchConfig()
baseline = B.load_classical_best()
print("BenchConfig:", cfg.model, "cost=", cfg.cost, "n_steps=", cfg.n_steps)
print("classical_best.json:", baseline)""",
    ),
    (
        "markdown",
        """## Fixed-hyper edge map (`comparison_study.py --fixed-hyper`)

Same table as `REPORT.md` §3.1 (model × cost in bp). Edge = mean validation
Sharpe (learned − classical), averaged over training seeds.
""",
    ),
    (
        "code",
        """EDGE_JSON = ROOT / "submission" / "data" / "comparison" / "edge_map_fixed.json"
EDGE_PNG = ROOT / "submission" / "data" / "comparison" / "edge_map_fixed.png"
cells = json.loads(EDGE_JSON.read_text())
edge = pl.DataFrame(cells).with_columns((pl.col("cost") * 1e4).alias("cost_bp"))
edge.select(
    "model",
    "cost_bp",
    "edge_mean",
    "edge_min",
    "win_rate",
    "deep_sharpe",
    "classical_sharpe",
    "deep_std",
    "classical_std",
).sort("model", "cost_bp")""",
    ),
    (
        "code",
        """if EDGE_PNG.is_file():
    plt.figure(figsize=(8, 5))
    plt.imshow(plt.imread(EDGE_PNG))
    plt.axis("off")
    plt.title("Committed edge map (see REPORT §3.1)")
    plt.show()""",
    ),
    (
        "markdown",
        """## Locked holdout (`scripts/holdout_eval.py`)

Single run on the five holdout seeds; summary matches `REPORT.md` §3.4.
""",
    ),
    (
        "code",
        """HOLDOUT_LOG = ROOT / "submission" / "data" / "holdout_history.jsonl"
last = HOLDOUT_LOG.read_text().strip().splitlines()[-1]
holdout = json.loads(last)
pl.DataFrame([holdout]).select("edge_mean", "edge_std", "edge_min", "win_rate", "n_seeds", "timestamp")""",
    ),
    (
        "markdown",
        """### Reading (from the report, not re-derived here)

At **20 bp** on the frozen Heston cell the fixed-hyper comparison is flat
(`heston` edge about +0.014 in the map above). The holdout mean edge (+0.0339)
is the same order of magnitude — a small win, not a overturning of the classical
band story. High-cost Sharpe edges in the map are premium-harvesting artefacts;
see `REPORT.md` §3.1 regime 3.
""",
    ),
]


RESEARCH_CELLS = [
    (
        "markdown",
        """# Research path: RQ-A headline + liquid 10 bp footnote

**Main line (20 bp frozen cell):** RQ-A cost–CVaR Pareto (`scripts/rq_a_pareto.py`
→ `experiments/rq_a/results/`). Verdict **`dominated`**: deep policies are cheaper
on mean path cost but sit **above** the classical cost–tail hull on CVaR95.

**Footnote only (10 bp liquid):** Sepp Case I uses **k/2** in eq. (2.5); our
`hedge.rollout` ledger charges **full-rate** `cost`. Runs at `cost = 0.001` align
with Sepp liquid-index **k** numerically, not Case I economics. The brittle chain
**H → K → L → I → J** is indexed in `experiments/README.md` — do **not** promote
it to the course title (see root README).

This notebook loads committed `experiments/*/results/*.json` only.
""",
    ),
    (
        "code",
        """import json
import pathlib
import sys

import matplotlib.pyplot as plt
import polars as pl

ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
sys.path.insert(0, str(ROOT))

from hedging import bench as B

print("Headline cell:", B.BenchConfig().model, "cost=", B.BenchConfig().cost)""",
    ),
    (
        "markdown",
        """## RQ-A (headline auxiliary)

Full-run JSON: `experiments/rq_a/results/pareto.json`. Pareto plane uses
**mean_cost** and **cvar95** from `hedge.rollout` / `hedging/metrics.py` (lower
is better on both axes).
""",
    ),
    (
        "code",
        """RQ_A = json.loads((ROOT / "experiments" / "rq_a" / "results" / "pareto.json").read_text())
print("verdict:", RQ_A["verdict"])
classical = pl.DataFrame(RQ_A["classical"])
deep = pl.DataFrame(RQ_A["deep"])
cb = classical.filter(pl.col("label") == "classical_best").row(0, named=True)
print("classical_best mean_cost", cb["mean_cost"], "cvar95", cb["cvar95"])
deep.select("label", "mean_cost", "cvar95")""",
    ),
    (
        "code",
        """pareto_png = ROOT / "experiments" / "rq_a" / "results" / "pareto.png"
if pareto_png.is_file():
    plt.figure(figsize=(6, 4))
    plt.imshow(plt.imread(pareto_png))
    plt.axis("off")
    plt.title("RQ-A committed Pareto figure")
    plt.show()""",
    ),
    (
        "markdown",
        """## Liquid 10 bp footnote chain (soft)

Re-swept classical hull at `cost = 0.001`; never `classical_best.json` as baseline.
Verdict strings only — details in each folder's `NOTES.md` and `REPORT.md` spine.
""",
    ),
    (
        "code",
        """FOOTNOTE = [
    ("RQ-H", "rq_liquid_10bp", "liquid_10bp.json"),
    ("RQ-K", "rq_liquid_baseline", "liquid_baseline.json"),
    ("RQ-L", "rq_liquid_alpha", "liquid_alpha.json"),
    ("RQ-I", "rq_liquid_robust", "liquid_robust.json"),
    ("RQ-J", "rq_liquid_anatomy", "liquid_anatomy.json"),
]
rows = []
for tag, folder, fname in FOOTNOTE:
    path = ROOT / "experiments" / folder / "results" / fname
    payload = json.loads(path.read_text())
    rows.append({"step": tag, "folder": folder, "verdict": payload["verdict"]})
pl.DataFrame(rows)""",
    ),
    (
        "markdown",
        """### How to read this notebook

- **RQ-A** supports the locked title: classical cost–tail rules remain the
  reference on the 20 bp cell even when deep hedging trims mean transaction cost.
- The **liquid chain** is a sensitivity footnote; several steps fail or narrow
  quickly under robustness checks (`RQ-I`, `RQ-K`). Cite `experiments/README.md`
  and the JSON audit fields, not notebook-derived Sharpe numbers.
""",
    ),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    legacy = OUT / "02_deep_hedging_heston.ipynb"
    if legacy.exists():
        legacy.unlink()
    for name, cells in (
        ("01_sepp_replication.ipynb", SEPP_CELLS),
        ("02_deep_hedging.ipynb", DEEP_CELLS),
        ("03_classical_vs_deep.ipynb", CLASSICAL_CELLS),
        ("04_research_path.ipynb", RESEARCH_CELLS),
    ):
        path = OUT / name
        path.write_text(json.dumps(_notebook(cells), indent=1) + "\n")
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
