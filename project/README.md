# Classic hedging is all you need

CityU **DSC4107** — [hluaguo/dsc4107](https://github.com/hluaguo/dsc4107)

We replicate two papers on the same simulated desk (sell one ATM call, hedge with
the underlying, pay proportional transaction costs), then ask whether a neural
hedger beats a strong classical policy on a **fixed** benchmark cell.

**Title (locked):** **Classic hedging is all you need** at the course cell (Heston,
one-month ATM call, **20 bp** full-rate cost). Details and numbers: [`submission/report/report.pdf`](submission/report/report.pdf).

## Quick start

```bash
uv sync
uv run pytest -q
jupyter lab notebooks/   # or open the four notebooks in your IDE
```

Python version is pinned in `.python-version`. Scripts use GPU when available.

## Notebooks (start here)

| # | Notebook | Role |
| --- | --- | --- |
| 01 | [`notebooks/01_sepp_replication.ipynb`](notebooks/01_sepp_replication.ipynb) | Sepp Section 5 replication |
| 02 | [`notebooks/02_deep_hedging.ipynb`](notebooks/02_deep_hedging.ipynb) | Buehler Heston + frictionless check (artifacts) |
| 03 | [`notebooks/03_classical_vs_deep.ipynb`](notebooks/03_classical_vs_deep.ipynb) | Frozen bench, comparison map, holdout display |
| 04 | [`notebooks/04_research_path.ipynb`](notebooks/04_research_path.ipynb) | RQ-A + liquid footnote chain (soft) |

Index: [`notebooks/README.md`](notebooks/README.md). Regenerate from sources:
`uv run python scripts/build_notebooks.py`.

## Story spine (English)

1. **Papers.** Sepp (2013) — analytic discrete-hedging Sharpe, optimal frequency and
   bands. Buehler et al. (2019) — deep hedging with a CVaR-style objective under costs.
2. **Replications.** Sepp Section 5 tables/figures (`scripts/sepp_replication.py`);
   Buehler Heston experiment plus frictionless delta check
   (`scripts/deep_hedging_replication.py`). Reference PDFs: [`docs/papers/`](docs/papers/).
3. **Frozen cell.** `BenchConfig` in `hedging/bench.py`: Heston, ATM 1M call,
   `cost = 0.002` (20 bp **full-rate** ledger), 21 steps. Classical champion pinned in
   `bench/classical_best.json` (delta band 0.1).
4. **Classical vs deep on that cell.** Historical 4×5 comparison and holdout in
   the [report](submission/report/report.pdf). Decision rule: deep wins only if it beats the tuned
   band on **both** mean cost and CVaR95. At 20 bp the learned hedger has a small
   Sharpe edge on Heston (+0.034 on the locked holdout) but a 24% worse CVaR95.
   RQ-A (`experiments/rq_a/`) shows the default deep policy is **cheaper on mean cost
   but worse on CVaR95**, verdict `dominated`. Tail-matched nets (RQ-D, RQ-F) reach
   the band's CVaR95 only at higher cost, so none clears the band on both axes.
   Low-α nets do reach tighter tails than any classical rule tried (see the report).
5. **Follow-up RQs (same cell, then liquid footnote).** Mandate, risk-matched α, train-time
   CVaR cap, cost stress, policy anatomy — indexed in
   [`experiments/README.md`](experiments/README.md). These sharpen *why* the headline
   holds; they do not replace it.
6. **Liquid 10 bp (footnote only).** Sepp Case I uses **k/2** in eq. (2.5); our simulator
   charges **full-rate** `cost`. Runs at `cost = 0.001` match Sepp’s **k** numerically but
   not Case I economics. A brittle soft path (do **not** promote to main claim):
   **H** `clears` (narrow) → **K** `dominated` → **L** `a05_only` → **I** `fails` → **J**
   `band_like`. See liquid rows in [`experiments/README.md`](experiments/README.md).

Committed full-run JSON and figures: `experiments/*/results/` and
[`submission/data/`](submission/data/). Replication commands: see below.

## Where to read

| Path | Role |
| --- | --- |
| [`notebooks/`](notebooks/) | **Public notebook path** — four notebooks (see [`notebooks/README.md`](notebooks/README.md)) |
| [`submission/report/report.pdf`](submission/report/report.pdf) | Full report (replications, comparison, holdout) |
| [`experiments/README.md`](experiments/README.md) | RQ-A … RQ-L index and verdicts |
| [`submission/`](submission/) | PDF report, slides, mirrored artifacts |
| [`docs/papers/`](docs/papers/) | Sepp & Buehler PDFs |
| [`archive/scripts/`](archive/scripts/) | Historical comparison + exploration scripts |

Optional background notes (plan, extra probes) live under [`docs/`](docs/) and are not
required for grading.

## Layout

```
hedging/       simulators, Sepp formulas, classical + deep hedgers, bench config
bench/         pinned classical baseline (classical_best.json)
notebooks/     course story (four notebooks)
scripts/       holdout, benchmark loop, replication + RQ entry shims
experiments/   one folder per RQ + committed full-run results/
archive/       off-path scripts and old autoresearch notes
tests/         checks (frozen)
submission/    course PDFs and data mirror
```

Holdout seeds are used **only** in `scripts/holdout_eval.py`.

## Reproduce the headline cell

```bash
uv run python scripts/rq_a_pareto.py    # → experiments/rq_a/results/
uv run python scripts/benchmark_loop.py # → results/ (gitignored; see submission/data/)
```

Sepp / Buehler / 4×5 comparison:

```bash
uv run python scripts/sepp_replication.py               # Sepp tables + figures
uv run python scripts/sepp_replication.py --mc --mc-paths 60000
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3
uv run python archive/scripts/comparison_study.py --fixed-hyper \
    --costs 0.0 0.002 0.005 0.01 0.02
uv run python archive/scripts/comparison_study.py --models heston jd \
    --costs 0.002 0.005 0.01 --trials 8
```

All scripts pick CUDA automatically. Outputs land in `results/` (gitignored); the
committed copies the report cites live under `submission/data/`.
