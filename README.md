# Classical versus learned option hedging

CityU **DSC4107** course project ([GitHub](https://github.com/hluaguo/dsc4107)).
We replicate Sepp (2013) and the Heston experiment of Buehler et al. (2019), then
compare discrete classical hedging to deep hedging on one shared simulator with
proportional transaction costs.

**Headline finding:** at the course benchmark (**Heston**, at-the-money one-month call,
**20 bp** proportional cost), a well-tuned **classical delta band** matches or beats
the learned hedger on risk-adjusted P&L — **classic hedging is all you need** at
moderate frictions. Deep hedging can trade more cheaply but often buys that with
worse tail risk; see [`REPORT.md`](REPORT.md) §3 and the RQ-A Pareto cell in
[`experiments/rq_a/`](experiments/rq_a/). Follow-up “liquid 10 bp” runs (Sepp **k**
under our full-rate ledger) are documented as a soft footnote in
[`experiments/README.md`](experiments/README.md), not as a new main claim.

Course deliverables (PDF report and slides) live in [`submission/`](submission/).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) (Python version pinned in `.python-version`).

```bash
uv sync
uv run pytest -q
```

Training scripts use CUDA when available (`--device auto`) and otherwise run on CPU.

## What to read

| Document | Purpose |
| --- | --- |
| [`REPORT.md`](REPORT.md) | Full write-up: both replications, 4×5 comparison, holdout, limitations |
| [`experiments/README.md`](experiments/README.md) | Index of research questions (RQ-A … RQ-K) with verdicts and result paths |
| [`docs/README.md`](docs/README.md) | Plan, exploration notes, reference papers, agent conventions |
| [`submission/report/report.pdf`](submission/report/report.pdf) | Short PDF report for grading |

## Repository layout

```
hedging/          Simulators, Sepp formulas, classical + deep hedgers, frozen benchmark
bench/            Pinned classical baseline (classical_best.json)
scripts/          Runnable entry points (replications, benchmark, RQ shims, holdout)
experiments/      One folder per RQ with NOTES.md + committed full-run results/
notebooks/        Two replication notebooks (generated from scripts/build_notebooks.py)
tests/            Analytic and accounting checks (frozen)
submission/       PDF report, slides, and mirrored result artifacts for graders
docs/             Research plan, exploration notes, reference PDFs
archive/          Historical autoresearch session (not needed to reproduce headline)
```

Frozen modules (`hedging/market.py`, `hedge.py`, `bench.py`, …) and the test suite
are intentionally stable; see [`docs/agents.md`](docs/agents.md) for contributor rules.

## Reproduce the headline benchmark

The frozen cell is `BenchConfig` in `hedging/bench.py`: Heston, 20 bp cost, 21 steps.
The classical champion is stored in `bench/classical_best.json` (delta band 0.1).

```bash
uv run python scripts/tune_classical.py       # re-freeze baseline (usually skip; JSON is pinned)
uv run python scripts/benchmark_loop.py       # one scored iteration → results/ (gitignored)
uv run python scripts/rq_a_pareto.py          # RQ-A cost–CVaR Pareto → experiments/rq_a/results/
uv run python scripts/holdout_eval.py         # locked holdout (run only when intentionally re-evaluating)
```

Replication and comparison scripts (Sepp tables, deep Heston run, 4×5 edge map) are
listed in [`REPORT.md`](REPORT.md) §6. Their default outputs go to `results/`; committed
copies for the written report are under `submission/data/`.

## Research questions (summary)

Original weekend cells are under [`experiments/`](experiments/). Each folder has a
`NOTES.md` with the question, acceptance rules, and the full-run table. The
[`experiments/README.md`](experiments/README.md) table links every RQ to its verdict.

## Tests and holdout

- `uv run pytest -q` — closed-form Sepp checks, rollout accounting, frictionless sanity.
- The **holdout** seed pool is only read inside `scripts/holdout_eval.py`; do not use
  those seeds elsewhere.

## References

PDFs: [`docs/papers/`](docs/papers/). Primary citations in `REPORT.md`.
