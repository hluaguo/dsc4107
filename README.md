# Classical versus learned option hedging

Working material for a replication-and-comparison study. The economics are
simple to state: sell one call option, hedge it with the underlying, and pay a
proportional cost on every trade. Sepp (2013) optimises that hedge inside a
small hand-built strategy family; Buehler et al. (2019) replaces the family with
a neural network. The repository reproduces enough of both to put them on one
testbed and ask which gives the better risk-adjusted profit and loss.

## Where things live

| Path | Role |
| --- | --- |
| `hedging/bs.py` | Black-Scholes price and greeks, implied volatility |
| `hedging/market.py` | Four simulators: GBM, jump-diffusion, Heston, Heston plus jumps |
| `hedging/hedge.py` | The shared rollout engine and the classical strategies |
| `hedging/deep.py` | The learned hedger: features, network, objective, training |
| `hedging/hyper.py` | Training hyperparameters, the dials the search may turn |
| `hedging/bench.py` | The frozen benchmark: config, seed pools, scoring |
| `hedging/metrics.py` | Sharpe, CVaR and the other reported statistics |
| `hedging/sepp.py` | Sepp (2013) closed forms, optimal frequency, band conversion |
| `scripts/tune_classical.py` | Freezes the classical baseline once |
| `scripts/benchmark_loop.py` | One benchmark iteration, machine-readable output |
| `scripts/sepp_replication.py` | Sepp Section 5 tables and figures (optional Monte Carlo check) |
| `scripts/deep_hedging_replication.py` | Frictionless delta check and the Heston deep-hedging experiment |
| `scripts/comparison_study.py` | The head-to-head map and the Optuna hyperparameter add-on |
| `scripts/build_notebooks.py` | Regenerates the two replication notebooks |
| `scripts/holdout_eval.py` | The final run on the locked holdout pool |
| `notebooks/` | The two replication notebooks (built from `build_notebooks.py`) |
| `tests/` | Analytic and accounting checks |
| `REPORT.md` | The written report: both replications and the comparison |
| `AGENTS.md` | Repository conventions: why-style docs, `uv add`, frozen files |

## The benchmark

One contract, one market, one cost level, all fixed in `hedging/bench.BenchConfig`:
an at-the-money one-month call, a Heston market, and a 20 bp proportional cost.
Both strategies receive the same premium and are scored on the same paths, so
the difference between them is the difference between their hedges.

The classical baseline is chosen once by `scripts/tune_classical.py` and written
to `bench/classical_best.json`. At this cost level the winner is a delta band
(`delta_band = 0.1`): rebalance only when the Black-Scholes delta has drifted
past the band, which trades less than a fixed calendar schedule and is a
genuinely strong bar rather than a straw one.

## Running

```bash
uv run pytest -q                                   # checks
uv run python scripts/tune_classical.py            # freeze the baseline (once)
uv run python scripts/benchmark_loop.py            # one benchmark iteration
uv run python scripts/holdout_eval.py              # final, locked evaluation
```

`scripts/benchmark_loop.py` prints `METRIC name=value` lines and appends the
per-seed rows and a summary to `results/`.

## Replications and the comparison

```bash
uv run python scripts/sepp_replication.py            # Sepp tables + figures
uv run python scripts/sepp_replication.py --mc --mc-paths 60000
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3
uv run python scripts/comparison_study.py --fixed-hyper \
    --costs 0.0 0.002 0.005 0.01 0.02
uv run python scripts/comparison_study.py --models heston jd \
    --costs 0.002 0.005 0.01 --trials 8
```

All training scripts select CUDA when it is present (`--device auto` by
default) and fall back to the CPU otherwise. The Optuna search requires the
`optuna` dependency, added with `uv add optuna`. `scripts/comparison_study.py`
ranks trials on a seed pool (`SEARCH_SEEDS`) disjoint from training, validation
and holdout, and it prints the deep and classical P&L standard deviations next
to the Sharpe edge so a Sharpe gain from under-hedging is visible. The full
findings are in `REPORT.md`.

## Guarding against overfitting

The obvious way to fool this setup is to tune against the paths you score on.
Four things make that hard:

1. Three disjoint seed pools. The network trains on `train` seeds, the running
   experiments score on `val` seeds, and a `holdout` pool is never read until
   the end.
2. Averaging over training seeds. Every number averages several independent
   initialisations, so a lucky run cannot pass as an improvement.
3. The generation gap. The benchmark reports the in-sample edge next to the
   validation edge; a widening gap means the hedger memorised its training
   paths.
4. Per-seed reporting. The mean edge is published with its spread, its minimum
   and its win rate, so a mean that hides disagreement between seeds is visible.

## Autoresearch

The autonomous optimisation session lives in `.auto/`. `prompt.md` states the
objective, the metric, the files in scope and the guard rails. `measure.sh`
runs one iteration, `checks.sh` runs the tests, and `ideas.md` holds the
backlog. The session optimises the learned hedger only; the benchmark and the
baseline stay frozen.
