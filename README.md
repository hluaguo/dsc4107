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
| `scripts/tune_classical.py` | Freezes the classical baseline once |
| `scripts/benchmark_loop.py` | One benchmark iteration, machine-readable output |
| `scripts/holdout_eval.py` | The final run on the locked holdout pool |
| `tests/` | Analytic and accounting checks |

## The benchmark

One contract, one market, one cost level, all fixed in `hedging/bench.BenchConfig`:
an at-the-money one-month call, a Heston market, and a 20 bp proportional cost.
Both strategies receive the same premium and are scored on the same paths, so
the difference between them is the difference between their hedges.

The classical baseline is chosen once by `scripts/tune_classical.py` and written
to `bench/classical_best.json`. At this cost level the winner is a time rule
that opens the hedge at t=0 and holds to maturity, which is a genuinely strong
bar rather than a straw one.

## Running

```bash
uv run pytest -q                                   # checks
uv run python scripts/tune_classical.py            # freeze the baseline (once)
uv run python scripts/benchmark_loop.py            # one benchmark iteration
uv run python scripts/holdout_eval.py              # final, locked evaluation
```

`scripts/benchmark_loop.py` prints `METRIC name=value` lines and appends the
per-seed rows and a summary to `results/`.

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
