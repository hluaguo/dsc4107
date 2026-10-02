# Autoresearch: learned hedging edge over the classical baseline

## Objective

Improve the learned hedger in `hedging/deep.py` and its training setup in
`hedging/hyper.py` so it beats the frozen classical delta-hedging baseline on
out-of-sample paths. The benchmark trains the network on one seed pool and
scores it on a disjoint validation pool under a Heston market with proportional
transaction costs.

The starting point is not competitive. The default objective is CVaR, which
gives a strong tail and a poor Sharpe ratio, while the classical baseline is
Sharpe-tuned. Closing that gap is the task. The mean-versus-tail tradeoff is the
main axis to explore.

## Metrics

- Primary: `edge_mean` (higher is better). Mean over validation seeds of
  (deep Sharpe minus classical Sharpe).
- Secondary: `edge_std`, `edge_min`, `edge_median`, `edge_tstat`, `win_rate`,
  `cvar_edge_mean`, `val_deep_sharpe_mean`, `val_classical_sharpe_mean`,
  `train_edge`, `generation_gap`, `robust_edge_gbm`, `robust_edge_jd`.

Read the guard rails before claiming a win:

- `generation_gap` near zero is healthy. A gap that grows positive means the
  hedger is memorising the training paths.
- `edge_min` and `win_rate` say whether the edge holds on every validation seed
  or rests on one lucky seed.
- `cvar_edge_mean` is the second perspective. A change that lifts Sharpe by
  widening the tail is a tradeoff to record, not automatically a win.

## How to run

`./.auto/measure.sh` prints `METRIC name=value` lines.

## Files in scope

- `hedging/deep.py`: network, features, objective, training loop.
- `hedging/hyper.py`: training hyperparameters.
- New modules under `hedging/` that `deep.py` imports.
- `README.md`, only when a change alters the documented method.

## Off limits

- `hedging/market.py`, `hedging/bs.py`, `hedging/hedge.py`, `hedging/metrics.py`
- `hedging/bench.py`, the benchmark definition and its seed pools
- `bench/classical_best.json`, the frozen baseline
- `tests/` and `scripts/`: do not weaken a check to make a number pass
- the holdout seed pool: never simulate it, never read it
- pool sizes, cost level, contract and seeds are fixed

`.auto/measure.sh` may be extended with extra instrumentation, but the metrics
it reports and the benchmark it calls must not change.

## Constraints

- `./.auto/checks.sh` must pass.
- Every reported number averages over the training seeds in
  `Hyper.torch_seeds`. Do not shrink that list to chase a lucky run.
- An improvement must hold across the validation seeds, not only on their mean.
  If `win_rate` falls, the edge is not real.
- When performance ties, prefer the simpler change.

## What's been tried

- Objective sweep at 120 epochs with 3 training seeds:
  - `alpha=1.0` (mean objective): `edge_mean` +0.113, deep Sharpe +0.005
  - `alpha=0.5`: `edge_mean` -0.393
  - `alpha=0.1`: `edge_mean` -0.407
  - `alpha=0.05` (default CVaR): `edge_mean` -0.405

  The mean objective already clears the baseline, but it widens the tail
  (`cvar_edge_mean` about -4.1). The CVaR objectives protect the tail
  (`cvar_edge_mean` about +2.9) and surrender the Sharpe. A useful result sits
  between the two ends.

- The frozen baseline is `time` with `frequency=21`, that is, open the hedge at
  t=0 and hold to maturity. At a 20 bp cost this is the Sharpe-optimal member of
  the family the classical tune searches, so beating it is not free.
