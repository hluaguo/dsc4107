# Autoresearch: learned hedging edge over the classical baseline

## Objective

Improve the learned hedger in `hedging/deep.py` and its training setup in
`hedging/hyper.py` so it beats the frozen classical delta-hedging baseline on
out-of-sample paths. The benchmark trains the network on one seed pool and
scores it on a disjoint validation pool under a Heston market with proportional
transaction costs.

The benchmark carries a volatility risk premium: the option is sold at 25%
implied volatility while the Heston long-run volatility is 20%. Without that
premium, doing nothing dominates hedging and a Sharpe-based comparison collapses
into "trade as little as possible", which the network discovers and which makes
the metric meaningless. The premium keeps variance reduction worth paying for.

The learned hedger currently starts behind. The frozen classical baseline is a
delta band, tuned on its own seed pool, and the default CVaR hedger does not yet
match it. Closing that gap is the task.

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
- Watch the P&L standard deviation behind `val_deep_sharpe_mean`. A strategy
  that raises Sharpe by refusing to hedge is exploiting the premium, not
  hedging, and does not count as a result.

## How to run

`./archive/autoresearch/measure.sh` prints `METRIC name=value` lines.

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
- pool sizes, cost level, contract, premium and seeds are fixed

`archive/autoresearch/measure.sh` may be extended with extra instrumentation, but the metrics
it reports and the benchmark it calls must not change.

## Constraints

- `./archive/autoresearch/checks.sh` must pass.
- Every reported number averages over the training seeds in
  `Hyper.torch_seeds`. Do not shrink that list to chase a lucky run.
- An improvement must hold across the validation seeds, not only on their mean.
  If `win_rate` falls, the edge is not real.
- When performance ties, prefer the simpler change.

## What's been tried

- The first benchmark had implied volatility equal to realised (both 0.20). It
  was discarded after diagnosis: the Sharpe-maximising strategy was to sell and
  refuse to hedge, the mean objective found that strategy (P&L standard
  deviation 3.2 against the classical 1.7), and the metric rewarded it. The
  premium was introduced to remove this failure mode.
- Under the premium, with a 200 epoch, 128 wide, 3 deep network:
  - CVaR (`alpha=0.05`, the default): Sharpe +0.283, P&L standard deviation
    0.63, a genuine hedge that clears the old classical bar.
  - mean (`alpha=1.0`): Sharpe +0.188, standard deviation 3.2, still not
    hedging and still cashing the premium.
  So the tail objective is the one that hedges, and it is the correct default.
- The frozen classical baseline is now a delta band (`delta_band=0.1`), Sharpe
  about 0.353 on its tuning pool. Beating it is a real target rather than a
  formality.
