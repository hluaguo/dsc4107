# Broader exploration

> Companion to `REPORT.md` and `SIDEQUEST.md`. These probes reuse the frozen
> modules read-only, train with `train`/`val`/`search` seeds only, and write
> under `results/exploration/`. The holdout pool is never read and no earlier
> result is changed. The only source change is an additive, default-preserving
> `features` argument in `hedging/deep.py` for the ablation.

Five axes were explored: the objective as the hedging/harvesting dial, what the
observables contribute, cross-model transfer, estimation noise in the headline
edge, and whether a larger search changes any conclusion.

## 1. The objective is the hedging-versus-harvesting dial

`scripts/explore_objective.py` sweeps the CVaR tail fraction `alpha` on four
cells. The pattern is clean and confirms the report's mechanism from the inside
rather than inferring it.

Heston at 20 bp:

| alpha | Sharpe | mean | std | turnover |
| --- | --- | --- | --- | --- |
| 0.05 | +0.290 | 0.178 | 0.615 | 207 |
| 0.35 | +0.320 | 0.175 | 0.548 | 209 |
| 0.50 | +0.343 | 0.194 | 0.564 | 199 |
| **0.65** | **+0.362** | 0.229 | 0.633 | 182 |
| 0.80 | +0.259 | 0.392 | 1.510 | 101 |
| 1.00 | +0.187 | 0.591 | 3.160 | 2.1 |

Heston at 100 bp:

| alpha | Sharpe | mean | std | turnover |
| --- | --- | --- | --- | --- |
| 0.05 | −1.541 | −1.170 | 0.760 | 176 |
| 0.35 | −0.578 | −0.630 | 1.089 | 122 |
| 0.50 | −0.096 | −0.167 | 1.744 | 76 |
| 0.65 | +0.182 | 0.570 | 3.140 | 2.3 |
| 1.00 | +0.182 | 0.578 | 3.171 | 1.1 |

Two new facts fall out that the fixed-alpha comparison could not see:

- **The tail objective is the hedge.** As `alpha -> 0` the network trades
  heavily (turnover ~200) and its P&L standard deviation sits near the
  classical one — it is hedging. As `alpha -> 1` turnover collapses to ~2 and
  the standard deviation triples: it stops hedging and collects the premium.
  `alpha` is not a refinement dial; it is the dial that selects which strategy
  family the network lands in.
- **The best `alpha` is cost-dependent.** At 20 bp the Sharpe peaks at
  `alpha = 0.65`; at 100 bp every hedging objective loses money and only the
  near-mean objective is positive. The autoresearch default of 0.65 is optimal
  *for the 20 bp benchmark* and is the wrong objective once costs rise. A
  single fixed objective is a hidden assumption in the earlier comparison.

The JD and SVJ panels show the same switch (JD 20 bp: `alpha = 1` gives +0.052
at std 4.35; JD 100 bp: only `alpha >= 0.65` is positive). Plot:
`results/exploration/objective_sweep.png`.

## 2. What the observables contribute

`scripts/explore_features.py` retrains with one input removed (Heston).

| features | 20 bp Sharpe | 20 bp turnover | 100 bp Sharpe | 100 bp turnover |
| --- | --- | --- | --- | --- |
| full (moneyness, time, pos) | +0.362 | 182 | +0.182 | 2.3 |
| no_pos | +0.306 | 199 | +0.182 | 1.0 |
| no_time | +0.348 | 174 | +0.181 | 2.9 |
| no_moneyness | +0.246 | 73 | +0.182 | 1.2 |

- **The current position is what buys the band.** Removing it costs +0.056
  Sharpe and *raises* turnover from 182 to 199: without knowledge of where it
  already is, the network must recompute and re-trade a target every step
  instead of holding inside a band. This is the first direct evidence for the
  report's claim that the position feedback is the band mechanism.
- **Moneyness matters most** (+0.116 when removed): the policy is tracking the
  delta, as it should.
- Changing `alpha` dominates changing features. At 100 bp the feature set is
  irrelevant because the network has already stopped trading.

Plot: `results/exploration/feature_ablation.png`.

## 3. The learned policy transfers across market models

`scripts/explore_transfer.py` trains on each model and scores the *same* nets on
all four (Heston example row values, 20 bp):

| train \ test | gbm | jd | heston | svj |
| --- | --- | --- | --- | --- |
| gbm | +0.347 | −0.081 | +0.346 | −0.076 |
| jd | +0.349 | −0.081 | +0.347 | −0.077 |
| heston | +0.345 | −0.072 | +0.366 | −0.067 |
| svj | +0.330 | −0.067 | +0.364 | −0.062 |
| **BS delta** | +0.247 | −0.121 | +0.254 | −0.116 |

The diagonal is barely above the off-diagonal: training on GBM loses only about
0.02 Sharpe on Heston and vice versa. The row order is nearly constant and the
column order dominates, which means **the ranking across markets is a property
of the market, not of overfitting to the training dynamics.** The network never
observes the volatility state or the model, so this is the expected result, but
it is worth stating: the learned hedge is close to a universal function of
(moneyness, time, position). It also beats plain BS delta in every column by
about +0.04 to +0.11, including the jump markets where hedging is hardest.

Plot and full matrix: `results/exploration/transfer_matrix.png`.

## 4. How much of the headline edge is noise?

`scripts/explore_robustness.py` puts confidence intervals on the frozen
benchmark cell (Heston, 20 bp, default 200-epoch budget), which the plan asks
for but the comparison did not report.

| source | mean edge | 95% CI |
| --- | --- | --- |
| training-seed bootstrap (8 edges) | +0.0319 | [+0.0288, +0.0350] |
| paired path bootstrap (320k paths) | +0.0319 | [+0.0290, +0.0347] |

Both intervals exclude zero, and they agree, so the benchmark win is not a
single-seed or single-path fluke. The path-count sweep is stable:

| val paths | edge | seed spread |
| --- | --- | --- |
| 5,000 | +0.0364 | 0.0098 |
| 20,000 | +0.0328 | 0.0023 |
| 40,000 | +0.0319 | 0.0048 |
| 80,000 | +0.0340 | 0.0033 |

The 40,000-path choice used everywhere is not biasing the estimate; the value
bounces inside the seed spread. This is the quantitative backing for the
report's claim that the moderate-cost edge is real but small.

## 5. A larger search finds the harvesting strategy

`scripts/explore_optimise.py` reruns the Optuna add-on with 30 trials instead of
8 on the two weak cells.

| model | cost | 8 trials (plan) | 30 trials | best alpha | deep std |
| --- | --- | --- | --- | --- | --- |
| heston | 20 bp | +0.021 | **+0.027** | 0.61 | 0.60 |
| jd | 20 bp | −0.077 | **+0.044** | 0.90 | **4.28** |

The larger search changes two conclusions, in opposite directions:

- **Heston 20 bp** improves only from +0.021 to +0.027 with the same hedging
  configuration (`alpha = 0.61`, std 0.60). More search does not manufacture a
  large edge; the cell really is near a tie.
- **JD 20 bp** flips from a clear loss to a positive edge, but the winning
  configuration is `alpha = 0.90` with P&L standard deviation **4.28** against
  the classical family's ~2.47. The search did not find a better jump hedge; it
  found the premium-harvesting regime that the standard-deviation guard rail
  was built to catch. Left unconstrained, the optimiser will take the
  harvesting route whenever it scores better.

This is the sharpest lesson of the exploration: the objective (or the absence
of a standard-deviation guard) decides which strategy family the search lands
in, so any claim that a learned hedger "wins" must be read together with its
P&L spread.

## Files

| Path | Role |
| --- | --- |
| `scripts/explore_objective.py` | alpha sweep, hedging versus harvesting |
| `scripts/explore_features.py` | input ablation |
| `scripts/explore_transfer.py` | cross-model transfer matrix |
| `scripts/explore_robustness.py` | bootstrap CIs and path-count stability |
| `scripts/explore_optimise.py` | 30-trial search on the two weak cells |
| `results/exploration/*.json` | the raw tables |
| `results/exploration/*.png` | the plots |
