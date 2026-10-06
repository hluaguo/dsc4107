# Classical versus learned option hedging

This report covers three things the plan asks for: the replication of Sepp
(2013), the replication of the Heston experiment of Buehler, Gonon, Teichmann
and Wood (2019), and the head-to-head comparison that neither paper runs. All
numbers are produced by the scripts in `scripts/` and written under `results/`;
nothing here is hand-computed.

The economic problem is the same in both papers. A desk sells one at-the-money
one-month call, hedges it with the underlying, and pays a proportional cost on
every trade. Sepp finds the best strategy inside a small analytic family;
Buehler et al. replace the family with a network. The comparison asks what the
extra freedom buys once the classical dials are set well.

## 1. Replication of Sepp (2013)

Sepp writes the Sharpe ratio of a discrete delta hedge as

```
S(N) = (u - c sqrt(N)) / (sqrt(T) sqrt(p / N + f)),
```

where `u` is the mean hedged P&L per unit time, `c` the cost rate, `p` the
local-variance term and `f` the remaining variance floor. `n*` is the frequency
that maximises `S`, and an equivalent price band `alpha_s` and delta band
`alpha_d` can be read off from it. `hedging/sepp.py` implements the closed forms
for the four processes (log-normal DF, jump-diffusion JD, Heston SV, and SVJ)
for both of Sepp's parameter cases.

**Table 2 reproduction.** `scripts/sepp_replication.py` prints our coefficients
next to the paper's. The mean coefficient `u` matches closely; the largest
relative error across the eight rows is 3.9%.

| case | model | u ours | u paper | p ours | p paper | N* ours | N* paper | S ours | S paper |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I | gbm | 0.5693 | 0.5692 | 0.2519 | 0.2461 | 242 | 239 | 0.83 | 0.89 |
| I | jd | 0.5640 | 0.5510 | 0.2118 | 0.2006 | 147 | 138 | 0.65 | 0.66 |
| I | heston | 0.6621 | 0.6703 | 0.2583 | 0.2524 | 75 | 74 | 0.43 | 0.45 |
| I | svj | 0.6551 | 0.6550 | 0.2164 | 0.2085 | 70 | 68 | 0.43 | 0.45 |
| II | gbm | 1.8080 | 1.8558 | 0.6138 | 0.5998 | 130 | 129 | 1.18 | 1.30 |
| II | jd | 1.7826 | 1.7628 | 0.4765 | 0.4423 | 88 | 75 | 1.01 | 0.99 |
| II | heston | 2.0054 | 2.0876 | 0.6359 | 0.6214 | 49 | 49 | 0.65 | 0.73 |
| II | svj | 1.9738 | 1.9995 | 0.4903 | 0.4634 | 45 | 43 | 0.68 | 0.72 |

`u` matches to four decimals for GBM case I (0.5693 vs 0.5692) and SVJ case I
(0.6551 vs 0.6550).

Jump-diffusion needed a real fix rather than a caveat. The JD branch had
dropped the same `q = sqrt(pi^3)/4` midpoint normalisation that the GBM and SVJ
branches carry, so `p` came out 24% below the paper and the optimal frequency
correspondingly high. Equations (7.13)-(7.14) attach `q` to the whole
`2 sigma_r^4 + lambda sigma_r^2 nu^2` bracket of `p` and to the jump term of
`f`; restoring it moves `p` to +5.6% and `f` to -9% of Table 2 and, more
tellingly, matches the paper's JD Sharpe (0.65 vs 0.66 in case I, 1.01 vs 0.99
in case II). `scripts/diagnose_jd.py` isolates the fix: the second-moment gamma
`gamma2` was never the problem (it matches the simulator to 0.2%), only the
coefficient was.

**Sharpe-optimal frequencies and bands.** The optimum is interior: with no cost
(`c = 0`) more frequent hedging always lowers variance, but with cost the Sharpe
curve turns over. The optima move to lower frequency as the model gets more
complex, because a stochastic-volatility or jump process has a variance floor
`f` that extra trading cannot remove, so paying more cost buys less.

| case | model | N* | Sharpe | price band | delta band |
| --- | --- | --- | --- | --- | --- |
| I | gbm | 242 | 0.83 | 0.0096 | 0.0345 |
| I | jd | 147 | 0.65 | 0.0124 | 0.0442 |
| I | heston | 75 | 0.43 | 0.0174 | 0.0621 |
| I | svj | 70 | 0.43 | 0.0180 | 0.0642 |
| II | gbm | 130 | 1.18 | 0.0219 | 0.0446 |
| II | jd | 88 | 1.01 | 0.0267 | 0.0543 |
| II | heston | 49 | 0.65 | 0.0359 | 0.0730 |
| II | svj | 45 | 0.68 | 0.0373 | 0.0758 |

Figures: `results/figures/sepp_pnl_case*.png`, `sepp_sharpe_case*.png`,
`sepp_term_structure.png`.

**Monte Carlo cross-check.** `scripts/sepp_replication.py --mc --mc-paths 60000`
re-simulates the market at the analytic optimum so the closed forms and the
simulator can be compared directly. The mean P&L matches for every model and
the P&L volatility matches for GBM, Heston and SVJ. After the `q` fix the JD
volatility is within 3% at case I and 12% at case II, and the analytic and
simulated JD Sharpe ratios now agree (0.65 vs 0.65; 1.01 vs 0.95). The residual
case-II gap is the paper's own variance approximation, not a transcription
error.

| case | model | P&L analytic | P&L MC | vol analytic | vol MC | Sharpe analytic | Sharpe MC |
| --- | --- | --- | --- | --- | --- | --- | --- |
| I | gbm | 0.00318 | 0.00309 | 0.00385 | 0.00365 | 0.83 | 0.85 |
| I | jd | 0.00373 | 0.00381 | 0.00571 | 0.00590 | 0.65 | 0.65 |
| I | heston | 0.00511 | 0.00537 | 0.01191 | 0.01173 | 0.43 | 0.46 |
| I | svj | 0.00515 | 0.00542 | 0.01187 | 0.01168 | 0.43 | 0.46 |
| II | gbm | 0.01098 | 0.01102 | 0.00932 | 0.00916 | 1.18 | 1.20 |
| II | jd | 0.01220 | 0.01295 | 0.01213 | 0.01370 | 1.01 | 0.95 |
| II | heston | 0.01526 | 0.01695 | 0.02338 | 0.02369 | 0.65 | 0.72 |
| II | svj | 0.01539 | 0.01709 | 0.02280 | 0.02380 | 0.68 | 0.72 |

## 2. Replication of Buehler et al. (2019)

`hedging/deep.py` implements the deep hedger in PyTorch. The state at each step
is (log-moneyness, time, current position); the network outputs the position to
hold, and the loss is the mean of the worst `alpha` fraction of terminal P&L
(the CVaR objective the paper studies). The rollout charges the proportional
cost on every trade including the opening and closing trades, so the network
cannot hide a trade from the loss.

**The frictionless check.** With zero cost the learned policy should rediscover
the Black-Scholes delta; if it does not, no other number is trustworthy. Training
the default hedger with cost 0 on 20,000 paths gives

```
correlation = 0.9907   slope = 1.148   RMSE = 0.0581
mean learned = 0.499   mean BS delta = 0.508
```

The correlation near one says the policy has the right shape; the slope near one
and the nearly equal means say it is on the right scale. This is the strongest
single piece of evidence that the learned part of the project is implemented
correctly.

**The Heston experiment.** `scripts/deep_hedging_replication.py` trains on the
Heston market with a volatility risk premium (option sold at 25% implied against
a 20% long-run realised level, without which a Sharpe comparison collapses into
"do nothing") and scores on the eight validation seeds. The learned hedger is
compared against plain BS delta rebalanced every step (the paper's comparator)
and against the frozen delta-band baseline.

| cost (bp) | learned | BS delta | frozen band | edge vs delta | edge vs band |
| --- | --- | --- | --- | --- | --- |
| 0 | 1.154 | 1.139 | 0.987 | +0.016 | +0.167 |
| 20 | 0.373 | 0.254 | 0.352 | +0.119 | +0.020 |
| 50 | 0.105 | -0.970 | -0.526 | +1.075 | +0.631 |
| 100 | 0.182 | -2.276 | -1.573 | +2.458 | +1.755 |

At zero cost the network matches the delta hedge, as the frictionless check
predicts. Above zero cost it beats the every-step delta hedge decisively,
because trading every step is exactly the mistake that costs punish, and it also
beats the cost-aware frozen band. The result is qualitative, as the paper's is:
the direction and the ordering are the point, not the fourth decimal. Every
number averages three independent training seeds and the per-seed minimum is the
same sign as the mean (see `results/deep_hedging/results.json`).

## 3. The comparison

The comparison is the original contribution. For every cell of a 4 models by 5
cost levels grid, the classical baseline is re-tuned on its own seed pool
(`tune_seeds`) over time schedules and price/delta bands, and the network is
trained on `train` seeds and scored on the same `val` seeds. Re-tuning the
classical side per cell is what makes the comparison fair: a stale classical
setting would manufacture a win. The reported number is the mean per-seed Sharpe
edge, learned minus classical, averaged over the three training seeds.

### 3.1 One fixed network configuration everywhere

`scripts/comparison_study.py --fixed-hyper` uses the same explicit
hyperparameters in every cell (hidden 64, depth 3, lr 1e-3, alpha 0.65, 80
epochs), so no cell can win by getting a luckier search. The edge map:

| model | cost (bp) | edge | edge min | win | deep Sharpe | classical Sharpe | deep std | classical std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gbm | 0 | +0.030 | +0.014 | 1.00 | 1.258 | 1.228 | 0.46 | 0.47 |
| gbm | 20 | -0.001 | -0.008 | 0.50 | 0.347 | 0.348 | 0.55 | 0.58 |
| gbm | 50 | +0.038 | +0.032 | 1.00 | 0.073 | 0.035 | 1.72 | 1.74 |
| gbm | 100 | +0.422 | +0.416 | 1.00 | 0.161 | -0.261 | 3.44 | 1.74 |
| gbm | 200 | +1.009 | +1.004 | 1.00 | 0.158 | -0.851 | 3.48 | 1.74 |
| jd | 0 | -0.006 | -0.009 | 0.00 | 0.116 | 0.121 | 1.98 | 1.89 |
| jd | 20 | -0.090 | -0.096 | 0.00 | -0.081 | 0.008 | 1.94 | 2.47 |
| jd | 50 | -0.032 | -0.037 | 0.00 | -0.149 | -0.117 | 2.19 | 2.47 |
| jd | 100 | +0.369 | +0.361 | 1.00 | 0.044 | -0.325 | 4.21 | 2.47 |
| jd | 200 | +0.785 | +0.770 | 1.00 | 0.042 | -0.743 | 4.37 | 2.46 |
| heston | 0 | +0.020 | +0.007 | 1.00 | 1.159 | 1.139 | 0.51 | 0.52 |
| heston | 20 | +0.014 | +0.006 | 1.00 | 0.366 | 0.352 | 0.61 | 0.61 |
| heston | 50 | +0.054 | +0.047 | 1.00 | 0.100 | 0.046 | 1.60 | 1.74 |
| heston | 100 | +0.430 | +0.425 | 1.00 | 0.179 | -0.251 | 3.11 | 1.73 |
| heston | 200 | +1.020 | +1.015 | 1.00 | 0.174 | -0.846 | 3.17 | 1.73 |
| svj | 0 | -0.012 | -0.016 | 0.00 | 0.113 | 0.125 | 2.11 | 1.92 |
| svj | 20 | -0.076 | -0.082 | 0.00 | -0.062 | 0.014 | 2.05 | 2.45 |
| svj | 50 | +0.020 | +0.016 | 1.00 | -0.092 | -0.112 | 2.36 | 2.45 |
| svj | 100 | +0.372 | +0.368 | 1.00 | 0.050 | -0.322 | 3.97 | 2.45 |
| svj | 200 | +0.798 | +0.789 | 1.00 | 0.054 | -0.744 | 4.10 | 2.44 |

Three regimes appear.

1. **Cost near zero.** The two strategies tie, and the deep P&L standard
   deviation is within a few percent of the classical one. GBM and Heston give a
   small positive edge (+0.02 to +0.03, winning on every seed), JD and SVJ a
   small negative one. This is the regime the frictionless check covers, and it
   is a tie as it should be.

2. **Cost 20 bp.** The comparison is essentially flat: edges are between -0.09
   and +0.01. At the benchmark's own cost level the learned hedger does not beat
   a well-tuned classical band.

3. **Cost 50 bp and above.** The learned edge becomes large and consistent,
   but the P&L standard deviations tell the important part of the story: the
   deep strategy's volatility rises from ~0.5 at low cost to 3-4 at high cost,
   while its mean P&L climbs toward the option premium. The network has learned
   to stop hedging and collect the volatility risk premium. The classical family
   cannot follow, because every classical candidate opens a full delta at
   `t = 0` (`hedge.BandDelta` returns the target on the first step, and
   `TimeBasedDelta` rebalances at `j = 0`), so it pays the opening cost and then
   carries the short-option exposure. A high-cost Sharpe "win" that comes from
   not hedging is not evidence of better hedging, and it is recorded here as a
   change of risk profile rather than a superior hedge.

The guard rail that makes this visible is the pair of standard-deviation
columns; the sign flip of `cvar_edge_mean` (negative, i.e. deeper tail, whenever
the deep Sharpe edge is large) points the same way. The edge map is in
`results/comparison/edge_map_fixed.png`.

### 3.2 Optuna hyperparameter search

`scripts/comparison_study.py` also runs the plan's Optuna add-on. Trials are
ranked on a fourth seed pool (`SEARCH_SEEDS`) that is disjoint from train,
validation and holdout, so the search cannot select on the seeds that produce
the headline number. Searches were run for Heston and jump-diffusion at 20, 50
and 100 bp.

| model | cost (bp) | search edge | reported edge | win | deep Sharpe | classical Sharpe | deep std | classical std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heston | 20 | +0.011 | +0.021 | 1.00 | 0.374 | 0.352 | 0.60 | 0.61 |
| heston | 50 | +0.140 | +0.139 | 1.00 | 0.185 | 0.046 | 3.09 | 1.74 |
| heston | 100 | +0.436 | +0.435 | 1.00 | 0.185 | -0.251 | 3.18 | 1.73 |
| jd | 20 | -0.075 | -0.077 | 0.00 | -0.069 | 0.008 | 1.96 | 2.47 |
| jd | 50 | +0.158 | +0.156 | 1.00 | 0.039 | -0.117 | 3.98 | 2.47 |
| jd | 100 | +0.379 | +0.377 | 1.00 | 0.052 | -0.325 | 4.33 | 2.47 |

The search edge and the reported validation edge agree closely, which is the
check that the search pool is not leaking into the report. The searched
configuration does not change the qualitative picture: Heston at 20 bp stays a
tie, jump-diffusion at 20 bp stays a loss, and at high cost the search again
prefers low-frequency, high-variance configurations whose std jumps to 3-4. The
search cannot repair the jump-diffusion gap at 20 bp.

### 3.3 What the comparison says

- With no frictions the network recovers the classical hedge and the two tie.
- At the benchmark's 20 bp the learned hedger does not beat the best classical
  band, and on the jump models it is slightly behind.
- Above roughly 50 bp the learned hedger's Sharpe advantage is real but is
  mostly the volatility risk premium being harvested by a strategy that has
  learned to stop trading. That is a legitimate strategy, but it is a different
  product, not a better hedge.
- The single honest gap that remains at moderate cost is that the classical
  family here has no "trade less than delta" member. The network's freedom to
  choose a partial or zero hedge is exactly the extra freedom the comparison was
  meant to measure; it shows up as a premium-harvesting edge rather than as a
  lower-variance hedge.

### 3.4 The locked holdout

`scripts/holdout_eval.py` was run once, after every choice was frozen, on the
five holdout seeds that no experiment above ever touched. It uses the frozen
benchmark (Heston, 20 bp, delta-band baseline) and the default training budget
(200 epochs, hidden 128, depth 3, three seeds). The result is the only
validation-free estimate of the edge in the project:

| metric | value |
| --- | --- |
| mean edge | +0.0339 |
| edge std | 0.0031 |
| edge min | +0.0293 |
| win rate | 1.00 (5 seeds) |

The holdout edge is the same sign and the same order of magnitude as the
validation edge for the Heston 20 bp cell (+0.014 fixed-hyper, +0.021 searched,
+0.020 for the frozen-band comparator in Section 2). If the loop had been
fitting its own noise the holdout edge would be near zero or negative; it is
not, so the small benchmark win is real. It remains small by design: this is the
regime where the two strategies nearly tie, which is itself the headline result.

## 4. Correctness and overfitting guards

- **Frictionless delta recovery** (Section 2) has correlation 0.9907.
- **Simulator versus closed form** is covered by `tests/test_sepp.py`, which pins
  the analytic formulas to Monte Carlo (`uv run pytest -q`: 23 passed).
- **Disjoint seed pools.** Train, validation, tuning, search and holdout are
  separate; the Optuna search uses `SEARCH_SEEDS` and never the validation seeds.
- **Multiple training seeds.** Every reported comparison number averages seeds
  0, 1 and 2.
- **Per-seed spread.** `edge_min` and `win_rate` are reported next to the mean.
  Where the deep edge is large, it is positive on every validation seed; where
  the strategies tie, individual seeds disagree, as they should.
- **Premium-harvesting check.** The deep and classical P&L standard deviations
  are printed side by side so a Sharpe gain from under-hedging is visible.

## 5. Open issues and limitations

- **Jump-diffusion variance. Resolved.** The JD `p`/`f` were missing the
  `q = sqrt(pi^3)/4` midpoint factor of equations (7.13)-(7.14), which the GBM
  and SVJ branches already carried. Restoring it moves `p` from -24% to +5.6%
  of Table 2 and matches the JD Sharpe. The only residual is the paper's own
  variance approximation at case II, where the analytic volatility stays about
  12% below the simulator; `scripts/diagnose_jd.py` separates the two.
- **Classical family.** The classical candidates always open a full initial
  hedge, so they cannot express "under-hedge" or "hold flat". The high-cost deep
  wins are partly an artefact of that asymmetry. Adding an initial-trade flag or
  a partial-delta member would make the high-cost comparison interpretable; it
  was left out because `hedge.py` is frozen for this project.
- **Compute scale.** This is a course-scale testbed. The network runs use 80-100
  epochs, 2048 training paths per seed, three seeds, and a small MLP; the
  Optuna search uses 8 trials. The direction of the results is stable across the
  fixed and searched configurations, but the exact edges should not be read as
  paper-grade magnitudes.
- **Notebooks.** `notebooks/01_sepp_replication.ipynb` and
  `notebooks/02_deep_hedging_heston.ipynb` are generated from
  `scripts/build_notebooks.py`. The executed source of truth is the scripts in
  `scripts/`; the notebooks need Jupyter installed to show inline outputs.

## 6. How to reproduce

```bash
uv run pytest -q                                        # 23 checks
uv run python scripts/sepp_replication.py               # Sepp tables + figures
uv run python scripts/sepp_replication.py --mc --mc-paths 60000
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3
uv run python scripts/comparison_study.py --fixed-hyper \
    --costs 0.0 0.002 0.005 0.01 0.02
uv run python scripts/comparison_study.py --models heston jd \
    --costs 0.002 0.005 0.01 --trials 8
```

All scripts pick CUDA automatically. Outputs land in `results/sepp/`,
`results/deep_hedging/` and `results/comparison/`.

## 7. File map

| Path | Role |
| --- | --- |
| `hedging/sepp.py` | Sepp closed forms, optimal frequency, band conversion |
| `hedging/market.py` | GBM, JD, Heston and SVJ simulators |
| `hedging/hedge.py` | The one P&L rollout and the classical strategies |
| `hedging/deep.py` | Learned hedger: features, network, CVaR loss, training |
| `hedging/bench.py` | Frozen benchmark config, seed pools, scoring |
| `tests/test_sepp.py` | Closed form against Monte Carlo and Table 2 |
| `scripts/sepp_replication.py` | Section 5 tables and figures |
| `scripts/deep_hedging_replication.py` | Frictionless check and Heston experiment |
| `scripts/comparison_study.py` | The head-to-head map and the Optuna add-on |
| `scripts/holdout_eval.py` | The single locked holdout run |
