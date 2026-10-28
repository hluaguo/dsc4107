# Classic hedging is all you need

This report documents the Sepp (2013) replication, the Buehler et al. (2019)
Heston experiment, and a head-to-head comparison neither paper runs. All numbers
come from `scripts/` into `results/`; nothing here is hand-computed.

The headline is §3. At the course's nominal **20 bp** friction, a classical
band re-tuned per cell does not lose to the learned hedger in any hedging sense
that matters: Sharpe edges sit between about −0.09 and +0.01, holdout confirms
only a tiny positive gap, and the loss distribution shows the network buying mean
with a fatter left tail (§3.5). High-cost "wins" are premium harvesting, not
better hedges (§3.1 regime 3). **Classic hedging is enough** at the benchmark
friction; deep is not a uniform upgrade. Optional §8 records a separate liquid
10 bp footnote — recipe-specific, mandate-dependent, and brittle — not the lead
story.

The economic setup is the usual short ATM one-month call hedged in the underlying
with proportional cost on every trade. Sepp optimises inside an analytic family;
Buehler et al. replace the family with a network. Sections 1–2 check both sides
are implemented correctly; §3 measures what the extra freedom buys once classical
dials are set well — and at 20 bp the answer is essentially a tie, with caveats
below.

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

### 3.5 The loss distribution

The plan asks for the loss distribution next to the summary statistics, because
Sharpe alone cannot show where the risk sits. `scripts/explore_distribution.py`
plots the pooled terminal P&L of the learned hedger and the band on the frozen
benchmark (figure `results/exploration/distribution.png`).

| strategy | mean | std | skew | CVaR 95% | 5% quantile |
| --- | --- | --- | --- | --- | --- |
| learned | 0.240 | 0.626 | **-0.699** | **1.321** | -0.880 |
| band | 0.216 | 0.613 | -0.010 | 1.066 | -0.752 |

The two distributions have nearly the same spread, but the learned one is far
more negatively skewed. The +0.032 Sharpe edge comes with a CVaR that is 24%
worse: the network earns its extra mean by accepting a fatter left tail, which
is exactly the trade the seller's own risk measure is meant to expose. So even
at 20 bp, the one regime where the bootstrap says the edge is real, the edge is
not a uniform improvement. Whether it counts as a win depends on whether the
desk scores Sharpe or tail risk, and the two disagree.

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

## 8. Footnote: liquid 10 bp full-rate

**Read after §3.** The frozen **20 bp** benchmark (`bench/classical_best.json`,
holdout in §3.4) is the report's economic backdrop: classical bands already suffice
there. What follows is a **Lead-locked** side study at **Sepp liquid-index k** under
**our** full-rate ledger — whether any deep policy **clears** the **re-swept** 10 bp
val-Sharpe champion on shared OOS rollout (`mean_cost` strictly below champion
**and** `cvar95` ≤ champion). It is **not** a broad liquid deep win: unconstrained
`Hyper()` is **dominated** (RQ-K); the only clears need **risk-matched α** or
**Lagrange** tail budgets (RQ-H), cost margins are **~0.4%**, and expanded torch
seeds break most of them (RQ-I). Mandate engineering, not a default learned hedger
beating classical bands.

**Friction convention (must not be blurred).** Sepp (2013) eq (2.5) charges proportional
cost with factor **k/2** on |Δδ|·S; Case I sets **k = 0.001**. Our `hedge.rollout` and
`BenchConfig` charge the **full** proportional rate (Buehler et al. 2019 eq 5.7). Every
liquid cell sets `cost = 0.001` — numerically Sepp’s **k**, but **not** exact Case I
economics (~5 bp one-way under Sepp’s half-spread). The 10 bp classical **hull is
re-swept** on the RQ-A grid at that cost; **`classical_best.json` is never the comparison
baseline** (audit `loaded_classical_best_json: false` in each JSON). Do not read the 20 bp
frozen band row as the 10 bp champion.

**Shared 10 bp champion (all RQ-H–L).** Highest mean validation Sharpe on the re-swept
grid: **`band_delta_0.05`** with OOS `mean_cost` **0.2127** and `cvar95` **0.7367**
(source: `experiments/rq_liquid_10bp/results/liquid_10bp.json` and matching NOTES).

| RQ | Folder | Locked verdict | What it adds |
| --- | --- | --- | --- |
| **H** | `experiments/rq_liquid_10bp/` | **`clears`** | Risk-matched α sweep **or** Lagrange excess-CVaR at training α=0.65 can each produce **one** torch seed that beats the champion on cost **and** tail: `rm_seed_1` at selected α=**0.05** (cost **0.2118**, CVaR **0.7059**); `lag_seed_1` (cost **0.2119**, CVaR **0.7210**). Margins on cost are **~0.4%** — real but not economic dominance. |
| **K** | `experiments/rq_liquid_baseline/` | **`dominated`** | Default `Hyper()` (α=0.65, no RM sweep, no Lagrange): **0 / 6** torch seeds clear; every deep row is weakly dominated by some hull point. Expanded train paths **1001–1006** (vs default bench train elsewhere) do not rescue the headline. |
| **L** | `experiments/rq_liquid_alpha/` | **`a05_only`** | Full risk-matched α grid on torch **0–4**: only training **α = 0.05** yields any raw sweep point that clears; among **selected** points, clears appear only at α=0.05 on seeds **1** and **3** (seed 4 selects α=0.05 but cost stays above champion). This is **grid-specific**, not a cited α theorem. |
| **I** | `experiments/rq_liquid_robust/` | pooled **`fails`** | Fixed H recipes (RM α=0.05, Lag α=0.65), torch **0–5**: **5 / 12** arm×seed rows clear — above one-seed luck, below the half-bar for **`robust`**. Per-arm footnote: Lag alone hits **`robust`** (3/6 seeds); RM fixed clears **2/6** (seeds 1, 3). |
| **J** | `experiments/rq_liquid_anatomy/` | **`band_like`** | Among seeds with OOS `cvar95` ≤ champion, fixed RM (α=0.05) and Lag (α=0.65) recipes meet RQ-E-style Lit bars vs the champion band using **ε = champion δ-band 0.05** — not literal Leland ε→0 no-trade. Applies to these **mandate recipes**, not unconstrained Hyper (see K). |

**Seed pools (do not treat cells as exact twins).** RQ-H uses default `Hyper().torch_seeds`
(0–2). RQ-I and RQ-J expand torch inits to **0–5** with default bench train paths
(1001–1003). RQ-K uses torch **0–5** and train **1001–1006**. RQ-L uses torch **0–4**.
Cross-cell comparisons are on the **same OOS val seeds** and champion, not identical
training draws.

**How this sits under §3.** §3 already shows **20 bp** is flat. This footnote adds
that at **10 bp full-rate k-level** the same classical lesson holds for the default
learner (K), while H/I/L show **some** mandate recipes can clear on **some** seeds with
negligible cost improvement — too small and too seed-sensitive to reopen the §3 headline.
Do not promote liquid H–L as competing with "classic hedging is all you need."

Reproduce from `experiments/rq_liquid_*/NOTES.md` and `scripts/rq_liquid_*.py`; primary
artifacts under `experiments/rq_liquid_*/results/*.json`.
