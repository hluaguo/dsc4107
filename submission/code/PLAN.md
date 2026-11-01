# Project plan: classical versus learned option hedging

## The topic

We replicate two papers and then compare them on a shared testbed. Sepp (2013), *When You Hedge Discretely*, derives analytic formulas for the profit and loss of delta hedging when rebalancing is discrete and each trade costs money, then finds the hedging frequency and price bands that maximise the Sharpe ratio. Buehler, Gonon, Teichmann and Wood (2019), *Deep Hedging*, sets the analytic strategy aside and trains a neural network to pick hedge positions under the same kind of frictions. The two papers attack one problem from opposite ends, and neither runs the head-to-head comparison. That comparison is the original part of this project.

## Research questions

1. Can a Monte Carlo simulator reproduce Sepp's formulas, tables and figures?
2. Can we reproduce the main claim of Deep Hedging, that a trained network beats Black-Scholes delta hedging under transaction costs in a Heston market?
3. On one testbed, which approach gives the better risk-adjusted profit and loss, and does the answer change with the market model and the size of the transaction cost?

## Replication of Sepp (2013)

We build a simulator that sells a vanilla option, hedges it by holding delta shares, and pays a proportional cost on every trade. It covers Sepp's four return processes: log-normal diffusion, jump-diffusion, stochastic volatility, and stochastic volatility with jumps. For each one we compute the mean P and L, the expected transaction costs, and the P and L volatility at several hedging frequencies, and check the numbers against Sepp's closed-form expressions. The last step reproduces the Sharpe-maximising frequency and band results in his Section 5.

The formulas in the appendix are the reference for the simulator, not the other way round. Where the simulation and the formula disagree, we track down which one is wrong before moving on.

## Replication of Buehler et al. (2019)

We implement the deep hedger in PyTorch. At each step the network reads the state (time, price, current hedge position, and the volatility state in the Heston case) and outputs the position to hold. Training minimises a convex risk measure of the terminal P and L, such as conditional value at risk or the entropy risk measure. We reproduce their Heston experiment with proportional transaction costs and compare the learned strategy against Black-Scholes delta hedging.

One check matters more than the rest. With zero transaction costs the network should recover something close to the Black-Scholes delta. If it does not, the implementation is wrong, whatever the graphs say.

## The comparison

Both strategies run on the same simulated paths, the same option, and the same cost levels. We ask three things:

- Does the trained network beat the optimally tuned classical strategy, or does it merely tie with it once the classical dials are set well?
- How does the ranking shift as transaction costs rise from zero to levels that clearly discourage trading?
- Does the ranking hold across the four market models, or does the classical band strategy win in some of them?

This is the question the papers leave open. Sepp optimises a small strategy family by hand; Buehler searches a much larger one with a network. Nobody has measured what the extra freedom buys under controlled conditions.

## Optimisation add-on

Two layers of optimisation sit on top of the replication.

The first comes from the papers themselves. Sepp maximises the Sharpe ratio over hedging frequency and band width, and we solve that optimisation directly. This gives the classical strategy its best possible setting, so the comparison is fair.

The second is a hyperparameter search with Optuna. We sweep the network size, learning rate, and training budget for the deep hedger, and the frequency and band parameters for the classical one, at each transaction cost level. The output is a map of which approach wins in which regime, with the boundary marked.

## Market models and testbed

The four models from Sepp serve as the testbeds, since Deep Hedging already works in a Heston setting and Heston is one of Sepp's four:

| Model | What it adds |
| --- | --- |
| Log-normal | Baseline; volatility is constant |
| Jump-diffusion | Overnight gaps and crashes |
| Stochastic volatility | Volatility clusters and drifts |
| Stochastic volatility with jumps | Both effects at once |

Parameters come from Sepp's Section 5 so the replication numbers line up, and we reuse them for the network experiments.

## Evaluation and correctness checks

Three checks keep us honest:

- The frictionless deep hedger should rediscover the Black-Scholes delta.
- The Monte Carlo P and L statistics should match Sepp's closed-form formulas within simulation error.
- Every reported Sharpe ratio carries a bootstrap confidence interval, so a small gap between strategies is not read as a win. Training runs use several seeds, since the network result has run-to-run variance.

We report mean P and L, P and L volatility, and Sharpe ratio for every strategy and regime, plus the loss distribution so the tail behaviour is visible.

## Deliverables

1. A simulator library with the four market models, proportional transaction costs, and a test suite checked against Sepp's formulas.
2. A notebook reproducing Sepp's Section 5 tables and figures.
3. A notebook reproducing the Heston experiment from Deep Hedging.
4. The comparison study with the Optuna sweeps and plots of where each strategy wins.
5. A written report covering the replication, the method, and the results.

## Risks

The analytic part of Sepp is heavy. We treat Appendix A as a reference and verify its results by simulation rather than re-deriving them.

The deep hedger reproduces a qualitative claim, not exact numbers. Training variance and seed choice affect the results, and we report ranges instead of single runs. The frictionless check and the seed spread tell us whether a result is real.

Data is not a risk here. Every experiment is simulated, so there is nothing to download and no licence to arrange.
