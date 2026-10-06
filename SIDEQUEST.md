# Sidequest: is the learned edge just a leverage choice?

> This is an exploratory probe, kept separate from `REPORT.md` and from the
> plan's results. It imports the frozen modules read-only and writes only under
> `results/sidequest/`. Nothing in the original plan was changed.

## The question

`REPORT.md` found large high-cost Sharpe edges for the learned hedger whose P&L
standard deviation grows with cost, and read them as premium harvesting plus a
structural blind spot in the classical family: every classical candidate opens
a full delta at `t = 0`, so none can "hedge less". That interpretation was an
assertion. This probe tests it by adding exactly the missing degree of freedom.

The extended classical family, `scripts/sidequest_participation.py`, holds
`lam * BS_delta` and rebalances every `frequency` steps. It contains the
textbook time-scheduled delta (`lam = 1`, `frequency = 1`) and the do-nothing
hedge (`lam = 0`). It is tuned on `tune_seeds`, the same disjoint pool the
classical baseline uses, and scored on `val_seeds`. If a tuned scalar `lam`
captures the network's edge, the network's extra freedom is worth little beyond
a leverage choice.

## Result

| model | cost (bp) | tuned λ* | freq* | participation Sharpe | learned Sharpe | learned − participation | no-hedge Sharpe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gbm | 0 | 1.00 | 1 | +1.228 | +1.258 | **+0.030** | +0.165 |
| gbm | 20 | 1.00 | 3 | +0.308 | +0.347 | **+0.039** | +0.165 |
| gbm | 50 | 0.00 | 1 | +0.165 | +0.073 | −0.092 | +0.165 |
| gbm | 100 | 0.00 | 1 | +0.165 | +0.161 | −0.004 | +0.165 |
| gbm | 200 | 0.00 | 1 | +0.165 | +0.158 | −0.008 | +0.165 |
| jd | 0 | 1.00 | 1 | +0.121 | +0.116 | −0.005 | +0.053 |
| jd | 20 | 0.00 | 1 | +0.053 | −0.081 | −0.135 | +0.053 |
| jd | 50 | 0.00 | 1 | +0.053 | −0.149 | −0.202 | +0.053 |
| jd | 100 | 0.00 | 1 | +0.053 | +0.044 | −0.009 | +0.053 |
| jd | 200 | 0.00 | 1 | +0.053 | +0.042 | −0.011 | +0.053 |
| heston | 0 | 1.00 | 1 | +1.139 | +1.159 | **+0.020** | +0.187 |
| heston | 20 | 1.00 | 3 | +0.317 | +0.366 | **+0.049** | +0.187 |
| heston | 50 | 0.00 | 1 | +0.187 | +0.100 | −0.087 | +0.187 |
| heston | 100 | 0.00 | 1 | +0.187 | +0.179 | −0.007 | +0.187 |
| heston | 200 | 0.00 | 1 | +0.187 | +0.174 | −0.012 | +0.187 |
| svj | 0 | 0.75 | 1 | +0.124 | +0.113 | −0.012 | +0.060 |
| svj | 20 | 0.00 | 1 | +0.060 | −0.062 | −0.122 | +0.060 |
| svj | 50 | 0.00 | 1 | +0.060 | −0.092 | −0.152 | +0.060 |
| svj | 100 | 0.00 | 1 | +0.060 | +0.050 | −0.010 | +0.060 |
| svj | 200 | 0.00 | 1 | +0.060 | +0.054 | −0.006 | +0.060 |

Frontier plot: `results/sidequest/participation_frontier.png`.

## What it means

1. **Where a hedge is worth holding (0-20 bp), the network does add value.** The
   tuned participation is the textbook delta (`lam ~ 1`) at a slower calendar,
   and the learned hedger sits +0.02 to +0.05 above it on GBM and Heston. This
   matches the small, genuine edge the report found, and it is not a leverage
   effect.

2. **Above ~20 bp, the tuned participation collapses to `lam = 0`.** Doing
   nothing is the best strategy in the whole extended classical family, because
   the volatility risk premium (25% implied against 20% realised) pays more than
   any hedge that has to trade.

3. **The learned hedger is *below* the do-nothing strategy in every high-cost
   cell** (−0.004 to −0.20). So the large high-cost "edge" over the report's
   classical family is entirely an artefact of that family being forced to open
   a full delta at `t = 0`. The network is not merely harvesting the premium
   well; it is a slightly *worse* premium harvester than a trivial `lam = 0`
   rule. Its under-hedging is partial, not complete.

4. **The extra freedom the network has over the scalar is worth about +0.02 to
   +0.05 Sharpe**, and only in the regime where hedging is actually worthwhile.

This sharpens rather than contradicts the report. The report said the high-cost
win is a change of risk profile; the probe shows the change is *not even the
best* change of risk profile. If the comparison is meant to measure hedge
quality rather than a forced-trading penalty, the classical family needs an
initial-trade gate (or a `lam = 0` member), and the high-cost cells should be
read as "both strategies are past the point where hedging pays", not as a win.
