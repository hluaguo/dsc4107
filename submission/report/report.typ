// Final written report: classical versus learned option hedging.
// Main body target: 2000 words or fewer, references and appendices excluded.

#set document(
  title: "Classic hedging is all you need",
  author: "DSC4107 Group",
)

#set page(
  paper: "a4",
  margin: (x: 2.3cm, y: 2.4cm),
  numbering: "1",
  number-align: center,
  footer: context [
    #set text(size: 8pt, fill: luma(45%))
    Classic hedging is all you need #h(1fr) #counter(page).display()
  ],
)

#set text(
  font: ("New Computer Modern", "Libertinus Serif", "Noto Serif"),
  size: 10.5pt,
  lang: "en",
)
#set par(justify: true, leading: 0.62em, spacing: 0.9em)
#set raw(tab-size: 2)

#show heading.where(level: 1): it => block(
  above: 1.4em,
  below: 0.7em,
)[
  #set text(size: 14pt, weight: "bold")
  #it
]
#show heading.where(level: 2): it => block(
  above: 1.0em,
  below: 0.5em,
)[
  #set text(size: 11.5pt, weight: "bold")
  #it
]

#show figure: set block(breakable: true)
#show figure.caption: set text(size: 9pt)

// ---------- title page ----------
#page(header: none, footer: none)[
  #v(2.6cm)
  #align(center)[
    #text(size: 21pt, weight: "bold")[Classic hedging is all you need]
    #v(0.5em)
    #text(
      size: 15pt,
    )[Classical versus learned option hedging under transaction costs]
    #v(1.2em)
    #text(size: 11pt, style: "italic")[
      Replications of Sepp (2013) and Buehler et al. (2019), with a head-to-head comparison
    ]
  ]
  #v(3.4cm)
  #align(center)[
    #text(size: 11pt)[
      DSC4107 Final Project \
      #v(0.3em)
      Lau Cheuk Yin
    ]
  ]
]

#pagebreak()

// =====================================================================
#heading(outlined: true)[Abstract]
// =====================================================================

Deep hedging is sold as strictly freer than classical bands: given costs, a neural
policy should beat a tuned delta schedule. Sepp (2013) and Buehler et al. (2019)
attack the same friction from opposite ends, yet neither runs a fair head-to-head on
one simulator with a classical baseline re-tuned per regime. We do. At the course's
nominal 20 bp friction, re-tuned classical does not lose in any hedging sense that
matters: Sharpe edges sit between about −0.09 and +0.01; a locked holdout confirms
only +0.0339; and the learned terminal P&L buys mean with skew −0.70 against −0.01
and a 24% worse CVaR at 95%. Classic hedging is enough; deep is not a uniform
upgrade. Above roughly 50 bp, large Sharpe gaps track P&L standard deviation rising
toward 3–4 and mean toward the option premium—premium harvesting, not tighter
hedges. A separate liquid 10 bp side study (Section 5) is recipe-specific and
brittle and does not reopen the headline.

// =====================================================================
#heading(outlined: true)[1. Introduction]
// =====================================================================

A desk sells an at-the-money one-month call and hedges with the underlying at
discrete times; each trade pays a proportional cost. More rebalancing tracks the
delta but spends more; less rebalancing saves cost but leaves variance.

Sepp (2013) maximises the Sharpe ratio inside an analytic delta-hedging family and
maps the optimum to price and delta bands. Buehler et al. (2019) replace that family
with a network trained on a convex risk measure of terminal P&L, free to trade less
or hedge partially. The usual reading is that extra freedom should dominate once
costs bite—but that expectation has never been tested on the same paths, contract
and cost schedule with a classical side re-tuned in every cell.

*The headline before the details.* On our frozen 20 bp benchmark (Heston, volatility
risk premium, classical band re-tuned per map cell), learned minus classical Sharpe
edges run from about −0.09 to +0.01; jump models favour classical. A locked holdout
on five unseen seeds gives mean edge +0.0339 (minimum +0.0293, win rate 1.00)—real
but tiny. Where a small Heston edge survives, it is not free: terminal P&L spread is
almost unchanged (std 0.626 vs 0.613) but skew is −0.70 against −0.01 and CVaR at
95% is 1.321 against 1.066, a 24% worse left tail. Classic hedging is all you need
at nominal friction; deep hedging is not a uniform upgrade.

*The twist.* Above roughly 50 bp the network posts large Sharpe wins, yet its P&L
standard deviation climbs from about 0.5 toward 3–4 while mean P&L approaches the
option premium—the signature of premium harvesting, not better hedging. Section 4
delivers that comparison after Section 3 checks that the simulator and both
papers' implementations are trustworthy.

// =====================================================================
#heading(outlined: true)[2. A fair head-to-head]
// =====================================================================

*One testbed.* Four Sepp market models (GBM, jump-diffusion, Heston, Heston with
jumps), one at-the-money one-month call, one proportional cost $c$ on every hedge
trade including open and close. The head-to-head map spans four models and five cost
levels; the classical baseline is re-tuned in every cell so a stale band cannot
manufacture a network win.

*Two implementations, one rollout.* `hedging/sepp.py` supplies the classical family;
`hedging/deep.py` is an MLP on log-moneyness, time and position with CVaR loss at
$alpha = 0.65$. Both use the same P&L rollout and cost accounting.

*Controls.* Disjoint train, validation, tuning, search and holdout seed pools;
reported means average three training seeds with per-seed minima and win rates
published. The frozen benchmark adds a 25% implied versus 20% realised volatility
risk premium so "do not hedge" cannot dominate Sharpe. The frictionless check is the
correctness anchor: at zero cost the learned policy must recover the Black-Scholes
delta.

// =====================================================================
#heading(outlined: true)[3. Replication checks]
// =====================================================================

#heading(outlined: true)[3.1 Sepp (2013)]

The closed forms track Table 2. The largest relative error in $u$ is 3.9%, and
the GBM case I pair (0.5693 against 0.5692) and the SVJ case I pair (0.6551
against 0.6550) differ by 0.0001. The optimum is interior and moves to lower
frequency as the model gains complexity: extra trading cannot remove a variance
floor $f$, so paying more cost buys less. GBM case I optimises at 242 rebalances
with a Sharpe of 0.83; Heston case I at 75 with 0.43.

The jump-diffusion row needed a real fix. The JD branch had dropped a factor
$q = sqrt(pi^3)\/4$ that the GBM and SVJ branches carry. `archive/scripts/diagnose_jd.py`
shows the second-moment gamma was correct to 0.2% all along; only the coefficient
was wrong. Restoring $q$ moves $p$ from 24% below the paper to 5.6% above it and
brings the analytic JD Sharpe (0.65) onto the simulated one (0.65) in case I.
The remaining case-II gap of about 12% in volatility is Sepp's own variance
approximation. The Monte Carlo cross-check passes for the other three models in
both mean and volatility.

#heading(outlined: true)[3.2 Deep hedging (Buehler et al. 2019)]

At zero cost the learned policy rediscovers the delta. Over 20,000 paths the
correlation between learned position and Black-Scholes delta is 0.9907, with
slope 1.148 and RMSE 0.0581; the mean learned position (0.499) is within 2% of
the mean delta (0.508).

The Heston experiment then reproduces the paper's qualitative claim. At zero
cost the network matches a delta hedge (Sharpe 1.154 against 1.139). As cost
rises it moves ahead of both every-step delta hedging and the frozen band.

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Cost (bp)*],
      [*Learned*],
      [*BS delta*],
      [*Frozen band*],
      [*Edge vs delta*],
      [*Edge vs band*],
    ),
    [0], [1.154], [1.139], [0.987], [+0.016], [+0.167],
    [20], [0.373], [0.254], [0.352], [+0.119], [+0.020],
    [50], [0.105], [-0.970], [-0.526], [+1.075], [+0.631],
    [100], [0.182], [-2.276], [-1.573], [+2.458], [+1.755],
  ),
  caption: [Heston Sharpe by cost. The learned hedger pulls ahead as cost rises; Section 4 pairs Sharpe with P&L spread.],
)

// =====================================================================
#heading(outlined: true)[4. Head-to-head comparison]
// =====================================================================

The edge map has three regimes, and the P&L standard deviations explain the
third.

Near zero cost the two approaches tie, which is the expected outcome once the
frictionless check passes. At 20 bp—the benchmark's own cost—the comparison is
essentially flat: edges run from −0.09 to +0.01, and on the jump models the
learned hedger is slightly behind. This is the headline result: classic hedging
is all you need at nominal friction. Above 50 bp the network's Sharpe edge becomes
large and consistent, but its P&L standard deviation climbs from about 0.5 to 3 or
4 while its mean P&L approaches the option premium. The network has learned to
stop hedging and collect the volatility risk premium. That is a legitimate
strategy but a different product from a hedge; a Sharpe win produced by not
hedging is a change of risk profile, not evidence of better hedging.

#figure(
  image("figures/edge_map_fixed.png", width: 88%),
  caption: [Learned minus classical Sharpe. High-cost wins arrive with a tripled P&L spread.],
)

The locked holdout confirms the moderate-cost result and its size. On five
unseen seeds the mean edge over the frozen band is +0.0339, with a minimum of
+0.0293 and a win rate of 1.00—the same order of magnitude as the validation
edge for Heston at 20 bp (+0.014 fixed hyperparameters, +0.021 after search).
The small benchmark win is real but small by design: this is the regime where the
two strategies nearly tie.

The Sharpe edge is not a tail win. On every one of the five holdout seeds the
learned CVaR95 is higher than the band, from 1.273 to 1.318 against 1.036 to
1.080.

It is also not a free improvement. On the validation distribution the two P&L
distributions have almost the same spread, but the learned one has skew -0.70 against -0.01 and a
CVaR at 95% of 1.321 against 1.066, which is a 24% worse tail. Whether the edge
counts as a win depends on whether the desk scores Sharpe or tail risk, and the
two measures disagree.

#heading(outlined: true)[4.1 Mechanism checks (not the headline)]

CVaR tail fraction, state ablations and a participation dial in an extended
classical family show that high-cost Sharpe gaps track under-hedging and the
map's rule that classical strategies always open a full delta at $t = 0$. Details
are in Appendix A figures; they do not overturn the 20 bp tie.

// =====================================================================
#heading(outlined: true)[5. Footnote: liquid 10 bp full-rate]
// =====================================================================

Sepp Case I charges $k/2$ on $|Delta delta| dot S$ per eq.~(2.5); our simulator uses
a full-rate ledger `cost · |trade| · S`. At Sepp liquid-index $k$ the side study sets
`cost = 0.001` (numerically Sepp's $k$, not Case I economics) and re-sweeps a 10 bp
classical hull—never the frozen 20 bp `classical_best.json`—asking whether any deep
recipe clears the validation-Sharpe champion on both cost and tail. The chain is soft
and mandate-specific: H clears on narrow margins, K is dominated by default training,
L clears only at selected $alpha = 0.05$ seeds, I fails pooled robustness, and J's
clearing recipes look band-like. It does not reopen the 20 bp headline.

// =====================================================================
#heading(outlined: true)[6. Limitations]
// =====================================================================

This is a course-scale testbed: 80 to 100 epochs, 2,048 training paths per seed,
three seeds, a small network and eight Optuna trials per cell. The direction of
the results is stable across fixed and searched configurations, but the exact
edges are not paper-grade magnitudes. The classical family in the map always
opens a full initial hedge, so it cannot express "under-hedge" or "hold flat";
the sidequest repairs that for interpretation but stayed outside the frozen
benchmark. The comparison scripts are covered by smoke tests rather than the
unit suite, so their results rest on seed discipline.

// =====================================================================
#heading(outlined: true)[7. Conclusion]
// =====================================================================

Classic hedging is all you need at the course's 20 bp benchmark. The trained
network ties a well-tuned classical delta band on Sharpe (edges about −0.09 to
+0.01), and the holdout confirms only a tiny gap. Where a small Heston edge
survives, it buys a worse left tail in CVaR. High-cost Sharpe advantages are
premium harvesting—visible in the paired P&L standard deviations—not better
hedges, and deep is not a uniform upgrade across the comparison grid. The
replications stand: Sepp's coefficients match Table 2, closed forms agree with
Monte Carlo, and the frictionless network recovers the Black-Scholes delta. The
liquid 10 bp footnote (Section 5) does not change that conclusion.

// =====================================================================
#heading(outlined: true)[References]
// =====================================================================

#enum(
  numbering: "[1]",
  [H. Buehler, L. Gonon, J. Teichmann and B. Wood. Deep hedging. _Quantitative Finance_, 19(8):1271-1291, 2019. doi:10.1080/14697688.2019.1571683.],
  [A. Sepp. When you hedge discretely: Optimization of Sharpe ratio for delta-hedging strategy under discrete hedging and transaction costs. _Journal of Investment Strategies_, 3(1):19-59, 2013.],
)

#pagebreak()

// =====================================================================
#heading(outlined: true)[Appendix A. Extended tables]
// =====================================================================

#heading(outlined: true)[A.1 Sepp Table 2 reproduction]

#figure(
  table(
    columns: 10,
    align: (left, left, right, right, right, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Case*],
      [*Model*],
      [*u ours*],
      [*u paper*],
      [*p ours*],
      [*p paper*],
      [N\* ours],
      [N\* paper],
      [S ours],
      [S paper],
    ),
    [I],
    [GBM],
    [0.5693],
    [0.5692],
    [0.2519],
    [0.2461],
    [242],
    [239],
    [0.83],
    [0.89],

    [I],
    [JD],
    [0.5640],
    [0.5510],
    [0.2118],
    [0.2006],
    [147],
    [138],
    [0.65],
    [0.66],

    [I],
    [Heston],
    [0.6621],
    [0.6703],
    [0.2583],
    [0.2524],
    [75],
    [74],
    [0.43],
    [0.45],

    [I],
    [SVJ],
    [0.6551],
    [0.6550],
    [0.2164],
    [0.2085],
    [70],
    [68],
    [0.43],
    [0.45],

    [II],
    [GBM],
    [1.8080],
    [1.8558],
    [0.6138],
    [0.5998],
    [130],
    [129],
    [1.18],
    [1.30],

    [II],
    [JD],
    [1.7826],
    [1.7628],
    [0.4765],
    [0.4423],
    [88],
    [75],
    [1.01],
    [0.99],

    [II],
    [Heston],
    [2.0054],
    [2.0876],
    [0.6359],
    [0.6214],
    [49],
    [49],
    [0.65],
    [0.73],

    [II],
    [SVJ],
    [1.9738],
    [1.9995],
    [0.4903],
    [0.4634],
    [45],
    [43],
    [0.68],
    [0.72],
  ),
  caption: [Closed-form coefficients against Sepp's Table 2. Largest relative error in $u$ is 3.9%.],
)

#heading(outlined: true)[A.2 Monte Carlo cross-check at the analytic optimum]

#figure(
  table(
    columns: 7,
    align: (left, left, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Case*],
      [*Model*],
      [*P&L analytic*],
      [*P&L MC*],
      [*vol analytic*],
      [*vol MC*],
      [*Sharpe a/MC*],
    ),
    [I], [GBM], [0.00318], [0.00309], [0.00385], [0.00365], [0.83 / 0.85],
    [I], [JD], [0.00373], [0.00381], [0.00571], [0.00590], [0.65 / 0.65],
    [I], [Heston], [0.00511], [0.00537], [0.01191], [0.01173], [0.43 / 0.46],
    [I], [SVJ], [0.00515], [0.00542], [0.01187], [0.01168], [0.43 / 0.46],
    [II], [GBM], [0.01098], [0.01102], [0.00932], [0.00916], [1.18 / 1.20],
    [II], [JD], [0.01220], [0.01295], [0.01213], [0.01370], [1.01 / 0.95],
    [II], [Heston], [0.01526], [0.01695], [0.02338], [0.02369], [0.65 / 0.72],
    [II], [SVJ], [0.01539], [0.01709], [0.02280], [0.02380], [0.68 / 0.72],
  ),
  caption: [60,000-path cross-check. Mean matches for every model; the residual case-II volatility gap is the paper's own approximation.],
)

#heading(outlined: true)[A.3 Full edge map, fixed configuration]

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Model*],
      [*Cost*],
      [*Edge*],
      [*Edge min*],
      [*Win*],
      [*Deep Sharpe*],
      [*Classical Sharpe*],
      [*Deep std / Cls std*],
    ),
    [GBM], [0 bp], [+0.030], [+0.014], [1.00], [1.258], [1.228], [0.46 / 0.47],
    [GBM], [20 bp], [-0.001], [-0.008], [0.50], [0.347], [0.348], [0.55 / 0.58],
    [GBM], [50 bp], [+0.038], [+0.032], [1.00], [0.073], [0.035], [1.72 / 1.74],
    [GBM],
    [100 bp],
    [+0.422],
    [+0.416],
    [1.00],
    [0.161],
    [-0.261],
    [3.44 / 1.74],

    [GBM],
    [200 bp],
    [+1.009],
    [+1.004],
    [1.00],
    [0.158],
    [-0.851],
    [3.48 / 1.74],

    [JD], [0 bp], [-0.006], [-0.009], [0.00], [0.116], [0.121], [1.98 / 1.89],
    [JD], [20 bp], [-0.090], [-0.096], [0.00], [-0.081], [0.008], [1.94 / 2.47],
    [JD],
    [50 bp],
    [-0.032],
    [-0.037],
    [0.00],
    [-0.149],
    [-0.117],
    [2.19 / 2.47],

    [JD],
    [100 bp],
    [+0.369],
    [+0.361],
    [1.00],
    [0.044],
    [-0.325],
    [4.21 / 2.47],

    [JD],
    [200 bp],
    [+0.785],
    [+0.770],
    [1.00],
    [0.042],
    [-0.743],
    [4.37 / 2.46],

    [Heston],
    [0 bp],
    [+0.020],
    [+0.007],
    [1.00],
    [1.159],
    [1.139],
    [0.51 / 0.52],

    [Heston],
    [20 bp],
    [+0.014],
    [+0.006],
    [1.00],
    [0.366],
    [0.352],
    [0.61 / 0.61],

    [Heston],
    [50 bp],
    [+0.054],
    [+0.047],
    [1.00],
    [0.100],
    [0.046],
    [1.60 / 1.74],

    [Heston],
    [100 bp],
    [+0.430],
    [+0.425],
    [1.00],
    [0.179],
    [-0.251],
    [3.11 / 1.73],

    [Heston],
    [200 bp],
    [+1.020],
    [+1.015],
    [1.00],
    [0.174],
    [-0.846],
    [3.17 / 1.73],

    [SVJ], [0 bp], [-0.012], [-0.016], [0.00], [0.113], [0.125], [2.11 / 1.92],
    [SVJ],
    [20 bp],
    [-0.076],
    [-0.082],
    [0.00],
    [-0.062],
    [0.014],
    [2.05 / 2.45],

    [SVJ],
    [50 bp],
    [+0.020],
    [+0.016],
    [1.00],
    [-0.092],
    [-0.112],
    [2.36 / 2.45],

    [SVJ],
    [100 bp],
    [+0.372],
    [+0.368],
    [1.00],
    [0.050],
    [-0.322],
    [3.97 / 2.45],

    [SVJ],
    [200 bp],
    [+0.798],
    [+0.789],
    [1.00],
    [0.054],
    [-0.744],
    [4.10 / 2.44],
  ),
  caption: [Learned minus classical Sharpe. The standard-deviation columns are the guard rail: where the edge is large, the network has stopped hedging.],
)

#heading(outlined: true)[A.4 Optuna hyperparameter search]

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Model*],
      [*Cost*],
      [*Search edge*],
      [*Reported edge*],
      [*Win*],
      [*Deep Sharpe*],
      [*Classical Sharpe*],
      [*Deep std / Cls std*],
    ),
    [Heston],
    [20 bp],
    [+0.011],
    [+0.021],
    [1.00],
    [0.374],
    [0.352],
    [0.60 / 0.61],

    [Heston],
    [50 bp],
    [+0.140],
    [+0.139],
    [1.00],
    [0.185],
    [0.046],
    [3.09 / 1.74],

    [Heston],
    [100 bp],
    [+0.436],
    [+0.435],
    [1.00],
    [0.185],
    [-0.251],
    [3.18 / 1.73],

    [JD], [20 bp], [-0.075], [-0.077], [0.00], [-0.069], [0.008], [1.96 / 2.47],
    [JD], [50 bp], [+0.158], [+0.156], [1.00], [0.039], [-0.117], [3.98 / 2.47],
    [JD],
    [100 bp],
    [+0.379],
    [+0.377],
    [1.00],
    [0.052],
    [-0.325],
    [4.33 / 2.47],
  ),
  caption: [Optuna search ranked on a disjoint search pool. The search does not change the qualitative picture.],
)

#heading(outlined: true)[A.5 Terminal P&L distribution on the frozen benchmark]

#figure(
  table(
    columns: 6,
    align: (left, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Strategy*], [*Mean*], [*Std*], [*Skew*], [*CVaR 95%*], [*5% quantile*]
    ),
    [Learned], [0.240], [0.626], [-0.699], [1.321], [-0.880],
    [Band], [0.216], [0.613], [-0.010], [1.066], [-0.752],
  ),
  caption: [Same spread, worse left tail. The +0.03 Sharpe edge is partly bought with a 24% worse CVaR.],
)

#heading(outlined: true)[A.6 Sidequest participation frontier]

#figure(
  table(
    columns: 8,
    align: (left, right, right, right, right, right, right, right),
    stroke: 0.4pt + luma(75%),
    table.header(
      [*Model*],
      [*Cost*],
      [*Tuned lambda*],
      [*Freq*],
      [*Participation Sharpe*],
      [*Learned Sharpe*],
      [*Learned - part.*],
      [*No-hedge Sharpe*],
    ),
    [GBM], [0 bp], [1.00], [1], [+1.228], [+1.258], [+0.030], [+0.165],
    [GBM], [20 bp], [1.00], [3], [+0.308], [+0.347], [+0.039], [+0.165],
    [GBM], [100 bp], [0.00], [1], [+0.165], [+0.161], [-0.004], [+0.165],
    [GBM], [200 bp], [0.00], [1], [+0.165], [+0.158], [-0.008], [+0.165],
    [JD], [20 bp], [0.00], [1], [+0.053], [-0.081], [-0.135], [+0.053],
    [JD], [100 bp], [0.00], [1], [+0.053], [+0.044], [-0.009], [+0.053],
    [Heston], [0 bp], [1.00], [1], [+1.139], [+1.159], [+0.020], [+0.187],
    [Heston], [20 bp], [1.00], [3], [+0.317], [+0.366], [+0.049], [+0.187],
    [Heston], [100 bp], [0.00], [1], [+0.187], [+0.179], [-0.007], [+0.187],
    [Heston], [200 bp], [0.00], [1], [+0.187], [+0.174], [-0.012], [+0.187],
    [SVJ], [20 bp], [0.00], [1], [+0.060], [-0.062], [-0.122], [+0.060],
    [SVJ], [100 bp], [0.00], [1], [+0.060], [+0.050], [-0.010], [+0.060],
    [SVJ], [200 bp], [0.00], [1], [+0.060], [+0.054], [-0.006], [+0.060],
  ),
  caption: [The extended classical family (participation dial lambda). Where the network posts a high-cost Sharpe win, the tuned dial is zero and the network is below do-nothing.],
)

#heading(outlined: true)[A.7 Figures]

#figure(
  image("figures/sepp_sharpe_caseI.png", width: 78%),
  caption: [Sepp case I: Sharpe against hedging frequency for the four models. The optimum is interior once cost is positive.],
)

#figure(
  image("figures/sharpe_vs_cost.png", width: 78%),
  caption: [Heston deep-hedging replication: Sharpe against transaction cost for the learned, BS delta and band strategies.],
)

#figure(
  image("figures/objective_sweep.png", width: 92%),
  caption: [CVaR tail fraction as a turnover dial. Large $alpha$ collapses trading frequency and inflates P&L spread.],
)

#figure(
  image("figures/transfer_matrix.png", width: 78%),
  caption: [Cross-model transfer: training on one model and scoring on all four. The column order dominates, so the ranking is a property of the market.],
)

#figure(
  image("figures/distribution.png", width: 88%),
  caption: [Terminal P&L on the frozen benchmark. The learned distribution has the same spread as the band but a heavier left tail.],
)

#figure(
  image("figures/participation_frontier.png", width: 88%),
  caption: [Participation frontier. Above 20 bp the tuned dial is zero and the network sits below do-nothing.],
)

// =====================================================================
#pagebreak()
#heading(outlined: true)[Appendix B. Source code repository]
// =====================================================================

The complete source code is on GitHub:

#align(center)[
  #link("https://github.com/hluaguo/dsc4107")[
    #text(font: "DejaVu Sans Mono", size: 10.5pt)[github.com/hluaguo/dsc4107]
  ]
]

The repository holds every module used to produce the numbers in this report:
the four market simulators, the shared P&L rollout, the Sepp closed forms, the
learned hedger and the comparison and exploration scripts. It also carries the
test suite and the pinned environment. A snapshot is attached under
`submission/code/`, and the raw experimental data sits under `submission/data/`.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    stroke: 0.4pt + luma(75%),
    table.header([*Path*], [*Role*]),
    [hedging/market.py], [GBM, JD, Heston and SVJ simulators (frozen)],
    [hedging/bs.py], [Black-Scholes price and greeks (frozen)],
    [hedging/hedge.py], [Shared P&L rollout and classical strategies (frozen)],
    [hedging/metrics.py], [Sharpe, CVaR and the reported statistics (frozen)],
    [hedging/bench.py], [Frozen benchmark config and seed pools],
    [hedging/sepp.py], [Sepp (2013) closed forms and Sharpe optimisation],
    [hedging/deep.py], [Learned hedger: features, network, CVaR loss],
    [hedging/hyper.py], [Training hyperparameters],
    [scripts/sepp_replication.py], [Sepp Table 2, optima and Monte Carlo check],
    [scripts/deep_hedging_replication.py],
    [Frictionless check and Heston experiment],

    [archive/scripts/comparison_study.py], [Head-to-head map and Optuna add-on],
    [scripts/holdout_eval.py], [The single locked holdout run],
    [#raw("archive/scripts/explore_*.py")],
    [Objective, feature, transfer and robustness probes],

    [archive/scripts/sidequest_participation.py], [The participation-dial test],
    [tests/], [Analytic and accounting checks (24 tests)],
  ),
  caption: [Repository layout.],
)

// =====================================================================
#pagebreak()
#set page(margin: (x: 2.3cm, y: 2.4cm))
#set par(leading: 0.62em)
#heading(outlined: true)[Appendix C. How to reproduce]
// =====================================================================

The full environment is pinned in `uv.lock`. Clone the repository and sync it:

```bash
# from the repository root
git clone https://github.com/hluaguo/dsc4107.git
cd dsc4107
uv sync
uv run pytest -q                                          # 24 checks
uv run python scripts/sepp_replication.py                 # Sepp tables + figures
uv run python scripts/sepp_replication.py --mc --mc-paths 60000
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3
uv run python archive/scripts/comparison_study.py --fixed-hyper \
    --costs 0.0 0.002 0.005 0.01 0.02
uv run python archive/scripts/comparison_study.py --models heston jd \
    --costs 0.002 0.005 0.01 --trials 8
uv run python scripts/holdout_eval.py                     # final locked run
```

All training scripts select CUDA when it is present and fall back to the CPU
otherwise. Raw experimental data and figures are mirrored under
`submission/data/`.
