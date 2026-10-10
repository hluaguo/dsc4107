// Final presentation slides. 16:9, compiled with Typst.
// Build: typst compile --root submission slides.typ slides.pdf

#set page(
  width: 33.87cm,
  height: 19.05cm,
  margin: 1.5cm,
  numbering: none,
)
#set text(font: ("New Computer Modern", "Libertinus Serif", "Noto Serif"), size: 19pt)
#set par(leading: 0.7em)

#show heading.where(level: 1): it => block(
  above: 0em, below: 1.0em,
)[
  #set text(size: 29pt, weight: "bold", fill: rgb("#173f5f"))
  #it
  #v(-0.2em)
  #line(length: 100%, stroke: 1.6pt + rgb("#173f5f"))
]

#show figure.caption: set text(size: 13pt)

#let accent = rgb("#b23a2e")
#let muted = luma(35%)

#let bullet(body) = block(above: 0.5em, below: 0.5em)[
  #grid(
    columns: (0.55cm, 1fr),
    gutter: 0em,
    [#text(fill: accent)[#sym.bullet]], body,
  )
]

#let small = text.with(size: 15pt, fill: muted)

// ---------------- 1. title ----------------
#align(center)[
  #v(3.0cm)
  #text(size: 40pt, weight: "bold")[Hedging or Harvesting?]
  #v(0.6em)
  #text(size: 24pt)[Classical versus Learned Option Hedging under Transaction Costs]
  #v(1.2em)
  #text(size: 18pt, style: "italic")[
    Replications of Sepp (2013) and Buehler et al. (2019), with a head-to-head comparison
  ]
  #v(2.4cm)
  #text(size: 17pt)[DSC4107 Final Project]
  #v(0.3em)
  #text(size: 17pt)[Group members: _[to be completed]_]
  #v(0.3em)
  #text(size: 17pt)[November 2026]
]

#pagebreak()

// ---------------- 2. problem ----------------
#heading[The problem]

#bullet[Sell an at-the-money one-month call, hedge it with the underlying.]
#bullet[Trades happen at discrete times, and each one costs a fixed fraction of notional.]
#bullet[More trading tracks the delta but pays more cost; less trading saves cost but leaves variance.]

#v(0.6em)
Two papers attack the same trade-off from opposite ends:

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    #block(fill: luma(96%), inset: 12pt, radius: 3pt)[
      *Sepp (2013)* \
      Closed-form P&L statistics for delta hedging. Maximises the Sharpe ratio over rebalancing frequency, then converts it to price and delta bands.
    ]
  ],
  [
    #block(fill: luma(96%), inset: 12pt, radius: 3pt)[
      *Buehler et al. (2019)* \
      A neural network chooses the hedge from market state, trained on a convex risk measure of terminal P&L.
    ]
  ],
)

#v(0.8em)
#text(fill: accent, weight: "bold")[Neither paper runs the two against each other under controlled conditions. That comparison is our contribution.]

#pagebreak()

// ---------------- 3. research questions ----------------
#heading[Research questions]

#v(0.4em)
#bullet[Can a Monte Carlo simulator reproduce Sepp's closed-form statistics, tables and figures?]
#v(0.5em)
#bullet[Can we reproduce the main claim of Deep Hedging, that a trained network beats Black-Scholes delta hedging under transaction costs in a Heston market?]
#v(0.5em)
#bullet[On one testbed, which approach gives the better risk-adjusted P&L, and does the ranking change with the market model and the size of the transaction cost?]

#v(1.2em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  The answer to RQ3 turns out to depend on what "better" means: Sharpe or tail risk.
]

#pagebreak()

// ---------------- 4. methods ----------------
#heading[Methods]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *Testbed*
    #bullet[Four market models: GBM, jump-diffusion, Heston, Heston with jumps]
    #bullet[Sepp's parameters, so replication lines up]
    #bullet[One ATM one-month call, one proportional cost, one shared P&L rollout]
    #bullet[Cost charged on opening and closing trades too]
  ],
  [
    *Comparison design*
    #bullet[Classical baseline re-tuned in every cell, so a stale setting cannot manufacture a win]
    #bullet[Learned network: state is moneyness, time, current position; CVaR loss]
    #bullet[Disjoint train, validation, tuning, search and holdout seed pools]
    #bullet[Every number averages three training seeds; per-seed minimum and win rate reported]
  ],
)

#v(1.0em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  *Correctness anchor.* With zero cost the learned policy must recover the Black-Scholes delta. If it does not, no learned result is trustworthy.
]

#pagebreak()

// ---------------- 5. sepp ----------------
#heading[Replication 1: Sepp (2013)]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #bullet[Closed forms for mean P&L `u`, cost `c`, local variance `p`, variance floor `f`]
    #bullet[Largest relative error in `u`: 3.9%]
    #bullet[GBM and SVJ case I match to 0.0001]
    #bullet[Monte Carlo cross-check (60,000 paths) agrees on mean and volatility]
    #v(0.5em)
    *The jump-diffusion fix* \
    The JD branch had dropped a factor `q = sqrt(pi^3)/4` that the other branches carry. Restoring it moves `p` from 24% below the paper to 5.6% above and brings the analytic JD Sharpe onto the simulated one.
  ],
  [
    #figure(
      image("../report/figures/sepp_sharpe_caseI.png", width: 100%),
      caption: [Sharpe against hedging frequency, case I.],
    )
  ],
)

#pagebreak()

// ---------------- 6. deep hedging ----------------
#heading[Replication 2: Deep Hedging]

#grid(
  columns: (0.9fr, 1.1fr),
  gutter: 1.2em,
  [
    #bullet[Frictionless check: correlation 0.9907, slope 1.148, RMSE 0.0581]
    #bullet[The learned policy rediscovers the delta, which validates the implementation]
    #bullet[In Heston, it matches the delta hedge at zero cost and pulls ahead as cost rises]
    #bullet[This reproduces the paper's qualitative claim]
  ],
  [
    #figure(
      image("../report/figures/sharpe_vs_cost.png", width: 100%),
      caption: [Heston Sharpe against cost.],
    )
  ],
)

#pagebreak()

// ---------------- 7. edge map ----------------
#heading[The head-to-head comparison]

#grid(
  columns: (1fr, 1.05fr),
  gutter: 1.0em,
  [
    #bullet[*Near zero cost:* the two approaches tie, as the frictionless check predicts]
    #bullet[*At 20 bp:* flat, edges from -0.09 to +0.02; the jump models favour classical]
    #bullet[*Above 50 bp:* a large Sharpe edge, but the P&L standard deviation triples]
    #v(0.6em)
    #text(fill: accent, weight: "bold")[The high-cost edge is the volatility risk premium being harvested, not better hedging.]
  ],
  [
    #figure(
      image("../report/figures/edge_map_fixed.png", width: 100%),
      caption: [Learned minus classical Sharpe across four models and five costs.],
    )
  ],
)

#pagebreak()

// ---------------- 8. robustness ----------------
#heading[Is the moderate-cost edge real?]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #bullet[Locked holdout, five unseen seeds: mean edge +0.0339, minimum +0.0293, win rate 1.00]
    #bullet[Bootstrap intervals on the validation pool exclude zero]
    #bullet[So the small Heston 20 bp edge is not seed or path noise]
    #v(0.5em)
    #text(fill: accent, weight: "bold")[But it is not free.]
    #bullet[The learned P&L has the same spread as the band]
    #bullet[Skew -0.70 against -0.01]
    #bullet[CVaR at 95% is 24% worse]
  ],
  [
    #figure(
      image("../report/figures/distribution.png", width: 100%),
      caption: [Terminal P&L on the frozen benchmark.],
    )
  ],
)

#pagebreak()

// ---------------- 9. sidequest ----------------
#heading[Hedging or harvesting: the mechanism]

#grid(
  columns: (1.05fr, 1fr),
  gutter: 1.0em,
  [
    #bullet[The CVaR tail fraction `alpha` is the dial. Large `alpha` collapses turnover and triples the P&L spread: the network stops hedging and collects the premium.]
    #bullet[Removing the current position from the state costs Sharpe and raises turnover, so position feedback produces the band.]
    #bullet[A participation dial `lambda` in the classical family tunes "how much delta" and "how often".]
    #bullet[At 0-20 bp the tuned dial is near the textbook delta and the network stays 0.02 to 0.05 above it.]
    #bullet[Above 20 bp the dial goes to zero, and the network is below do-nothing.]
  ],
  [
    #figure(
      image("../report/figures/participation_frontier.png", width: 100%),
      caption: [Participation frontier.],
    )
  ],
)

#pagebreak()

// ---------------- 10. conclusion ----------------
#heading[Conclusion]

#v(0.6em)
#bullet[The extra freedom of a neural hedger does not buy a better hedge in the benchmark regime.]
#v(0.5em)
#bullet[At 20 bp the network ties a well-tuned classical delta band, and the surviving holdout edge comes with a worse tail.]
#v(0.5em)
#bullet[Where it posts large gains, above roughly 50 bp, it has stopped hedging, and a classical family with the same freedom does that better.]
#v(0.5em)
#bullet[The replications hold up: Sepp's coefficients match Table 2, the closed forms agree with Monte Carlo, and the frictionless network recovers Black-Scholes.]

#v(1.0em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  Report any learned hedging edge next to its P&L spread. Sharpe alone hides the change of risk profile.
]

#pagebreak()

// ---------------- 11. reproduce ----------------
#heading[Reproduce it]

#v(0.4em)
#text(size: 22pt)[#link("https://github.com/hluaguo/dsc4107")[github.com/hluaguo/dsc4107]]

#v(0.8em)
```bash
git clone https://github.com/hluaguo/dsc4107.git
cd dsc4107
uv sync
uv run pytest -q                      # 24 checks
uv run python scripts/sepp_replication.py
uv run python scripts/deep_hedging_replication.py
uv run python scripts/comparison_study.py --fixed-hyper
uv run python scripts/holdout_eval.py
```

#v(0.6em)
#small[Full tables, source code and raw results are in the written report appendices and the `submission/` package.]
