// Final presentation slides. 16:9, compiled with Typst.
// Build: typst compile --root . slides/slides.typ slides/slides.pdf

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
  #text(size: 40pt, weight: "bold")[Classic hedging is all you need]
  #v(0.6em)
  #text(size: 24pt)[Classical versus learned option hedging under transaction costs]
  #v(1.2em)
  #text(size: 18pt, style: "italic")[
    Sepp (2013) and Buehler et al. (2019) replications, plus a head-to-head comparison
  ]
  #v(2.4cm)
  #text(size: 17pt)[DSC4107 Final Project]
  #v(0.3em)
  #text(size: 17pt)[Lau Cheuk Yin]
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
#bullet[Once the classical side is re-tuned per cell, does the learned hedger beat the best band at 20 bp—and where does the ranking change with cost?]

#v(1.2em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  *Headline answer (RQ3):* at 20 bp, classic hedging is enough. High-cost Sharpe wins are premium harvesting, not better hedges.
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
    #bullet[Classical baseline re-tuned in every cell]
    #bullet[Learned network: moneyness, time, current position; CVaR loss]
    #bullet[Disjoint train, validation, tuning, search and holdout seed pools]
    #bullet[Every number averages three training seeds]
  ],
)

#v(1.0em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  *Correctness anchor.* With zero cost the learned policy must recover the Black-Scholes delta (correlation 0.9907).
]

#pagebreak()

// ---------------- 5. sepp ----------------
#heading[Replication 1: Sepp (2013)]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #bullet[Largest relative error in `u`: 3.9%]
    #bullet[GBM and SVJ case I match to 0.0001]
    #bullet[60,000-path Monte Carlo agrees on mean and volatility]
    #v(0.5em)
    *Jump-diffusion fix:* restore factor `q = sqrt(pi^3)/4`; analytic JD Sharpe matches simulation (0.65).
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
    #bullet[Frictionless check validates the implementation]
    #bullet[Heston: matches delta at zero cost; beats every-step delta as cost rises]
    #bullet[Qualitative claim of the paper is reproduced]
    #v(0.5em)
    #small[This is not the headline: the frozen 20 bp benchmark is Section 7.]
  ],
  [
    #figure(
      image("../report/figures/sharpe_vs_cost.png", width: 100%),
      caption: [Heston Sharpe against cost.],
    )
  ],
)

#pagebreak()

// ---------------- 7. 20 bp headline ----------------
#heading[20 bp: classical vs deep]

#grid(
  columns: (1fr, 1.05fr),
  gutter: 1.0em,
  [
    #bullet[*Near zero cost:* tie (frictionless check)]
    #bullet[*At 20 bp:* Sharpe edges −0.09 to +0.01 — essentially flat]
    #bullet[Jump models: classical slightly ahead]
    #bullet[Holdout: mean edge +0.0339 — real but tiny]
    #v(0.6em)
    #text(fill: accent, weight: "bold")[Classic hedging is all you need at nominal friction.]
  ],
  [
    #figure(
      image("../report/figures/edge_map_fixed.png", width: 100%),
      caption: [Learned minus classical Sharpe (fixed hyperparameters).],
    )
  ],
)

#pagebreak()

// ---------------- 8. tail + high cost ----------------
#heading[Same spread, worse tail — and high-cost “wins”]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.0em,
  [
    #bullet[Learned vs band: mean 0.240 vs 0.216, std 0.626 vs 0.613]
    #bullet[Skew −0.70 vs −0.01; CVaR 95% 24% worse]
    #bullet[Sharpe edge buys mean with a fatter left tail]
    #v(0.5em)
    #bullet[*Above ~50 bp:* large Sharpe edge but std rises to 3–4]
    #bullet[P&L mean approaches the option premium → premium harvesting]
  ],
  [
    #figure(
      image("../report/figures/distribution.png", width: 100%),
      caption: [Terminal P&L on the frozen benchmark.],
    )
  ],
)

#pagebreak()

// ---------------- 9. liquid footnote ----------------
#heading[Liquid 10 bp (footnote only)]

#bullet[Sepp Case I charges *k/2* on $|Delta delta| dot S$ (eq.~2.5); side study at liquid-index *k* uses our *full-rate* ledger (`cost = 0.001` — numerically Sepp's *k*, not Case I economics)]
#bullet[10 bp classical hull re-swept; not `classical_best.json`]
#bullet[H `clears` (narrow) → K `dominated` → L `a05_only` → I `fails` → J `band_like`]
#v(0.6em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  Recipe-specific, mandate-dependent, brittle. Does not reopen the 20 bp headline.
]

#pagebreak()

// ---------------- 10. conclusion ----------------
#heading[Conclusion]

#v(0.6em)
#bullet[Replications hold: Sepp Table 2, Monte Carlo cross-check, frictionless delta recovery.]
#v(0.5em)
#bullet[At 20 bp a re-tuned classical band does not lose on Sharpe; deep is not a uniform upgrade.]
#v(0.5em)
#bullet[High-cost Sharpe gains track under-hedging and premium collection — report them with P&L spread and tail risk.]

#v(1.0em)
#text(size: 22pt)[#link("https://github.com/hluaguo/dsc4107")[github.com/hluaguo/dsc4107]]
#v(0.4em)
#small[Full tables: `submission/report/report.pdf` and root `REPORT.md`.]
