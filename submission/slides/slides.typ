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
    Sepp (2013) and Buehler et al. (2019) replications, plus a fair head-to-head
  ]
  #v(2.4cm)
  #text(size: 17pt)[DSC4107 Final Project]
  #v(0.3em)
  #text(size: 17pt)[Lau Cheuk Yin]
]

#pagebreak()

// ---------------- 2. hook ----------------
#heading[Expectation vs fair test]

#bullet[Deep hedging is sold as freer → should beat classical bands once costs bite.]
#bullet[Sepp (2013): closed-form Sharpe-optimal delta bands. Buehler et al. (2019): network on CVaR of terminal P&L.]
#bullet[Neither paper runs both on one simulator with a classical baseline *re-tuned per cell*.]

#v(0.8em)
#text(fill: accent, weight: "bold")[We run that comparison — shared rollout, four models, five costs, disjoint seeds, frictionless delta recovery as the correctness anchor.]

#pagebreak()

// ---------------- 3. methods ----------------
#heading[Fair test (light)]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *One testbed*
    #bullet[GBM, jump-diffusion, Heston, Heston+ jumps — Sepp parameters]
    #bullet[ATM one-month call; proportional cost on every trade]
    #bullet[Classical side re-tuned in every map cell]
  ],
  [
    *Controls*
    #bullet[Learned MLP: moneyness, time, position; CVaR $alpha = 0.65$]
    #bullet[Disjoint train / val / tuning / search / holdout pools]
    #bullet[Three torch seeds per reported mean]
  ],
)

#v(1.0em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  Frozen benchmark: Heston at 20 bp with 25% implied vs 20% realised vol (volatility risk premium).
]

#pagebreak()

// ---------------- 4. 20 bp headline (early) ----------------
#heading[20 bp: the headline]

#grid(
  columns: (1fr, 1.05fr),
  gutter: 1.0em,
  [
    #bullet[Learned − classical Sharpe edges *−0.09 to +0.01* — essentially flat]
    #bullet[Jump models: classical slightly ahead]
    #bullet[Holdout (5 seeds): mean edge *+0.0339* — real but tiny]
    #v(0.6em)
    #text(fill: accent, weight: "bold")[Classic hedging is all you need at nominal friction. Deep is not a uniform upgrade.]
  ],
  [
    #figure(
      image("../report/figures/edge_map_fixed.png", width: 100%),
      caption: [Learned minus classical Sharpe (fixed hyperparameters).],
    )
  ],
)

#pagebreak()

// ---------------- 5. replications ----------------
#heading[Replication checks (credibility)]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.0em,
  [
    *Sepp*
    #bullet[Table 2: largest $u$ error 3.9%; case I GBM/SVJ to 0.0001]
    #bullet[JD fix: restore $q$; analytic Sharpe 0.65 matches MC]
    #bullet[60,000-path cross-check on mean and vol]
  ],
  [
    *Deep hedging*
    #bullet[Zero cost: corr with BS delta 0.9907]
    #bullet[Heston qual. claim reproduced — beats every-step delta as cost rises]
    #small[Paper table is not the headline; the 20 bp map is.]
  ],
)

#pagebreak()

// ---------------- 6. tail + harvesting ----------------
#heading[Same spread, worse tail — and the high-cost twist]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.0em,
  [
    #bullet[Benchmark cell: mean 0.240 vs 0.216; std 0.626 vs 0.613]
    #bullet[Skew −0.70 vs −0.01; CVaR 95% *24% worse*]
    #v(0.5em)
    #bullet[*Above ~50 bp:* Sharpe “wins” with std → 3–4, mean → premium]
    #bullet[Paired spread and tail — *premium harvesting*, not better hedges]
  ],
  [
    #figure(
      image("../report/figures/distribution.png", width: 100%),
      caption: [Terminal P&L on the frozen benchmark.],
    )
  ],
)

#pagebreak()

// ---------------- 7. liquid footnote ----------------
#heading[Liquid 10 bp (footnote only)]

#bullet[Sepp Case I: *k/2* on $|Delta delta| dot S$ (eq.~2.5); side study at liquid-index *k* uses our *full-rate* ledger (`cost = 0.001` — numerically Sepp's *k*, not Case I economics)]
#bullet[10 bp classical hull re-swept; not `classical_best.json`]
#bullet[H *clears* (narrow) → K *dominated* → L *a05_only* → I *fails* → J *band_like*]
#v(0.6em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt)[
  Recipe-specific, mandate-dependent, brittle. Does not reopen the 20 bp headline.
]

#pagebreak()

// ---------------- 8. conclusion ----------------
#heading[Conclusion]

#v(0.6em)
#bullet[At 20 bp a re-tuned classical band does not lose on Sharpe; any tiny learned edge trades tail for mean.]
#v(0.5em)
#bullet[High-cost Sharpe gains track under-hedging — report them with P&L std and CVaR.]
#v(0.5em)
#bullet[Replications hold: Sepp Table 2, Monte Carlo cross-check, frictionless delta recovery.]

#v(1.0em)
#text(size: 22pt)[#link("https://github.com/hluaguo/dsc4107")[github.com/hluaguo/dsc4107]]
#v(0.4em)
#small[Full tables: `submission/report/report.pdf`.]
