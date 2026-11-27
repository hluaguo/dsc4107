// Final presentation slides. 16:9, compiled with Typst. Target: 10 minutes.
// Build: typst compile --root . slides/slides.typ slides/project_slide_lau_cheuk_yin_58525058.pdf

#set page(
  width: 33.87cm,
  height: 19.05cm,
  margin: 1.5cm,
  numbering: none,
  footer: context {
    let n = counter(page).get().first()
    if n > 1 {
      set text(size: 12pt, fill: luma(55%))
      [Classic hedging is all you need #h(1fr) #n]
    }
  },
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
#set figure(numbering: none)

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
#let takeaway(body) = block(fill: luma(96%), inset: 12pt, radius: 3pt, width: 100%)[
  #text(fill: accent, weight: "bold")[#body]
]
#let tbl(..args) = {
  set text(size: 16pt)
  table(stroke: 0.4pt + luma(75%), inset: 7pt, ..args)
}

// ---------------- 1. title ----------------
#align(center)[
  #v(3.0cm)
  #text(size: 40pt, weight: "bold")[Classic hedging is all you need]
  #v(0.6em)
  #text(size: 24pt)[Classical rules versus neural networks for option hedging]
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

// ---------------- 2. the problem ----------------
#heading[The hedging problem]

#grid(
  columns: (1fr, 1.15fr),
  gutter: 1.0em,
  [
    #bullet[A desk sells a one-month at-the-money call and hedges with the stock.]
    #bullet[Every trade pays a proportional cost.]
    #bullet[Rebalance often → tracks delta, but pays more cost.]
    #bullet[Rebalance rarely → cheap, but leaves risk.]
    #v(0.4em)
    #small[Sepp: with costs, the Sharpe-optimal frequency is interior — and lower in richer models.]
  ],
  [
    #figure(
      image("../report/figures/sepp_sharpe_caseI.png", width: 100%),
      caption: [Sepp case I: Sharpe against hedging frequency.],
    )
  ],
)

#pagebreak()

// ---------------- 3. hook ----------------
#heading[Two papers, opposite ends]

#bullet[*Sepp (2013):* closed-form Sharpe-optimal delta hedging → frequency and band rules.]
#bullet[*Buehler et al. (2019):* a neural network trained on CVaR of terminal P&L — free to trade less or hedge partially.]
#bullet[Usual reading: the network is strictly freer, so it should win once costs bite.]
#bullet[Neither paper runs both on one simulator with the classical side *re-tuned per setting*.]

#v(0.8em)
#takeaway[I run that comparison: shared P&L rollout, four market models, five cost levels, disjoint seeds.]

#pagebreak()

// ---------------- 4. simulation framework ----------------
#heading[Simulation framework]

#grid(
  columns: (1.1fr, 1fr),
  gutter: 1.4em,
  [
    *One simulated month*
    #bullet[Stock starts at \$100; the bank sells the option for \$2.88]
    #bullet[Each of 21 trading days: the strategy picks how many shares to hold and pays 0.2% on every trade]
    #bullet[Day 21: sell leftover shares, pay the client what the option owes]
    #bullet[Money left over = profit or loss for that month]
  ],
  [
    *Repeated many times*
    #bullet[4 market models: smooth, sudden jumps, moving volatility, both]
    #bullet[Thousands of price paths per model]
    #bullet[Same accounting code for every strategy]
    #bullet[Separate data for training, tuning and the final test]
    #bullet[Classical rules (Sepp band or time schedule) re-tuned for every model and fee]
  ],
)

#v(0.6em)
#block(fill: luma(96%), inset: 12pt, radius: 3pt, width: 100%)[
  The option is priced at 25% volatility; the stock really moves about 20%. Like an insurer, the bank has a small built-in margin — so the question is how much protection is worth paying for.
]

#pagebreak()

// ---------------- 5. decision rule ----------------
#heading[Why Sharpe is not enough]

#bullet[Sharpe rewards mean over spread. A policy that *stops hedging* and keeps the premium can post a high Sharpe.]
#bullet[A hedging desk cares about two things:]
#pad(left: 1.2cm)[
  #bullet[*mean cost* — average transaction cost paid per path]
  #bullet[*CVaR95* — average loss in the worst 5% of paths]
]

#v(0.6em)
#takeaway[Decision rule: the neural network wins only if it beats the tuned band on *both* mean cost and CVaR95.]
#v(0.3em)
#small[Sharpe is reported throughout, but it does not settle the question by itself.]

#pagebreak()

// ---------------- 6. Sepp replication ----------------
#heading[Replication 1: Sepp (2013)]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #set table(inset: 9pt)
    #show table: set text(size: 18pt)
    #tbl(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        [], table.cell(colspan: 2, align: center)[*Best trades / year*], table.cell(colspan: 2, align: center)[*Sharpe there*],
        [*Model*], [*Mine*], [*Paper*], [*Mine*], [*Paper*],
      ),
      [Smooth], [242], [239], [0.83], [0.89],
      [Jumps], [147], [138], [0.65], [0.66],
      [Moving vol], [75], [74], [0.43], [0.45],
      [Both], [70], [68], [0.43], [0.45],
    )
  ],
  [
    #v(0.4em)
    #bullet[My numbers closely match the paper's]
    #bullet[60,000 simulated paths agree with the formulas]
    #bullet[Every curve has a sweet spot — and richer markets favour trading less]
    #v(0.6em)
    #takeaway[The classical side is trustworthy.]
  ],
)

#pagebreak()

// ---------------- 7. deep replication ----------------
#heading[Replication 2: deep hedging (Buehler et al.)]

#grid(
  columns: (1fr, 1.1fr),
  gutter: 1.0em,
  [
    #bullet[*Zero fee:* the network learns the textbook (Black-Scholes) hedge on its own — 0.99 correlation]
    #bullet[Sharpe 1.15 vs 1.14 for the textbook hedge]
    #bullet[*With fees:* beats trading every day (20 bp: 0.37 vs 0.25)]
    #v(0.4em)
    #takeaway[The paper's claim reproduces. But does it beat a *properly tuned* classical band?]
  ],
  [
    #figure(
      image("figures/sharpe_vs_cost.png", width: 100%),
      caption: [Heston model: Sharpe as the trading fee rises.],
    )
  ],
)

#pagebreak()

// ---------------- 8. 20 bp headline ----------------
#heading[20 bp: the headline]

#grid(
  columns: (0.75fr, 1.25fr),
  gutter: 1.0em,
  [
    #bullet[At 20 bp (green), the two lines sit on top of each other — *a tie*]
    #bullet[In the jump models, classical is slightly ahead]
    #bullet[Final test on 5 fresh batches of data: network ahead by *+0.034* every time — real but tiny]
    #v(0.6em)
    #text(fill: accent, weight: "bold")[Small Sharpe edge on Heston. Does it pass the decision rule?]
  ],
  [
    #image("figures/edge_map.png", width: 100%)
  ],
)

#pagebreak()

// ---------------- 9. tail ----------------
#heading[Same spread, worse tail]

#grid(
  columns: (0.85fr, 1.15fr),
  gutter: 1.2em,
  [
    #v(0.4em)
    #bullet[*Spread:* almost the same — on a normal day they look alike]
    #bullet[*Worst 5% of outcomes:* the network loses *\$1.32* per option, the band *\$1.07* — *24% worse*]
    #bullet[Worse on every one of the 5 final test batches]
    #v(0.3em)
    #takeaway[The small Sharpe edge is paid for with worse bad days.]
  ],
  [
    #image("figures/tail.png", width: 100%)
  ],
)

#pagebreak()

// ---------------- 10. frontier ----------------
#heading[Can a safer-trained network beat the band?]

#grid(
  columns: (0.75fr, 1.25fr),
  gutter: 1.2em,
  [
    #v(0.4em)
    I retrained the network to care more about bad outcomes.
    #v(0.8em)
    #bullet[*Still no win:* nothing is cheaper _and_ safer than the band (green corner is empty).]
    #v(0.4em)
    #bullet[*One niche:* some networks are safer than any classical rule — for about 8% more cost.]
  ],
  [
    #image("figures/frontier.png", width: 100%)
  ],
)

#pagebreak()

// ---------------- 11. high-cost twist ----------------
#heading[The high-cost twist: it stops hedging]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    #tbl(
      columns: 5,
      align: (left, right, right, right, right),
      table.header(
        [], table.cell(colspan: 2, align: center)[*Sharpe*], table.cell(colspan: 2, align: center)[*Spread (\$)*],
        [*Fee*], [*Network*], [*Classical*], [*Network*], [*Classical*],
      ),
      [0 bp], [1.16], [1.14], [0.51], [0.52],
      [20 bp], [0.37], [0.35], [0.61], [0.61],
      [50 bp], [0.10], [0.05], [1.60], [1.74],
      [100 bp], [0.18], [−0.25], [3.11], [1.73],
      [200 bp], [0.17], [−0.85], [3.17], [1.73],
    )
    #v(0.3em)
    #small[Moving-volatility (Heston) model. Spread = standard deviation of profit, \$ per option.]
  ],
  [
    #bullet[At 100–200 bp (1–2% per trade), classical Sharpe goes negative; the network's stays positive]
    #bullet[…but the network's spread is *nearly double*]
    #bullet[It has learned to *stop hedging* and just keep the \$2.88 option fee]
  ],
)

#v(0.6em)
#takeaway[A Sharpe win from not hedging is a change of product, not a better hedge.]

#pagebreak()

// ---------------- 12. caveats ----------------
#heading[Caveats]

#grid(
  columns: (1fr, 1fr),
  gutter: 1.2em,
  [
    *Limited compute*
    #bullet[Limited computing resources: 80–100 epochs (200 for the holdout), 2,048 paths per seed, three seeds, small network]
    #bullet[Direction is stable across fixed and searched settings; exact edges are not paper-grade]
    #bullet[Holdout compares a 3-net ensemble against one rule — this favours the network]
  ],
  [
    *Liquid 10 bp side study*
    #bullet[Same test at 10 bp with a re-tuned band (Sepp's $k$ numerically; Sepp charges $k\/2$, I charge full rate)]
    #bullet[One recipe narrowly wins (about 0.4% cost margin), only at one $alpha$, on fewer than half the seeds]
    #small[Brittle; does not reopen the 20 bp headline.]
  ],
)

#pagebreak()

// ---------------- 13. conclusion ----------------
#heading[Conclusion]

#v(0.6em)
#bullet[At 20 bp the network's Sharpe edge is tiny (+0.034 on the final test) and costs a 24% worse tail; safer-trained networks cost more than the band. Nothing beats the Sharpe-selected band on both.]
#v(0.5em)
#bullet[Low-$alpha$ nets do reach tails no classical rule reaches (CVaR95 0.94 vs 0.99) — useful only if the desk wants that tail.]
#v(0.5em)
#bullet[High-cost Sharpe gains track under-hedging — report them with P&L std and CVaR.]
#v(0.5em)
#bullet[Replications hold: Sepp Table 2, Monte Carlo cross-check, frictionless delta recovery.]

#v(1.0em)
#text(size: 22pt)[#link("https://github.com/hluaguo/dsc4107")[github.com/hluaguo/dsc4107]]
#v(0.4em)
#small[Full tables: `submission/report/project_report_lau_cheuk_yin_58525058.pdf`.]
