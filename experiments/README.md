# Research experiments

One folder per question on the shared Heston ATM one-month simulator unless noted.
Committed full runs: `results/*.json` and `*.png` inside each folder. Re-run via
`scripts/rq_*.py` (see each `NOTES.md`).

## Main line (20 bp frozen cell, `cost = 0.002`)

| ID | Folder | Verdict | One-line read |
| --- | --- | --- | --- |
| RQ-A | [`rq_a/`](rq_a/) | `dominated` | Deep is cheaper on mean cost but **above** the classical cost–CVaR hull on tail. |
| RQ-B | [`rq_mandate/`](rq_mandate/) | `fail` | Activity floors do not yield a seed that beats CB on **both** cost and CVaR95. |
| RQ-D | [`rq_risk_matched/`](rq_risk_matched/) | `flat` | Post-hoc α selection can match CB tail, not at lower mean cost. |
| RQ-E | [`rq_policy_anatomy/`](rq_policy_anatomy/) | `band_like` | Risk-matched nets that beat CB on tail look like band / no-trade rules. |
| RQ-F | [`rq_hard_cvar/`](rq_hard_cvar/) | `fails` | Train-time Lagrange CVaR cap does not clear CB on both axes. |
| RQ-G | [`rq_cost_stress/`](rq_cost_stress/) | `mixed` | Hull domination at 10 & 20 bp; at **50 bp** the cell is `flat` (not a deep win story). |

Headline: [`REPORT.md`](../REPORT.md) + RQ-A. Do not oversell 50 bp or high-cost Sharpe edges
from premium harvesting (see report §3).

## Liquid 10 bp footnote only (`cost = 0.001`)

Sepp Case I uses **k/2** on |Δδ|·S; our ledger is **full-rate** `cost · |trade| · S`.
Setting `cost = 0.001` aligns with Sepp liquid-index **k**, not exact Case I friction.
Classical hull is **re-swept** at 10 bp — never `classical_best.json`.

**Brittle soft path (footnote, not main title):**

| Step | ID | Verdict | Read |
| --- | --- | --- | --- |
| 1 | **RQ-H** [`rq_liquid_10bp/`](rq_liquid_10bp/) | `clears` | Narrow RM/Lagrange clear vs re-swept champion. |
| 2 | **RQ-K** [`rq_liquid_baseline/`](rq_liquid_baseline/) | `dominated` | Default `Hyper()` does **not** clear. |
| 3 | **RQ-L** [`rq_liquid_alpha/`](rq_liquid_alpha/) | `a05_only` | Only selected α=0.05 seeds clear; not a multi-α triumph. |
| 4 | **RQ-I** [`rq_liquid_robust/`](rq_liquid_robust/) | `fails` | Expanded seeds do **not** robustly reproduce H. |
| 5 | **RQ-J** [`rq_liquid_anatomy/`](rq_liquid_anatomy/) | `band_like` | Clearing recipes look band-like, not a new mandate. |

Replication artifacts for the written report are also under
[`submission/data/`](../submission/data/). Regenerating Sepp/comparison runs writes to
gitignored [`results/`](../results/) at repo root.
