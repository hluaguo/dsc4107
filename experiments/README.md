# Research experiments (frozen cells)

Each folder is one research question on the shared Heston ATM one-month benchmark
(unless noted). Full runs write JSON and figures under `results/` inside the folder.
Re-run from the repo root via the matching `scripts/rq_*.py` shim (see each
`NOTES.md`).

| ID | Folder | Verdict (full run) | One-line read |
| --- | --- | --- | --- |
| RQ-A | [`rq_a/`](rq_a/) | `dominated` | Unconstrained deep hedger sits above the classical cost–CVaR hull at **20 bp**. |
| RQ-B | [`rq_mandate/`](rq_mandate/) | `fail` | Activity floors do not produce a seed that clears CB on cost **and** tail. |
| RQ-D | [`rq_risk_matched/`](rq_risk_matched/) | `flat` | Training-α selection can match CB tail but not at lower mean cost. |
| RQ-E | [`rq_policy_anatomy/`](rq_policy_anatomy/) | `band_like` | Risk-matched nets that beat CB on tail look like band-style no-trade rules. |
| RQ-F | [`rq_hard_cvar/`](rq_hard_cvar/) | `fails` | Train-time Lagrange CVaR cap does not clear CB on both axes. |
| RQ-G | [`rq_cost_stress/`](rq_cost_stress/) | `mixed` | Domination at 10 & 20 bp; **50 bp** is `flat` (not hull-dominated). |
| RQ-H | [`rq_liquid_10bp/`](rq_liquid_10bp/) | `clears` | Soft footnote: at Sepp **k** under our full-rate ledger, RM/Lagrange can clear re-swept hull. |
| RQ-I | [`rq_liquid_robust/`](rq_liquid_robust/) | `fails` | Expanded seeds do not robustly reproduce RQ-H clears. |
| RQ-J | [`rq_liquid_anatomy/`](rq_liquid_anatomy/) | `band_like` | Liquid-cell anatomy among CVaR-better seeds. |
| RQ-K | [`rq_liquid_baseline/`](rq_liquid_baseline/) | `dominated` | Unconstrained `Hyper()` at 10 bp does not clear the champion. |

**Headline (20 bp benchmark):** classical band rules remain the honest cost–risk
reference; see [`REPORT.md`](../REPORT.md) and RQ-A. Liquid RQs (H–K) use
`cost = 0.001` with a re-swept classical hull — a softer, ledger-specific
footnote, not a reversal of the main story.

Artifacts cited in the report for replications and the historical 4×5 comparison
are mirrored under [`submission/data/`](../submission/data/). Regenerating those
runs writes to gitignored [`results/`](../results/) at the repo root.
