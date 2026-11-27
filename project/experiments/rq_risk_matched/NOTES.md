# RQ-D: risk-matched deep vs classical on the frozen cell

## Research question

Can deep hedging hit **OOS CVaR95 ≤ `classical_best`** at strictly lower mean proportional
cost, or is every training-α / post-selection point nested in the RQ-A classical cost–CVaR
hull?

RQ-A trained the default `Hyper()` recipe (α = 0.65) and found cheaper mean cost but
**worse** tail risk than the pinned band baseline (~CVaR95 1.066 on the shared OOS).
RQ-D repeats the same validation ledger and classical frontier, but trains a grid of
networks while varying **training-α** in `deep.cvar_loss` — a **risk dial proxy**, not a
hard constraint that OOS CVaR95 must stay at or below the baseline during optimisation.
After each net is fit, every candidate is scored on the shared OOS paths via
`hedge.rollout`; **selection** and **clears** / `improves` use only those rollout numbers.

## Cell and design

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` — Heston ATM call, `cost=0.002`, `n_steps=21` |
| Validation | Same as RQ-A: all `val_seeds` × `n_val_paths` on identical paths |
| Classical | Reuses RQ-A grid + `classical_best` via `hedge.rollout` (imported helpers) |
| Training | Default `Hyper()` architecture/budget; **training-α ∈ {0.05, 0.12, …, 0.65}** full cell (smoke: two values) — dial only, no CVaR95 cap in the loss |
| OOS selection (per torch seed) | One net per training-α; score each on pooled validation rollout; **select** min mean cost among OOS points with CVaR95 ≤ `classical_best`, else lowest OOS CVaR95 |
| Ensemble | Pathwise average of the three selected nets (same mixing as RQ-A) |
| Axes | **mean_cost** and **cvar95** only from `hedge.rollout` (deep positions via `deep.hedge_positions`) |

Reference tail level: `classical_best.cvar95` from the frozen cell / RQ-A full run (~1.066).
That number guides **post-hoc selection** on OOS rollout metrics; the trainer never
enforces CVaR95 ≤ baseline (and this experiment does not add such a constraint). See
`scripts/explore_objective.py` for the same training-α knob elsewhere in the repo.

## Acceptance / verdict

All criteria are evaluated on **OOS rollout** mean cost and CVaR95 **after** the α grid
and per-seed selection — not on training loss or in-sample paths.

| Verdict | Criterion |
| --- | --- |
| `improves` (“clears”) | **Any** selected seed has OOS CVaR95 ≤ `classical_best` **and** OOS mean_cost **<** `classical_best` |
| `nested` | Every sweep, selected, and ensemble OOS point is weakly dominated by at least one classical grid point (lower is better on both axes) |
| `flat` | Otherwise |

## Cite-or-die (why this question)

- **Sepp (2013)** — discrete hedging with proportional costs; the classical hull here is
  the empirical analogue of his frequency/band Sharpe surface, but RQ-D scores **cost
  and CVaR** like a tail-aware seller rather than re-opening the Sharpe leaderboard.
- **Leland (1985)** — cost-adjusted discrete delta hedging; motivates why the baseline
  is a **band** rule at 20 bp, not a frictionless continuous hedge.
- **Ahn & Wilmott (2009)** — risk-matched / cost-aware discrete hedging framing; RQ-D
  asks whether a learned policy can **clear** the same **OOS tail budget** as the
  hand-built champion without paying more turnover — tested only after rollout scoring.
- **Buehler et al. (2019)** — deep hedging with convex risk objectives; the training-α
  sweep stays inside their CVaR-loss family as a dial, not as a guaranteed tail match.

(Li 2025 classical frequency sweeps are adjacent literature only; no deep comparator.)

## How to run

From repo root:

```bash
uv run python scripts/rq_risk_matched.py --quick   # CI / dev smoke → results/risk_matched_quick.json
uv run python scripts/rq_risk_matched.py           # weekend full cell → results/risk_matched.json
```

Implementation: `experiments/rq_risk_matched/run_risk_matched.py` (CLI shim:
`scripts/rq_risk_matched.py`).

## Results

### Full run (2026-10-30)

**Automated verdict: `flat`**

Wall time **618.9 s** (~10.3 min) on CPU (no CUDA). Artifacts: `results/risk_matched.json`,
`results/risk_matched.png`.

| Policy | mean_cost (OOS) | cvar95 (OOS) | training-α chosen |
| --- | ---: | ---: | ---: |
| **classical_best** | 0.3776 | 1.0661 | — |
| deep_seed_0_selected | 0.4020 | 1.0131 | 0.30 |
| deep_seed_1_selected | 0.3939 | 1.0553 | 0.40 |
| deep_seed_2_selected | 0.4052 | 0.9693 | 0.20 |
| deep_ensemble_selected | 0.4004 | 1.0071 | (0.30, 0.40, 0.20) |

**Clears baseline (`improves`)?** No — three OOS-selected nets meet CVaR95 ≤ `classical_best`,
but none have mean_cost **<** 0.3776, so the Lead **clears** rule fails. Several low training-α
**sweep** points beat the baseline on OOS CVaR yet sit at **higher** OOS cost, so the scatter is
not fully `nested`; verdict **`flat`**.

### Smoke run (`--quick`) — not headline evidence

Verdict **`nested`** on shrunk pools (5 epochs, one seed). See `results/risk_matched_quick.json`.
