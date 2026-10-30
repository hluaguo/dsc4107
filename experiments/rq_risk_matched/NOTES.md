# RQ-D: risk-matched deep vs classical on the frozen cell

## Research question

Can deep hedging hit **CVaR95 ≤ `classical_best`** at weakly lower mean proportional
cost, or is every multi-λ / risk-matched point nested in the RQ-A classical cost–CVaR
hull?

RQ-A trained the default `Hyper()` recipe (α = 0.65) and found cheaper mean cost but
**worse** tail risk than the pinned band baseline (~CVaR95 1.066 on the shared OOS).
This experiment keeps the same validation ledger and classical frontier, but sweeps the
CVaR-tail fraction **α** in `deep.cvar_loss` per torch seed and **selects** the trained
policy that best matches the classical tail target before scoring the headline seeds.

## Cell and design

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` — Heston ATM call, `cost=0.002`, `n_steps=21` |
| Validation | Same as RQ-A: all `val_seeds` × `n_val_paths` on identical paths |
| Classical | Reuses RQ-A grid + `classical_best` via `hedge.rollout` (imported helpers) |
| Learned | Default `Hyper()` architecture/training budget; **α ∈ {0.05, 0.12, …, 0.65}** full cell (smoke: two α values) |
| Per seed | Train one net per α; score each on pooled validation rollout; **select** min mean cost among points with CVaR95 ≤ baseline, else lowest CVaR95 |
| Ensemble | Pathwise average of the three selected nets (same mixing as RQ-A) |
| Axes | **mean_cost** and **cvar95** only from `hedge.rollout` (deep positions via `deep.hedge_positions`) |

Soft target: `classical_best.cvar95` from the frozen cell / RQ-A full run (~1.066).
The α sweep is the codebase-native risk knob (see `scripts/explore_objective.py`);
we do not introduce a parallel trainer or a second torch cost ledger.

## Acceptance / verdict

| Verdict | Criterion |
| --- | --- |
| `improves` (“clears”) | **Any** selected seed has CVaR95 ≤ `classical_best` **and** mean_cost **<** `classical_best` (strictly cheaper at matched-or-better tail) |
| `nested` | Every sweep, selected, and ensemble point is weakly dominated by at least one classical grid point (lower is better on both axes) |
| `flat` | Otherwise |

## Cite-or-die (why this question)

- **Sepp (2013)** — discrete hedging with proportional costs; the classical hull here is
  the empirical analogue of his frequency/band Sharpe surface, but RQ-D scores **cost
  and CVaR** like a tail-aware seller rather than re-opening the Sharpe leaderboard.
- **Leland (1985)** — cost-adjusted discrete delta hedging; motivates why the baseline
  is a **band** rule at 20 bp, not a frictionless continuous hedge.
- **Ahn & Wilmott (2009)** — risk-matched / cost-aware discrete hedging framing; RQ-D
  asks whether a learned policy can meet the same **tail budget** as the hand-built
  champion without paying more turnover.
- **Buehler et al. (2019)** — deep hedging with convex risk objectives; the α sweep
  stays inside their CVaR-loss family instead of unconstrained mean-cost chasing.

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

| Policy | mean_cost | cvar95 | selected α |
| --- | ---: | ---: | ---: |
| **classical_best** | 0.3776 | 1.0661 | — |
| deep_seed_0_selected | 0.4020 | 1.0131 | 0.30 |
| deep_seed_1_selected | 0.3939 | 1.0553 | 0.40 |
| deep_seed_2_selected | 0.4052 | 0.9693 | 0.20 |
| deep_ensemble_selected | 0.4004 | 1.0071 | (0.30, 0.40, 0.20) |

**Clears baseline (`improves`)?** No — every selected seed reaches CVaR95 ≤ `classical_best`
on at least one α in the sweep, but the risk-matched pick trades **higher** mean cost for
that tail (none have mean_cost **<** 0.3776). Several low-α sweep points sit **below** the
baseline on CVaR yet above it on cost (northwest of the pinned point), so the hull is not
`nested`; the run is **`flat`**.

### Smoke run (`--quick`) — not headline evidence

Verdict **`nested`** on shrunk pools (5 epochs, one seed). See `results/risk_matched_quick.json`.
