# RQ-G: cost stress (10 / 20 / 50 bp)

## Research question

When proportional transaction cost is stressed to **k ∈ {0.001, 0.002, 0.005}** (10, 20,
50 bp), does unconstrained deep hedging (`Hyper()` defaults, all `torch_seeds`) remain
**weakly dominated** by the classical **cost–CVaR hull** re-evaluated at that same **k**?

The frozen benchmark cell is otherwise unchanged: Heston ATM one-month, shared validation
(8 `val_seeds` × 40_000 paths), positions scored through the same `hedge.rollout`
ledger for classical and deep policies.

## Classical hull at each cost (no pinned JSON reuse)

For **each** k we:

1. Sweep the RQ-A grid only — calendar `f ∈ {1,3,7,21}`, `delta_band ∈ {0.05,0.10,0.20}`,
   `price_band ∈ {0.01,0.02,0.05}` — with `BenchConfig.cost = k`.
2. Score every grid point on the shared validation pool (mean path cost, CVaR95, Sharpe).
3. Label the **champion** as the grid point with highest **mean validation Sharpe** at that k
   (audit field `champion_from_val_sharpe_sweep`).

We **never** load `bench/classical_best.json` into the hull at k ≠ 0.002. At k = 0.002 the
swept champion may coincide with the frozen band δ=0.10 baseline (`champion_matches_frozen_cb_at_20bp`);
that is an audit check, not an import of the JSON row.

Objectives for Pareto dominance (lower is better on both):

- **mean_cost** — mean path proportional transaction cost from `hedge.rollout`
- **cvar95** — 5% tail loss on terminal P&L (`hedging/metrics.py`)

## Per-k verdict (mirrors RQ-A)

Same rules as `experiments/rq_a/run_pareto.py` `acceptance_verdict`, with the per-k
**champion** replacing RQ-A’s pinned `classical_best` for the `improves` test:

| Verdict | Criterion |
| --- | --- |
| `dominated` | Every deep seed and the ensemble is weakly dominated by at least one classical grid point at this k. |
| `improves` | ≥2/3 seeds beat this k’s champion on CVaR95 with mean cost ≤ champion mean cost. |
| `flat` | Otherwise (includes cases where some deep points sit outside the hull but no seed clears the champion on both axes). |

Additional audit flag in JSON: `weakly_dominated_all_deep` — explicit count of the hull
domination rule above.

## Overall verdict

| Overall | Criterion |
| --- | --- |
| `dominated_all_k` | Per-k verdict is `dominated` for every k ∈ {0.001, 0.002, 0.005}. |
| `clears_some_k` | Per-k verdict is `improves` for at least one k. |
| `mixed` | Neither of the above (e.g. domination at low k but `flat` at 50 bp). |

## How to run

```bash
uv run python scripts/rq_cost_stress.py              # full cell → experiments/rq_cost_stress/results/cost_stress.json
uv run python scripts/rq_cost_stress.py --quick      # smoke only (do not cite as headline)
```

Implementation: `experiments/rq_cost_stress/run_cost_stress.py` (CLI shim: `scripts/rq_cost_stress.py`).

## Full-run result (2026-10-30)

**Overall verdict: `mixed`**

Wall time ~302 s on this cloud CPU (no CUDA). Artifacts: `results/cost_stress.json`, `results/cost_stress.png`.

| k (bp) | Per-k verdict | Champion (val Sharpe) | All deep weakly dominated? |
| --- | --- | --- | --- |
| 0.001 (10) | `dominated` | band δ=0.05 | yes |
| 0.002 (20) | `dominated` | band δ=0.10 (matches frozen CB spec) | yes |
| 0.005 (50) | `flat` | time f=21 | **no** |

At 10 and 20 bp, deep hedgers trade more cheaply than the Sharpe champion but sit **above**
the classical hull on CVaR95 — the same cost–tail pattern as RQ-A on the 20 bp cell. At 50 bp
the learned policies sit at **lower mean cost** and **higher tail loss** than any single grid
point simultaneously, so no classical point weakly dominates the deep scatter (`flat`, not
`improves`). Domination therefore **does not** survive at every cost level; it **does** hold
at 10 bp and 20 bp under the RQ-A weak-dominance rule.

### k=0.001 (10 bp) — per-k verdict `dominated`

Champion: **band_delta_0.05** | mean_cost 0.2127 | cvar95 0.7367

| label | mean_cost | cvar95 |
| --- | ---: | ---: |
| time_f1 | 0.2301 | 0.7232 |
| time_f3 | 0.1709 | 1.3130 |
| time_f7 | 0.1381 | 2.1260 |
| time_f21 | 0.1029 | 4.0186 |
| band_delta_0.05 | 0.2127 | 0.7367 |
| band_delta_0.10 | 0.1888 | 0.8441 |
| band_delta_0.20 | 0.1608 | 1.2056 |
| band_price_0.01 | 0.2013 | 0.7454 |
| band_price_0.02 | 0.1696 | 0.9148 |
| band_price_0.05 | 0.1310 | 1.5905 |

| deep | mean_cost | cvar95 | dominated? |
| --- | ---: | ---: | --- |
| deep_seed_0 | 0.2028 | 0.9367 | yes |
| deep_seed_1 | 0.2049 | 0.9500 | yes |
| deep_seed_2 | 0.2108 | 1.0576 | yes |
| deep_ensemble | 0.2061 | 0.9743 | yes |

### k=0.002 (20 bp) — per-k verdict `dominated`

Champion: **band_delta_0.10** | mean_cost 0.3776 | cvar95 1.0661 (same spec as `classical_best.json`)

| label | mean_cost | cvar95 |
| --- | ---: | ---: |
| time_f1 | 0.4602 | 0.9989 |
| time_f3 | 0.3418 | 1.5049 |
| time_f7 | 0.2762 | 2.2706 |
| time_f21 | 0.2058 | 4.1188 |
| band_delta_0.05 | 0.4255 | 0.9936 |
| band_delta_0.10 | 0.3776 | 1.0661 |
| band_delta_0.20 | 0.3215 | 1.3867 |
| band_price_0.01 | 0.4026 | 0.9928 |
| band_price_0.02 | 0.3391 | 1.1282 |
| band_price_0.05 | 0.2621 | 1.7514 |

| deep | mean_cost | cvar95 | dominated? |
| --- | ---: | ---: | --- |
| deep_seed_0 | 0.3536 | 1.3491 | yes |
| deep_seed_1 | 0.3555 | 1.2543 | yes |
| deep_seed_2 | 0.3692 | 1.3750 | yes |
| deep_ensemble | 0.3595 | 1.2912 | yes |

### k=0.005 (50 bp) — per-k verdict `flat`

Champion: **time_f21** | mean_cost 0.5144 | cvar95 4.4195

| label | mean_cost | cvar95 |
| --- | ---: | ---: |
| time_f1 | 1.1504 | 1.8978 |
| time_f3 | 0.8545 | 2.1191 |
| time_f7 | 0.6906 | 2.7254 |
| time_f21 | 0.5144 | 4.4195 |
| band_delta_0.05 | 1.0637 | 1.8414 |
| band_delta_0.10 | 0.9440 | 1.8104 |
| band_delta_0.20 | 0.8039 | 1.9856 |
| band_price_0.01 | 1.0066 | 1.8040 |
| band_price_0.02 | 0.8478 | 1.8258 |
| band_price_0.05 | 0.6552 | 2.2808 |

| deep | mean_cost | cvar95 | dominated? |
| --- | ---: | ---: | --- |
| deep_seed_0 | 0.4231 | 4.6439 | no |
| deep_seed_1 | 0.4154 | 4.6717 | no |
| deep_seed_2 | 0.4674 | 4.5157 | no |
| deep_ensemble | 0.4353 | 4.6087 | no |

Example: `deep_seed_1` (cost 0.415, CVaR 4.672) is not dominated by `time_f21` (lower CVaR
but higher cost) nor by band policies (lower CVaR at higher cost). No seed beats the 50 bp
champion on **both** axes, so the per-k label is `flat` rather than `improves`.

## Interpretation and citations

Discrete-time hedging with proportional costs goes back to **Leland (1985)** — widening
effective volatility as friction rises. **Sepp (2013)** supplies the Sharpe-optimal classical
band/time rules we span on the grid; **Ahn & Wilmott (2006)** motivate band triggers as a
practical response to transaction costs. Deep hedging (**Buehler et al.**) minimises a CVaR
training objective but is compared here on rollout cost and tail risk like RQ-A.

Under moderate costs (10–20 bp), the unconstrained network still looks like a **cheaper schedule
inside the hull with excess tail risk**. At 50 bp the coarser classical grid and the
cost–tail geometry leave a **non-dominated** region for deep policies relative to the fixed
grid — without claiming they beat the Sharpe champion on both objectives.
