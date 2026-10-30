# RQ-A: cost–CVaR Pareto on the frozen cell

## Research question

On the frozen Heston ATM one-month benchmark (20 bp proportional cost), is the
unconstrained deep hedger (`Hyper()` defaults, all `torch_seeds`) weakly dominated
by the classical **cost–risk** frontier spanned by calendar delta hedging and band
rules—including the pinned `classical_best` band champion?

Objectives (lower is better on both):

- **mean_cost** — mean path proportional transaction cost from `hedge.rollout`
- **cvar95** — 5% tail loss on terminal P&L (`hedging/metrics.py`)

## Cell and acceptance

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` — Heston, ATM call, `cost=0.002`, `n_steps=21` |
| Validation | 8 `val_seeds` × 40_000 paths (shared across all policies) |
| Classical grid | time `f ∈ {1,3,7,21}`, `delta_band ∈ {0.05,0.10,0.20}`, `price_band ∈ {0.01,0.02,0.05}`, plus `classical_best` |
| Learned | `train_ensemble` with default `Hyper()` — 200 epochs, 3 torch seeds + ensemble average |
| Scoring | Positions replayed through the same `hedge.rollout` ledger for classical and deep |

| Verdict | Criterion |
| --- | --- |
| `dominated` | Every deep seed and the ensemble is weakly dominated by at least one classical point (no point strictly northwest of the classical scatter). |
| `improves` | ≥2/3 seeds beat `classical_best` on CVaR95 with mean cost ≤ baseline. |
| `flat` | Otherwise. |

## How to run

From repo root (GPU used when available; this full run used CPU):

```bash
uv run python scripts/rq_a_pareto.py              # full cell → experiments/rq_a/results/pareto.json
uv run python scripts/rq_a/pareto.py --quick      # smoke only (do not cite as headline)
```

Implementation: `experiments/rq_a/run_pareto.py` (CLI shim: `scripts/rq_a_pareto.py`).

## Full-run result (2026-10-30)

**Automated verdict: `dominated`**

Wall time ~92 s on this cloud CPU (no CUDA). Artifacts: `results/pareto.json`, `results/pareto.png`.

### Pinned baseline vs deep

| Policy | mean_cost | cvar95 |
| --- | ---: | ---: |
| **classical_best** (band δ=0.10) | 0.3776 | 1.0661 |
| deep_seed_0 | 0.3536 | 1.3491 |
| deep_seed_1 | 0.3555 | 1.2543 |
| deep_seed_2 | 0.3692 | 1.3750 |
| deep_ensemble | 0.3595 | 1.2912 |

### Classical frontier (full grid)

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
| classical_best | 0.3776 | 1.0661 |

### Beats classical on **both** axes?

**No.** Every deep point has lower mean cost than `classical_best` but **higher** CVaR95.
Each deep seed is weakly dominated by at least one classical grid point (e.g. `deep_seed_0`
by `band_price_0.02`: cost 0.339 ≤ 0.354 and CVaR 1.128 ≤ 1.349). None of the three
seeds satisfy the `improves` rule (CVaR below baseline at no higher cost).

## Interpretation

The unconstrained deep hedger learns a **cheaper** trading schedule than the pinned
band baseline but pays for it in **tail risk**: CVaR95 stays above the classical hull
that already mixes calendar and band policies. For this cell, Sepp-style classical
rules remain the relevant cost–risk reference; the deep recipe does not clear the
Pareto hull on validation, hence `dominated`.

Training still minimises a CVaR-based objective; the gap is partly **policy class**
(unconstrained network vs discrete band/time rules) and partly that the classical
frontier explicitly spans cost–tail trade-offs the single learned policy does not
match simultaneously.
