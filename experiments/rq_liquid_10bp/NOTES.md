# RQ-H: liquid 10 bp (risk-matched + Lagrange vs re-swept hull)

## Research question

At **Sepp liquid-index k level** under our **full-rate** ledger (`BenchConfig.cost =
0.001`, 10 bp proportional on `hedge.rollout`), can **either** (1) an RQ-D-style
**training-α risk-matched sweep** or (2) an RQ-F-style **train-time Lagrange /
excess-CVaR** constraint (fixed `Hyper().alpha = 0.65`) produce **any** torch seed
that **clears** the **10 bp classical champion** on OOS rollout: `mean_cost` strictly
below the champion **and** `cvar95` ≤ champion?

## Cost convention (Sepp half-spread vs bench full rate)

Sepp (2013) eq (2.5) charges proportional costs with factor **k/2** on |Δδ|·S (half the quoted bid–ask; see also Leland 1985 / Toft 1996 as cited there). Sepp Table 1 / §5.1 Case I (liquid index) sets **k = 0.001**. Our `BenchConfig` / `hedge.rollout` charge the **full** proportional rate `cost · |trade| · S` with no ½. RQ-H sets `cost = 0.001` (10 bp full-rate) — numerically equal to Sepp’s Case I **k**, but **not** identical economics: Sepp’s effective one-way charge at that k is ~5 bp. Do not claim “exact Sepp Case I friction”; say “Sepp liquid-index **k** level under our full-rate ledger.” Buehler et al. (2019) eq (5.7) is also full-rate ε|n|S (their demo ε=0.01 ≫ 10 bp). Re-sweep classical hull at 0.001; never reuse `classical_best.json` (20 bp).

## Cell (only change vs frozen bench)

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` with **`cost = 0.001`** (10 bp full rate); Heston ATM 1M, `n_steps=21` |
| Validation | Shared OOS: `val_seeds` 2001–2008 × 40_000 paths |
| Deep training | Default `Hyper()` (all `torch_seeds`; not shrunk) |
| OOS scoring | Dual `hedge.rollout` ledger for classical and deep (same as RQ-A / D / F / G) |
| Classical hull | RQ-A / RQ-G grid only — calendar `f ∈ {1,3,7,21}`, `delta_band ∈ {0.05,0.10,0.20}`, `price_band ∈ {0.01,0.02,0.05}` at **k = 0.001** |
| Champion | Highest **mean validation Sharpe** on that grid at 10 bp (**not** `classical_best.json`) |

**Hard eval rule:** no arm may use `bench/classical_best.json` as the hull or as the
comparison baseline. Re-sweep only at k = 0.001. JSON audit:
`loaded_classical_best_json: false`. Frozen CB may appear only for optional
`champion_matches_frozen_cb_spec` audit (20 bp band δ=0.10); hull points are grid-only.

## Arms (two only)

| Arm | Mechanism | Headline rows |
| --- | --- | --- |
| **Risk-matched** | RQ-D: sweep training `alpha` in `deep.cvar_loss`; per seed select OOS point with min cost among `cvar95 ≤ champion`, else min CVaR | `rm_seed_*_selected` |
| **Lagrange / excess-CVaR** | RQ-F: fixed `Hyper().alpha`; dual ascent on λ · ReLU(batch_cvar95 − champion CVaR) | `lag_seed_*_cvar_capped` |

## Verdict (Lead locked)

Evaluated on **OOS rollout** mean cost and CVaR95 only.

| Verdict | Criterion |
| --- | --- |
| **`clears`** | **Any** torch seed (**either arm**) has OOS `mean_cost` **<** 10 bp champion **and** OOS `cvar95` ≤ champion |
| **`fails`** | Otherwise |

**Soft note (`interesting` in JSON):** true only if **`clears`** **or** some seed has
`cvar95` ≤ champion with `mean_cost` within **1%** of champion mean cost.

Do **not** claim exact Sepp Case I economics anywhere in PR/NOTES — use “Sepp
liquid-index **k** level under our full-rate ledger.”

## How to run

```bash
uv run python scripts/rq_liquid_10bp.py --quick   # smoke → results/liquid_10bp_quick.json
uv run python scripts/rq_liquid_10bp.py           # full cell → results/liquid_10bp.json
```

Implementation: `experiments/rq_liquid_10bp/run_liquid_10bp.py` (CLI:
`scripts/rq_liquid_10bp.py`).

## Full-run result (2026-10-31)

**Automated verdict: `clears`** (`soft_interesting`: true)

Wall time **686.3 s** (~11.4 min) on cloud CPU (no CUDA). Artifacts:
`results/liquid_10bp.json`, `results/liquid_10bp.png`.

**10 bp champion (val Sharpe):** `band_delta_0.05` | mean_cost **0.2127** | cvar95 **0.7367**
(audit: does **not** match frozen 20 bp `classical_best` spec δ=0.10;
`loaded_classical_best_json: false`).

| Arm | Arm verdict | Clearing seed(s) |
| --- | --- | --- |
| Risk-matched α sweep | `clears` | `rm_seed_1` (α=0.05 selected): cost **0.2118** &lt; champion, CVaR **0.7059** ≤ champion |
| Lagrange excess-CVaR | `clears` | `lag_seed_1`: cost **0.2119** &lt; champion, CVaR **0.7210** ≤ champion |

### Classical grid @ 10 bp (hull points only)

| label | mean_cost | cvar95 | mean_sharpe |
| --- | ---: | ---: | ---: |
| time_f1 | 0.2301 | 0.7232 | 0.699 |
| time_f3 | 0.1709 | 1.3130 | 0.539 |
| time_f7 | 0.1381 | 2.1260 | 0.412 |
| time_f21 | 0.1029 | 4.0186 | 0.283 |
| **band_delta_0.05** | **0.2127** | **0.7367** | **0.708** |
| band_delta_0.10 | 0.1888 | 0.8441 | 0.669 |
| band_delta_0.20 | 0.1608 | 1.2056 | 0.544 |
| band_price_0.01 | 0.2013 | 0.7454 | 0.624 |
| band_price_0.02 | 0.1696 | 0.9148 | 0.579 |
| band_price_0.05 | 0.1310 | 1.5905 | 0.394 |

### Risk-matched — selected seeds (headline)

| label | α | mean_cost | cvar95 | clears champion? |
| --- | ---: | ---: | ---: | --- |
| rm_seed_0_selected | 0.20 | 0.2163 | 0.7214 | no (cost above champion) |
| rm_seed_1_selected | 0.05 | 0.2118 | 0.7059 | **yes** |
| rm_seed_2_selected | 0.12 | 0.2169 | 0.7052 | no (cost above champion) |
| rm_ensemble_selected | — | 0.2150 | 0.6967 | no (cost above champion) |

### Lagrange — per seed

| label | mean_cost | cvar95 | clears champion? |
| --- | ---: | ---: | --- |
| lag_seed_0 | 0.2174 | 0.7688 | no |
| lag_seed_1 | 0.2119 | 0.7210 | **yes** |
| lag_seed_2 | 0.2109 | 0.7563 | no (tail above champion) |
| lag_ensemble | 0.2134 | 0.7174 | no (cost above champion) |

### Interpretation

Unconstrained deep hedging at 10 bp (RQ-G) sat **inside** the hull on cost but **above**
the Sharpe champion on tail risk. With an explicit **risk-matched α dial** or a
**train-time Lagrange tail budget** tied to the **re-swept** 10 bp champion (not the
20 bp JSON row), **one torch seed per arm** clears the Lead-locked rule on OOS rollout.
This is a **mandate-style** comparison against the **10 bp** classical champion, not a
claim of beating Sepp Case I economics under the half-spread ledger.
