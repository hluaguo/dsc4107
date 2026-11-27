# RQ-I: liquid 10 bp robustness (fixed RQ-H recipes, more seeds)

## Research question

RQ-H reported **one torch seed per arm** clearing the re-swept 10 bp classical
champion on OOS rollout. RQ-I holds the **same cell** and the **fixed clearing
recipes** from that run — risk-matched training at **α = 0.05** (no α sweep) and
Lagrange excess-CVaR at **α = 0.65** — but trains on **more** `torch_seeds` than
RQ-H's `(0, 1, 2)`. The Lead verdict asks whether the soft liquid clear is
**robust** or **brittle** one-seed luck.

## Cost convention

**Cost convention.** Same as RQ-H: Sepp (2013) eq (2.5) charges **k/2** on |Δδ|·S; Case I liquid-index **k = 0.001**. Our ledger is **full-rate** `cost · |trade| · S` (Buehler et al. 2019 eq 5.7). RQ-I keeps `cost = 0.001` — Sepp liquid-index **k** level under our full-rate ledger, **not** exact Case I friction. Re-sweep hull at 0.001; never reuse `classical_best.json`.

## Framing

**Framing.** RQ-I asks whether the RQ-H one-seed clears (`rm` α=0.05 / `lag` α=0.65) survive an expanded train-seed set — robustness of a soft liquid clear, not a new paper claim. Do not invent multi-seed “stability” cites; verdicts stay Lead-locked (`clears` / `brittle` / `interesting`).

(JSON maps Lead language to **`robust`** / **`brittle`** / **`fails`**; `soft_interesting` is true only when verdict is **`robust`**.)

## Cell (same ledger as RQ-H)

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` with **`cost = 0.001`** (10 bp full rate); Heston ATM 1M, `n_steps=21` |
| Validation | Shared OOS: `val_seeds` 2001–2008 × 40_000 paths |
| Deep training | `Hyper()` training hyperparams; **`torch_seeds` expanded** (see below); no holdout seeds |
| OOS scoring | Dual `hedge.rollout` ledger for classical and deep |
| Classical hull | RQ-A grid at **k = 0.001** only — never `classical_best.json` |
| Champion | Highest **mean validation Sharpe** on that grid at 10 bp |

Audit: `loaded_classical_best_json: false`.

## Seed design vs RQ-H

| | RQ-H | RQ-I (full) |
| --- | --- | --- |
| `torch_seeds` | `(0, 1, 2)` — default `Hyper()` | `(0, 1, 2, 3, 4, 5)` — retains H's three, adds 3–5 |
| RM arm | Full α sweep; headline **selected** α=0.05 at seed 1 | **Fixed** train α=0.05 only (H's clearing recipe) |
| Lag arm | Fixed `Hyper().alpha = 0.65` | Same fixed α=0.65 |

Quick smoke: `torch_seeds = (0, 1)` and `Bench._quick_setup`.

## Arms (two only)

| Arm | Mechanism | Headline rows |
| --- | --- | --- |
| **Risk-matched fixed α** | Train once per seed at **α = 0.05** (RQ-H `rm_seed_1` recipe) | `rm_fixed_seed_*` |
| **Lagrange fixed α** | RQ-F-style dual on batch CVaR vs champion tail; **α = 0.65** | `lag_fixed_seed_*` |

## Verdict (Lead locked)

Evaluated on **OOS rollout** `mean_cost` and `cvar95` vs the **10 bp champion**.

**Primary (pooled):** count every **arm × seed** row (`kind=seed`). A row **clears** if
`mean_cost` **<** champion **and** `cvar95` ≤ champion.

| Verdict | Criterion (pooled rows) |
| --- | --- |
| **`robust`** | ≥ **half** of pooled rows clear |
| **`brittle`** | **Exactly one** pooled row clears (RQ-H-like luck) |
| **`fails`** | Otherwise |

**Per-arm** columns in JSON use the same fractions but only that arm's seeds
(documented separately; headline is **pooled**).

Tiny cost margins are acceptable; do not oversell economics vs Sepp Case I.

## How to run

```bash
uv run python scripts/rq_liquid_robust.py --quick   # smoke → results/liquid_robust_quick.json
uv run python scripts/rq_liquid_robust.py           # full cell → results/liquid_robust.json
```

Implementation: `experiments/rq_liquid_robust/run_liquid_robust.py` (CLI:
`scripts/rq_liquid_robust.py`).

## Full-run result (2026-10-31)

**Automated pooled verdict: `fails`** (`soft_interesting`: false)

Wall time **375.5 s** (~6.3 min) on cloud CPU. Artifacts:
`results/liquid_robust.json`, `results/liquid_robust.png`.

**10 bp champion (val Sharpe):** `band_delta_0.05` | mean_cost **0.2127** | cvar95 **0.7367**
(`loaded_classical_best_json: false`).

| Scope | Verdict | Clearing rows |
| --- | --- | --- |
| **Pooled** (12 arm×seed rows) | **`fails`** | **5 / 12** clear (need ≥ 6 for `robust`; not `brittle` — more than one) |
| RM fixed α=0.05 | `fails` | 2 / 6 seeds (1, 3) |
| Lag fixed α=0.65 | **`robust`** | 3 / 6 seeds (1, 4, 5) |

Pooled clearing `torch_seeds`: **1, 3, 4, 5** (seed 1 clears on **both** arms; 3 on RM only; 4–5 on Lag only).

### RM fixed α=0.05

| seed | mean_cost | cvar95 | clears champion? |
| ---: | ---: | ---: | --- |
| 0 | 0.2229 | 0.7148 | no (cost) |
| 1 | 0.2118 | 0.7059 | **yes** |
| 2 | 0.2206 | 0.7120 | no (cost) |
| 3 | 0.2117 | 0.7040 | **yes** |
| 4 | 0.2148 | 0.6977 | no (cost) |
| 5 | 0.2142 | 0.7095 | no (cost) |

### Lag fixed α=0.65

| seed | mean_cost | cvar95 | clears champion? |
| ---: | ---: | ---: | --- |
| 0 | 0.2174 | 0.7688 | no (cost + tail) |
| 1 | 0.2119 | 0.7210 | **yes** |
| 2 | 0.2109 | 0.7563 | no (tail) |
| 3 | 0.2187 | 0.7679 | no (cost + tail) |
| 4 | 0.2101 | 0.7168 | **yes** |
| 5 | 0.2102 | 0.7101 | **yes** |

### Interpretation

RQ-H's one-seed-per-arm picture **does not** generalize as a pooled robust clear: fixed **α=0.05**
replicates on **two** of six seeds, while fixed Lagrange **α=0.65** clears on **half** of its seeds
(per-arm `robust`) but RM misses enough that the **pooled** count stops at **5/12**. That is
stronger than RQ-H **brittle** luck (more than one clearing row) yet **below** the Lead half bar
for an overall **`robust`** headline — a **mixed** liquid path, not a mandate to promote either
recipe without seed risk. Still not a Sepp Case I friction claim.
