# RQ-L: liquid 10 bp risk-matched α grid

## Research question

At Sepp liquid-index **k** under our full-rate ledger (`cost = 0.001`), is **α = 0.05**
special for **clearing** the re-swept 10 bp val-Sharpe champion after the RQ-D
**risk-matched** post-selection rule, or do **other training-α** values clear too when
each torch seed picks min OOS cost among `cvar95 ≤ champion`?

This is **α-grid robustness** of the soft clear seen in RQ-H (risk-matched arm), not a
second Lagrange arm and not a claim that 0.05 is a cited optimum.

## Literature (soft cite — paste as-is)

**Cost.** Sepp (2013) eq (2.5) **k/2**; Case I **k=0.001**. Our ledger **full-rate** (Buehler 2019 eq 5.7). All liquid RQs: `cost=0.001` = Sepp liquid-index **k** under full-rate — **not** Case I. Re-sweep hull; never `classical_best.json`.

**L framing:** α-grid robustness of the soft clear — not a Buehler α theorem; α=0.05 is an H recipe, not a cited optimum.

Eval verdicts: `a05_only` / `several_clear` / `none`.

## Cell

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` with **`cost = 0.001`**; Heston ATM 1M, `n_steps=21` |
| Validation | Shared OOS: `val_seeds` 2001–2008 × 40_000 paths |
| Deep training | Default `Hyper()` budget; **`torch_seeds = (0, 1, 2, 3, 4)`** (same expanded set as RQ-I; no holdout) |
| Training-α grid | **{0.05, 0.12, 0.20, 0.30, 0.40}** — compact RQ-D subset including 0.05 and neighbors |
| OOS scoring | Dual `hedge.rollout` ledger (classical + deep), same as RQ-A / D / H |
| Classical hull | RQ-A / G grid at **k = 0.001** only; champion = max val Sharpe on that grid |
| JSON audit | `loaded_classical_best_json: false` |

## Selection (RQ-D)

Per torch seed, after scoring every α on the shared OOS pool:

1. **Feasible set:** sweep points with OOS `cvar95 ≤ champion.cvar95`.
2. **Selected point:** minimum OOS `mean_cost` in the feasible set; tie-break lower `cvar95`.
3. If the feasible set is empty: selected point = minimum OOS `cvar95`; tie-break lower `mean_cost`.

Headline metrics for that seed are the **selected** point only (not the full sweep).

## Clear and verdict (Lead locked)

**Clear (per seed):** on the **selected** OOS point,

`mean_cost < champion.mean_cost` **and** `cvar95 ≤ champion.cvar95`.

| Verdict | Rule |
| --- | --- |
| **`none`** | No torch seed’s selected point clears. |
| **`a05_only`** | At least one selected seed clears, and **every** clearing selected seed has `alpha == 0.05` (equivalently: the set of distinct clearing training-α values is exactly `{0.05}`). |
| **`several_clear`** | At least one selected seed clears, and **some** clearing selected seed has `alpha ≠ 0.05` (two or more distinct clearing α values, or any clear at α other than 0.05). |

JSON also records **`sweep_alphas_that_clear`**: any grid α whose **raw** sweep point clears (even if not selected), for audit only — verdict uses **selected** points.

## How to run

```bash
uv run python scripts/rq_liquid_alpha.py --quick   # smoke → results/liquid_alpha_quick.json
uv run python scripts/rq_liquid_alpha.py           # full cell → results/liquid_alpha.json
```

Implementation: `experiments/rq_liquid_alpha/run_liquid_alpha.py` (CLI: `scripts/rq_liquid_alpha.py`).

## Results

### Full run (2026-10-31)

**Automated verdict: `a05_only`**

Wall time **667.8 s** (~11.1 min) on cloud CPU (no CUDA). Artifacts:
`results/liquid_alpha.json`, `results/liquid_alpha.png`.

**10 bp champion (val Sharpe):** `band_delta_0.05` | mean_cost **0.2127** | cvar95 **0.7367**
(`loaded_classical_best_json: false`).

| torch seed | selected α | mean_cost | cvar95 | clears? |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.20 | 0.2163 | 0.7214 | no |
| 1 | 0.05 | 0.2118 | 0.7059 | **yes** |
| 2 | 0.12 | 0.2169 | 0.7052 | no |
| 3 | 0.05 | 0.2117 | 0.7040 | **yes** |
| 4 | 0.05 | 0.2148 | 0.6977 | no (cost above champion) |
| ensemble | — | 0.2143 | 0.6949 | no (cost above champion) |

**Sweep audit:** only training-α **0.05** produced any raw sweep point with both
`mean_cost < champion` and `cvar95 ≤ champion` (`sweep_alphas_that_clear: [0.05]`).
Among **selected** points, the two clears both sit at α = 0.05 → **`a05_only`**, not
`several_clear`.

### Smoke run (`--quick`) — not headline evidence

Verdict **`none`** on shrunk pools. See `results/liquid_alpha_quick.json`.
