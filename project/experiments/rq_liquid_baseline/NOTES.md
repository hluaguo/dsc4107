# RQ-K: liquid 10 bp unconstrained baseline

## Research question

At Sepp liquid-index **k** under our full-rate ledger (`BenchConfig.cost = 0.001`), does
**unconstrained** default `Hyper()` — CVaR training at `alpha = 0.65`, **no** risk-match
α selection, **no** Lagrange excess-CVaR — produce **any** torch seed that **clears** the
**10 bp classical champion** on shared OOS rollout (`mean_cost` strictly below champion
**and** `cvar95` ≤ champion)?

Purpose: isolate whether RQ-H's clears need the risk-match / Lagrange knobs.

## Cost convention (shared with RQ-I / RQ-H)

**Cost.** Sepp (2013) eq (2.5) **k/2**; Case I **k=0.001**. Our ledger **full-rate** (Buehler 2019 eq 5.7). All liquid RQs: `cost=0.001` = Sepp liquid-index **k** under full-rate — **not** Case I. Re-sweep hull; never `classical_best.json`.

**K framing:** unconstrained Hyper isolates whether H's clears need risk-match / Lagrange — no new paper claim.

Eval verdicts: dominated / clears / mixed.

## Cell

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` with **`cost = 0.001`**, Heston ATM 1M, `n_steps=21` |
| Training paths | Expanded **train** seeds (RQ-I lock): **1001–1006** × `n_train_paths` |
| Torch inits | Expanded beyond RQ-H's {0,1,2} (RQ-I lock): **0, 1, 2, 3, 4, 5** |
| Deep training | Default `Hyper()` CVaR (`alpha=0.65`); no RM sweep; no Lagrange |
| Validation | Shared OOS: `val_seeds` 2001–2008 × 40_000 paths |
| OOS scoring | Dual `hedge.rollout` ledger for classical and deep |
| Classical hull | RQ-A / RQ-G grid at **k = 0.001** only |
| Champion | Highest **mean validation Sharpe** on that grid (**not** `classical_best.json`) |

JSON audit: `loaded_classical_best_json: false`. Frozen CB may appear only for optional
`champion_matches_frozen_cb_spec` audit.

## Verdict (Lead locked)

Evaluated on OOS rollout mean cost and CVaR95 only (lower is better on both).

| Verdict | Criterion |
| --- | --- |
| **`clears`** | **Any** torch seed has OOS `mean_cost` **<** 10 bp champion **and** OOS `cvar95` ≤ champion |
| **`dominated`** | **Every** deep row (each seed **and** the ensemble) is weakly dominated by at least one classical grid point (RQ-A hull rule) |
| **`mixed`** | Not dominated and no seed clears the champion |

**Soft note (`soft_interesting` in JSON):** true if **`clears`**, or **≥ half** of torch seeds clear the champion, or some seed has `cvar95` ≤ champion with `mean_cost` within **1%** of champion mean cost.

## How to run

```bash
uv run python scripts/rq_liquid_baseline.py --quick   # smoke → results/liquid_baseline_quick.json
uv run python scripts/rq_liquid_baseline.py           # full cell → results/liquid_baseline.json
```

Implementation: `experiments/rq_liquid_baseline/run_liquid_baseline.py` (CLI:
`scripts/rq_liquid_baseline.py`).

## Full-run result (2026-10-31)

**Automated verdict: `dominated`** (`soft_interesting`: false)

Wall time **310.6 s** (~5.2 min) on cloud CPU (no CUDA). Artifacts:
`results/liquid_baseline.json`, `results/liquid_baseline.png`.

**10 bp champion (val Sharpe):** `band_delta_0.05` | mean_cost **0.2127** | cvar95 **0.7367**
(`loaded_classical_best_json: false`; champion spec ≠ frozen 20 bp δ=0.10).

| Policy | mean_cost | cvar95 | Clears champion? |
| --- | ---: | ---: | --- |
| deep_seed_0 | 0.2070 | 1.0084 | no (tail above champion) |
| deep_seed_1 | 0.2108 | 1.0158 | no |
| deep_seed_2 | 0.2062 | 0.9754 | no |
| deep_seed_3 | 0.2047 | 0.9720 | no |
| deep_seed_4 | 0.2054 | 0.9840 | no |
| deep_seed_5 | 0.1988 | 0.8861 | no (cost below champion; tail still above) |
| deep_ensemble | 0.2055 | 0.9678 | no |

**Clearing seeds:** 0 / 6. Every deep row (seeds + ensemble) is weakly dominated by at least
one classical grid point on OOS rollout — same cost–tail pattern as RQ-G at 10 bp.

### Interpretation

Unconstrained `Hyper()` at liquid **k** under full-rate **does not** clear the re-swept 10 bp
Sharpe champion without risk-match or Lagrange (contrast RQ-H `clears`). Expanded RQ-I/K
train paths (1001–1006) and six torch inits do not change that headline: cheaper schedules
still sit **above** the champion on CVaR95. This supports the K framing — H's clears trace
to the mandate knobs, not to default CVaR training alone.
