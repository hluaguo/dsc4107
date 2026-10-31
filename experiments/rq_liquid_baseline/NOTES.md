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

## Full-run result

*(Filled after full run — must match `results/liquid_baseline.json`.)*
