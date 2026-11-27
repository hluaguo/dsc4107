# RQ-F: train-time CVaR cap vs `classical_best`

## Research question

On the frozen Heston ATM 1M cell (20 bp), if deep hedging is trained with a
**train-time** soft excess / Lagrange mechanism that targets OOS-style **CVaR95 ≤
`classical_best`** — **not** the RQ-D training-α dial — can **any** torch seed
**clear** the baseline on both rollout axes: OOS mean_cost **<** `classical_best`
**and** OOS CVaR95 ≤ `classical_best`?

## Cell and design

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` — Heston ATM call, `cost=0.002`, `n_steps=21` |
| Validation | Same as RQ-A: all `val_seeds` × `n_val_paths` on identical paths |
| Classical | RQ-A grid + `classical_best` via `hedge.rollout` (shared helpers) |
| Training | Default `Hyper()` (200 epochs, 3 torch seeds, **α = 0.65 fixed** — not swept) |
| Train constraint | Per mini-batch: base `deep.cvar_loss(pnl, alpha)` plus **λ · ReLU(batch_cvar95 − target)** with dual ascent on λ (`dual_lr = 0.75`). Target `batch_cvar95` matches `metrics.cvar95` (5% quantile tail). **Target level** = OOS `classical_best.cvar95` computed on the shared validation pool before training. |
| **Not RQ-D** | RQ-D varies **training-α** in `cvar_loss` and selects post hoc on OOS rollout. RQ-F keeps α at `Hyper()` default and adds a **separate tail budget penalty** during optimisation only. |
| OOS scoring | `hedge.rollout` for **both** classical and deep (`deep.hedge_positions` → same cost ledger as RQ-A) |
| Ensemble | Pathwise average of the three capped nets (same mixing as RQ-A) |

Equivalent formulation (documented, not a second code path): a **soft excess** penalty
ρ · ReLU(batch_cvar95 − target)² is the same family as the Lagrange term when ρ is
tuned; this run uses **dual ascent on λ** so the trainer reacts when batches breach
the classical tail level.

## Acceptance / verdict (Lead locked)

Evaluated on **OOS rollout** mean cost and CVaR95 only — not training loss or
in-sample paths.

| Verdict | Criterion |
| --- | --- |
| **`clears`** | **Any** torch seed has OOS `mean_cost` **<** `classical_best` **and** OOS `cvar95` ≤ `classical_best` |
| **`fails`** | Otherwise |

## Cite-or-die

- **Buehler et al. (2019)** — deep hedging with convex tail objectives; RQ-F keeps
  their CVaR loss class but adds an explicit **train-time tail budget** tied to the
  classical champion’s OOS CVaR95, rather than hoping a single α matches Sepp’s band.
- **Ahn & Wilmott (2009)** — risk-matched discrete hedging under costs; motivates
  asking whether a learned policy can **respect the same tail allowance** as the
  hand-built baseline while beating its turnover bill on rollout.
- **Li (2025)** — classical frequency literature only (no deep comparator in that
  line); cited softly as adjacent work, not as a method we implement here.

(Sepp 2013 / Leland 1985 underpin the classical hull via RQ-A; not re-derived here.)

## How to run

From repo root:

```bash
uv run python scripts/rq_hard_cvar.py --quick   # smoke → results/hard_cvar_quick.json
uv run python scripts/rq_hard_cvar.py           # full cell → results/hard_cvar.json
```

Implementation: `experiments/rq_hard_cvar/run_hard_cvar.py` (CLI shim:
`scripts/rq_hard_cvar.py`).

## Results

### Full run (2026-10-30)

**Automated verdict: `fails`**

Wall time **93.4 s** on CPU (no CUDA). Artifacts: `results/hard_cvar.json`,
`results/hard_cvar.png`.

| Policy | mean_cost (OOS) | cvar95 (OOS) | notes |
| --- | ---: | ---: | --- |
| **classical_best** | 0.3776 | 1.0661 | pinned band champion |
| deep_seed_0_cvar_capped | 0.4090 | 1.1496 | λ≈4.46; tail above baseline |
| deep_seed_1_cvar_capped | 0.3821 | 1.0452 | tail ≤ baseline; cost still above |
| deep_seed_2_cvar_capped | 0.3887 | 1.0312 | tail ≤ baseline; cost still above |
| deep_ensemble_cvar_capped | 0.3933 | 1.0524 | pathwise mix |

**Clears?** No — seeds 1 and 2 meet OOS CVaR95 ≤ `classical_best` but none achieve
OOS mean_cost **<** 0.3776, so the Lead **clears** rule fails. Seed 0 still overshoots
the tail budget on rollout despite positive in-training excess pressure (mean epoch
excess ~0.03 on training batches).

**Interpretation:** A batch-level Lagrange cap on 5% tail loss can pull some seeds
under the classical CVaR95 on the shared OOS pool, but not simultaneously under its
mean cost — the cost–tail trade-off that RQ-A and RQ-D already hint at persists when
the constraint is enforced during training rather than via an α sweep.

### Smoke run (`--quick`)

Not headline evidence. See `results/hard_cvar_quick.json`.
