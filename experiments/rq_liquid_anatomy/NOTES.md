# RQ-J: liquid 10 bp policy anatomy (fixed Hyper recipes)

## Research question

At Sepp liquid-index **k** under our **full-rate** ledger (`BenchConfig.cost = 0.001`),
do the **fixed** deep recipes from RQ-H — (1) CVaR loss at training **α = 0.05** and
(2) Lagrange excess-CVaR at **α = 0.65** — exhibit **band-like** or **distinct**
rebalancing anatomy versus the **10 bp val-Sharpe band champion**, among seeds that
meet the champion on OOS tail risk?

## Cite-or-die (liquid RQs)

**Cost.** Sepp (2013) eq (2.5) **k/2**; Case I **k=0.001**. Our ledger **full-rate** (Buehler 2019 eq 5.7). All liquid RQs: `cost=0.001` = Sepp liquid-index **k** under full-rate — **not** Case I. Re-sweep hull; never `classical_best.json`.

**J framing:** reuse RQ-E Lit bars vs 10 bp Band champ (no-trade ≥0.8×, |Δδ|≤1.25×); **not** literal Leland ε=0 no-trade.

Eval: band_like / distinct / inconclusive per those bars.

## Cell

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` with **`cost = 0.001`**; Heston ATM 1M, `n_steps=21` |
| Validation | Shared OOS: `val_seeds` 2001–2008 × 40_000 paths |
| Scoring | Dual `hedge.rollout` for classical and deep |
| Classical hull | RQ-A / RQ-G grid at 10 bp — re-sweep only; **`audit.loaded_classical_best_json: false`** |
| Champion | Highest mean validation Sharpe on that grid (RQ-H: `band_delta_0.05`) |
| Train seeds | **Expanded fixed set** `(0, 1, 2, 3, 4, 5)` — documented for RQ-I alignment; no holdout |

### Fixed recipes (no OOS α sweep)

| Recipe | Training |
| --- | --- |
| **Risk-matched dial (fixed)** | Single `deep.cvar_loss` with **α = 0.05** per torch seed |
| **Lagrange excess-CVaR** | Fixed **`Hyper().alpha = 0.65`** + train-time dual on batch CVaR vs champion tail |

### Anatomy metrics (same family as RQ-E)

Path-pooled over shared OOS from position sequences: **no-trade mass** (fraction of
steps with \|Δδ\| ≤ ε), **trade rate**, **mean \|Δδ\|**, **mean \|Δδ\| when trading**,
**time-in-band vs BS δ**, **turnover timing**. Here **ε = champion band width**
(δ-band 0.05 for the 10 bp Sharpe champion), not Leland ε→0.

### Lit bars vs 10 bp band champion

Primary comparator: **anatomy of the val-Sharpe champion** (band policy). Per seed
(recipe row), among seeds with **OOS CVaR95 ≤ champion**:

| Per-seed label | Criterion |
| --- | --- |
| **`band_like`** | no-trade mass ≥ **0.8 ×** champion band’s no-trade mass **and** mean \|Δδ\| ≤ **1.25 ×** champion’s mean \|Δδ\| **and** CVaR95 ≤ champion |
| **`distinct`** | CVaR95 ≤ champion but fails a ratio bar |
| **`inconclusive`** | CVaR not ≤ champion, or mixed labels under headline rules |

**Headline verdict** (among CVaR ≤ champion seeds only, both recipes pooled):

| Verdict | Rule |
| --- | --- |
| `band_like` | Every such seed is `band_like` |
| `distinct` | Every such seed is `distinct` |
| `inconclusive` | No qualifying seeds, or a mix of `band_like` / `distinct` |

## How to run

```bash
uv run python scripts/rq_liquid_anatomy.py --quick   # smoke → results/liquid_anatomy_quick.json
uv run python scripts/rq_liquid_anatomy.py           # full cell → results/liquid_anatomy.json
```

Implementation: `experiments/rq_liquid_anatomy/run_liquid_anatomy.py` (CLI:
`scripts/rq_liquid_anatomy.py`).

## Results

_(Full run populates this section from `results/liquid_anatomy.json`.)_
