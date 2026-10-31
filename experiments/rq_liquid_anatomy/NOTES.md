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
| **Torch init seeds** (`Hyper.torch_seeds`) | Expanded fixed set **`(0, 1, 2, 3, 4, 5)`** — documented for RQ-I alignment; no holdout |
| **Path train seeds** (`BenchConfig.train_seeds`) | Frozen bench default **`(1001, 1002, 1003)`** — `train_single_net` / Lagrange call `B.training_paths(cfg)`; not the torch-init list |

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

### Full run (2026-10-31)

**Automated verdict: `band_like`**

Wall time **385.2 s** on cloud CPU. Artifacts: `results/liquid_anatomy.json`,
`results/liquid_anatomy.png`. **10 bp champion (val Sharpe):** `band_delta_0.05` |
mean_cost **0.2127** | cvar95 **0.7367** (anatomy ε = **0.05** champion δ-band).
`audit.loaded_classical_best_json: false`.

| Policy | mean_cost | cvar95 | no-trade mass | mean \|Δδ\| | seed_verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **band_delta_0.05** (champion / comparator) | 0.2127 | 0.7367 | 0.476 | 0.0759 | comparator |
| rm α=0.05 seed 0 | 0.2229 | 0.7148 | 0.509 | 0.0797 | `band_like` |
| rm α=0.05 seed 1 | 0.2118 | 0.7059 | 0.541 | 0.0749 | `band_like` |
| rm α=0.05 seed 2 | 0.2206 | 0.7120 | 0.516 | 0.0792 | `band_like` |
| rm α=0.05 seed 3 | 0.2117 | 0.7040 | 0.540 | 0.0751 | `band_like` |
| rm α=0.05 seed 4 | 0.2148 | 0.6977 | 0.532 | 0.0764 | `band_like` |
| rm α=0.05 seed 5 | 0.2142 | 0.7095 | 0.526 | 0.0765 | `band_like` |
| lag α=0.65 seed 0 | 0.2174 | 0.7688 | 0.517 | 0.0782 | `inconclusive` |
| lag α=0.65 seed 1 | 0.2119 | 0.7210 | 0.536 | 0.0757 | `band_like` |
| lag α=0.65 seed 2 | 0.2109 | 0.7563 | 0.551 | 0.0753 | `inconclusive` |
| lag α=0.65 seed 3 | 0.2187 | 0.7679 | 0.529 | 0.0783 | `inconclusive` |
| lag α=0.65 seed 4 | 0.2101 | 0.7168 | 0.549 | 0.0746 | `band_like` |
| lag α=0.65 seed 5 | 0.2102 | 0.7101 | 0.546 | 0.0748 | `band_like` |

**Headline verdict:** `band_like` — **9 of 12** recipe rows have OOS CVaR95 ≤ champion; all
nine meet Lit bars vs the champion band (no-trade mass ≥ 0.8×, mean \|Δδ\| ≤ 1.25×). Lagrange
torch seeds **0, 2, 3** are **`inconclusive`** per seed because CVaR95 **>** champion (0.7367).

Deep policies still show **near-unity trade rate** at machine ε but **more no-trade mass at
ε=0.05** than the band champion, with **comparable or smaller** mean \|Δδ\| on CVaR-qualifying
rows — consistent with RQ-E-style **`band_like`** anatomy at 10 bp, not literal Leland ε→0
no-trade.

**Soft read (not headline):** `band_like` here applies only to these **fixed** risk-matched
(α=0.05) and Lagrange (α=0.65) recipes among seeds that **clear the champion tail** — not a
claim that default unconstrained `Hyper()` is band-like (see **RQ-K**).
