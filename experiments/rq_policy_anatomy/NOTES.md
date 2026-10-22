# RQ-E: policy anatomy of risk-matched deep hedgers

## Research question

On the frozen Heston ATM 1M / 20 bp cell, do the **RQ-D α-selected** deep nets that beat
`classical_best` on OOS CVaR95 (while paying higher mean cost) rediscover a Leland/Imaki-style
**no-trade / band** structure, or do they rebalance with a **distinct** pattern that explains
the higher E[cost]?

## Cell and design

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` — Heston ATM call, `cost=0.002`, `n_steps=21` |
| Validation | Same OOS as RQ-A/RQ-D: all `val_seeds` × `n_val_paths` |
| Deep | RQ-D recipe: per torch seed, train α grid in `deep.cvar_loss`, **select** min mean_cost among OOS CVaR95 ≤ `classical_best` else min CVaR; helpers imported from `experiments/rq_risk_matched/` and `rq_a/` |
| Checkpoints | Prefer RQ-D **selection metadata** from `experiments/rq_risk_matched/results/*.json` (chosen α per seed) when present; weights are retrained (RQ-D does not ship checkpoints) |
| Comparators (anatomy baseline) | **`BandDelta(0.02)`** — ratios and pass/fail bars vs this policy only |
| Reference (context) | **`TimeBased` f=1** (`time_f1` from RQ-A grid); cost/CVaR via `hedge.rollout`, not used in acceptance |
| Cost / tail context | `mean_cost`, `cvar95` from `hedge.rollout` for all policies |

### Anatomy metrics (path-pooled over shared OOS)

Computed in `run_policy_anatomy.py` from position sequences (`make_classical_position_fn` replay or
`deep.hedge_positions`); frozen modules are not edited.

| Metric | Definition |
| --- | --- |
| **no-trade mass** | Fraction of path-steps with \|Δδ\| ≤ ε, ε = **0.02** (BandDelta(0.02) width) |
| **trade rate** | Fraction of path-steps with \|Δδ\| > 0 (context) |
| **mean \|Δδ\|** | Pooled mean of \|Δδ\| over all path-steps |
| **mean \|Δδ\| when trading** | Mean \|Δδ\| on steps with nonzero change (context) |
| **time-in-band vs BS δ** | Fraction of steps with \|pos − BS_delta(spot)\| ≤ 0.02 (Imaki/Leland occupancy; context) |
| **turnover timing** | Share of pooled \|Δδ\| mass at each step index (0 … n_steps−1) |

## Acceptance / verdict (Lit bars)

Evaluated **per RQ-D α-selected seed** against **`BandDelta(0.02)`** on the same OOS pool.
Headline verdict uses only seeds with **OOS CVaR95 ≤ `classical_best`**.

| Per-seed label | Criterion |
| --- | --- |
| **`band_like`** | (1) no-trade mass ≥ **0.8 ×** Band’s no-trade mass, **and** (2) mean \|Δδ\| ≤ **1.25 ×** Band’s mean \|Δδ\|, **and** (3) OOS CVaR95 ≤ `classical_best` |
| **`distinct`** | OOS CVaR95 ≤ `classical_best` but fails (1) or (2) — **costly chatter** anatomy |
| **`inconclusive`** | CVaR not better than baseline, or mixed seed labels under the rules below |

**Headline verdict** (among CVaR-better seeds only):

| Verdict | Rule |
| --- | --- |
| `band_like` | Every CVaR-better seed is `band_like` |
| `distinct` | Every CVaR-better seed is `distinct` |
| `inconclusive` | No CVaR-better seeds, or a mix of `band_like` / `distinct`, or other edge cases |

**Falsification sketch (documentation only):**

- A **band-like** story is falsified if CVaR-better deep nets fail the 0.8× / 1.25× bars vs Band(0.02).
- A **distinct costly-chatter** story is falsified if trade stats match Band yet mean cost stays higher without a separate turnover explanation (report cost in JSON; label `inconclusive` when anatomy matches band but cost does not).

## Cite-or-die

- **Leland (1985)** — proportional-cost discrete hedging; motivates band/no-trade regions at 20 bp.
- **Imaki et al. (arXiv:2103.01775)** — optimal rebalancing bands under transaction costs; informs no-trade mass and time-in-band readouts.
- **Buehler et al. (2019)** — deep hedging with tail losses; RQ-D α dial stays in their CVaR-loss family (not a hard OOS CVaR constraint in training).

## How to run

From repo root:

```bash
uv run python scripts/rq_policy_anatomy.py --quick   # smoke → results/policy_anatomy_quick.json
uv run python scripts/rq_policy_anatomy.py             # full cell → results/policy_anatomy.json
```

Implementation: `experiments/rq_policy_anatomy/run_policy_anatomy.py` (CLI shim:
`scripts/rq_policy_anatomy.py`).

## Results

### Smoke run (`--quick`) — not headline evidence

Wall time ~2–70 s depending on cache; artifacts: `results/policy_anatomy_quick.json`,
`results/policy_anatomy_quick.png`. **Verdict: `inconclusive`** (smoke pool; seed not CVaR-better
than shrunk baseline).

| Policy | mean_cost | cvar95 | no-trade mass | mean \|Δδ\| | label |
| --- | ---: | ---: | ---: | ---: | --- |
| BandDelta(0.02) | 0.451 | 1.016 | 0.245 | 0.082 | comparator |
| TimeBased f=1 | 0.459 | 1.022 | 0.251 | 0.084 | reference |
| classical_best (context) | 0.376 | 1.101 | — | — | acceptance CVaR cap |
| deep_seed_0_selected | 0.553 | 1.523 | 0.149 | 0.104 | `inconclusive` |

**Headline verdict (smoke):** `inconclusive`

### Full run (2026-10-30)

**Automated verdict: `band_like`**

Wall time **115.5 s** on CPU. RQ-D α hints from `risk_matched.json` (retrain one net per
seed at chosen α; no weight checkpoints in RQ-D). Artifacts: `results/policy_anatomy.json`,
`results/policy_anatomy.png`.

| Policy | mean_cost | cvar95 | no-trade mass | mean \|Δδ\| | seed_verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| **classical_best** (band δ=0.10) | 0.3776 | 1.0661 | — | — | CVaR cap |
| BandDelta(0.02) | 0.4524 | 0.9944 | 0.245 | 0.0823 | comparator |
| TimeBased f=1 | 0.4602 | 0.9989 | 0.251 | 0.0841 | reference |
| deep_seed_0_selected (α=0.30) | 0.4020 | 1.0131 | 0.296 | 0.0709 | `band_like` |
| deep_seed_1_selected (α=0.40) | 0.3939 | 1.0553 | 0.310 | 0.0693 | `band_like` |
| deep_seed_2_selected (α=0.20) | 0.4052 | 0.9693 | 0.290 | 0.0714 | `band_like` |
| deep_ensemble_selected | 0.4004 | 1.0071 | 0.299 | 0.0705 | — |

All three CVaR-better seeds meet Lit bars vs Band(0.02): no-trade mass ≥ 0.8× Band and mean
\|Δδ\| ≤ 1.25× Band. Deep policies still show **near-unity trade rate** at ε=0 (micro-adjustments
each step) but **more no-trade mass at ε=0.02** than Band(0.02), with **smaller** mean \|Δδ\|.
Mean cost remains **above** `classical_best` despite band-like anatomy — consistent with the
Lead’s “inconclusive” cost narrative if one expected distinct chatter, but anatomy verdict is
**`band_like`** under the locked bars.
