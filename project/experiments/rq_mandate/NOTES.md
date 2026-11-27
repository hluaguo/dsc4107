# RQ-B: mandate / activity floor on the frozen cell

## Research question

On the frozen Heston ATM one-month benchmark (20 bp proportional cost), if deep
hedging is trained and then deployed through a **mandate map** that hard-floors
out-of-sample **opening |δ|** and **turnover** to be at least `classical_best`'s
levels on the shared validation pool, can **any** default `Hyper()` torch seed
**clear** the pinned baseline on both OOS mean proportional cost **<** CB and OOS
CVaR95 ≤ CB?

This is not a Sepp-versus-Black–Scholes-delta diagnosis; it asks whether forcing
classical-scale **activity** removes the cheap-but-tail-heavy gap seen in RQ-A.

## Cell and design

| Setting | Value |
| --- | --- |
| Benchmark | `BenchConfig()` — Heston ATM call, `cost=0.002`, `n_steps=21` |
| Validation | Same as RQ-A: all `val_seeds` × `n_val_paths` on identical paths |
| Classical | RQ-A grid + `classical_best`; activity floors read from CB on this pool |
| Learned | Default `Hyper()` (200 epochs, three torch seeds); CVaR training loss unchanged |
| Mandate | In-sample penalty when batch mean opening |δ| or turnover sits below CB OOS floors; OOS positions passed through a deterministic rescaling map before `hedge.rollout` |
| Scoring | Dual ledger: `deep.hedge_positions` → mandate map → `hedge.rollout` (same as RQ-A / RQ-D) |

### Activity definitions (OOS, pooled over all validation paths)

| Metric | Definition |
| --- | --- |
| **mean_initial_abs_delta** | Mean of \|δ\| after the first rebalance (decision `j=0`), before transaction costs. Band rules open a full Black–Scholes delta on the first step, so every classical grid point shares the same opening level (~0.514 here). |
| **mean_turnover** | Mean path turnover from `hedge.rollout`: Σ\|Δδ\| over decision steps plus terminal \|δ\| (position units, not dollar notional). |
| **Hard floor** | Pooled OOS `mean_initial_abs_delta` **and** `mean_turnover` each **≥** the corresponding `classical_best` statistic on the same paths. |
| **floor_met** | Both inequalities hold **after** the mandate position map (seeds that miss the floor cannot **Pass**). |

Floors for this full run (from `classical_best` on the shared pool):

- opening |δ| ≥ **0.5144**
- turnover ≥ **1.8692**

## Acceptance (Lead-locked)

Evaluated on **OOS rollout** metrics only, after the mandate map.

| Verdict | Criterion |
| --- | --- |
| **`clears` (Pass)** | **Any** torch seed with `floor_met` **and** mean_cost **<** `classical_best` **and** CVaR95 ≤ `classical_best`. |
| **`interesting`** | At least one seed with `floor_met` and CVaR95 ≤ CB but **no** seed clears (mean_cost still ≥ CB for every seed that meets the tail bar). |
| **`fail`** | Otherwise (including all seeds above the CVaR bar while on the activity floor). |

## How to run

From repo root:

```bash
uv run python scripts/rq_mandate.py --quick   # smoke → experiments/rq_mandate/results/mandate_quick.json
uv run python scripts/rq_mandate.py           # full cell → experiments/rq_mandate/results/mandate.json
```

Implementation: `experiments/rq_mandate/run_mandate.py` (CLI shim: `scripts/rq_mandate.py`).

## Full-run result (2026-10-30)

**Automated verdict: `fail`**

Wall time **121.3 s** on CPU (no CUDA). Artifacts: `results/mandate.json`, `results/mandate.png`.

### Baseline vs mandate deep (OOS rollout)

| Policy | mean_cost | cvar95 | init \|δ\| | turnover | floor_met |
| --- | ---: | ---: | ---: | ---: | --- |
| **classical_best** | 0.3776 | 1.0661 | 0.5144 | 1.8692 | (reference) |
| mandate_seed_0 | 0.3795 | 1.1924 | 0.5206 | 1.8816 | yes |
| mandate_seed_1 | 0.3772 | 1.2356 | 0.5233 | 1.8692 | yes |
| mandate_seed_2 | 0.3908 | 1.3220 | 0.5276 | 1.9379 | yes |
| mandate_ensemble | 0.3815 | 1.2334 | 0.5239 | 1.8913 | yes |

**Pass / clears?** No. All three seeds meet the activity floor, but every seed has
CVaR95 **above** `classical_best` (1.19–1.32 vs 1.066). `mandate_seed_1` is
slightly cheaper on mean cost than CB (0.3772 vs 0.3776) yet still fails the tail
bar, so the run is **`fail`**, not **`interesting`** (no seed achieves CVaR95 ≤ CB).

### Contrast with RQ-A (unconstrained)

RQ-A's unconstrained deep nets traded **less** than the band baseline on turnover
while cutting cost and **raising** tail risk. RQ-B forces classical-scale opening
size and turnover; cost moves back toward CB, but CVaR95 does **not** recover to
the baseline band — the tail gap is not explained by low activity alone on this cell.

## References (NOTES cite-or-die)

- **Sepp (2013)** — discrete hedging with proportional costs; `classical_best` is
  the pinned band champion from the same frequency/band grid Sepp optimises in
  spirit, here scored on cost and CVaR rather than Sharpe alone.
- **Buehler et al. (2019)** — deep hedging with convex tail risk objectives; the
  learned side keeps the default CVaR training loss and network features, with
  mandate penalties/map layered only in this experiment script (frozen core untouched).
