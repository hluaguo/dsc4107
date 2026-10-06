# Handoff — classical vs learned option hedging (DSC4107)

> **Update (later session).** The plan below is complete: the full comparison
> map, the Optuna add-on, the Heston replication, the bootstrap intervals and
> the locked holdout were all run, and `REPORT.md`, `SIDEQUEST.md` and
> `EXPLORATION.md` were written. The JD variance issue listed under Open
> issues is resolved (missing `q = sqrt(pi^3)/4` factor in `hedging/sepp.py`);
> see `REPORT.md` and `scripts/diagnose_jd.py`. The historical text below is
> kept unchanged.

Written at a deliberate stopping point: **the two paper replications are
implemented and tested, the comparison/addon harness is written and smoke-tested,
but the full sweeps and the written report are not done.** Nothing below is
speculative; every "done" item was executed in this session.

Read `PLAN.md` (the research plan) and `AGENTS.md` (documentation and dependency
conventions — why-style docstrings, `uv add`, which files are off limits) before
continuing.

## State at handoff

- Branch: `main`, last commit before this work `87e6fbd`.
- This session's changes are **uncommitted** (see file inventory below); commit
  them before starting so the next diff is clean.
- `uv run pytest -q` → **23 passed** (13 pre-existing + 10 new Sepp tests).
- New dependency `optuna` added with `uv add optuna` (lockfile updated).
- No frozen file was modified. `hedging/deep.py` gained one additive helper
  only (`hedge_positions`), no behaviour change to training or the benchmark.

## Deliverable status against PLAN.md

| # | Deliverable | Status |
| - | - | - |
| 1 | Simulator library + test suite verified against Sepp's formulas | **Core done.** `hedging/sepp.py` + `tests/test_sepp.py` (10 tests). Four models already existed in `hedging/market.py`. |
| 2 | Notebook reproducing Sepp Section 5 tables/figures | **Generated, not executed end to end.** `notebooks/01_sepp_replication.ipynb`; script form in `scripts/sepp_replication.py`. |
| 3 | Notebook reproducing Deep Hedging Heston experiment | **Generated, not executed end to end.** `notebooks/02_deep_hedging_heston.ipynb`; script form in `scripts/deep_hedging_replication.py`. |
| 4 | Comparison study with Optuna sweeps + win-map | **Harness written and smoke-tested. Full run NOT done.** `scripts/comparison_study.py`. |
| 5 | Written report | **NOT started.** All numbers needed are in `results/`. |

## What is new / changed

| Path | What it is |
| - | - |
| `AGENTS.md` | Repo conventions: why-not-what docstrings, `uv add`/`uv remove`, frozen files, holdout rule. |
| `hedging/sepp.py` | Sepp (2013) closed forms: cash gamma, `z_integral`, Heston MGF, `q_sv`/`q_jd`, per-model `u,c,p,f`, `sepp_stats`, `optimal_frequency`, `band_from_frequency`, Table 1 presets. |
| `tests/test_sepp.py` | 10 checks pinning the formulas to Monte Carlo and to Table 2. |
| `scripts/sepp_replication.py` | Prints Table 2 reproduction + optima/bands, writes `results/sepp/*.json` and 5 figures. `--mc` adds a simulator cross-check. |
| `scripts/deep_hedging_replication.py` | Frictionless delta-recovery check + deep-vs-BS-delta-vs-band Sharpe across costs. |
| `scripts/comparison_study.py` | Optuna deep-hyper search + classical retuning per `(model, cost)`, edge map. `--fixed-hyper` skips the search. |
| `scripts/build_notebooks.py` | Regenerates the two `.ipynb` from readable sources. |
| `notebooks/01_sepp_replication.ipynb`, `notebooks/02_deep_hedging_heston.ipynb` | The two replication notebooks. |
| `hedging/deep.py` | **Only** added `hedge_positions()` (position audit for the frictionless check). Training, features, loss untouched. |
| `pyproject.toml`, `uv.lock` | `optuna` added via `uv add`. |

## Verified results so far (safe to quote in the report)

### Sepp Table 2 reproduction (case I / II, all in %)

From `uv run python scripts/sepp_replication.py`:

```
case   model |  u ours u paper |  c ours c paper |  p ours p paper |  f ours f paper |   N*   Np |     S    Sp
   I     gbm |  0.5693  0.5692 |  0.0144  0.0151 |  0.2519  0.2461 | 0.00044 0.00040 | 242.0  239 |  0.83  0.89
   I      jd |  0.5640  0.5510 |  0.0136  0.0139 |  0.1521  0.2006 | 0.00138 0.00200 | 144.0  138 |  0.76  0.66
   I  heston |  0.6621  0.6703 |  0.0144  0.0153 |  0.2583  0.2524 | 0.01075 0.01070 |  75.0   74 |  0.43  0.45
   I     svj |  0.6551  0.6550 |  0.0136  0.0142 |  0.2164  0.2085 | 0.01100 0.01100 |  70.0   68 |  0.43  0.45
  II     gbm |  1.8080  1.8558 |  0.0525  0.0571 |  0.6138  0.5998 | 0.00396 0.00400 | 130.0  129 |  1.18  1.30
  II      jd |  1.7826  1.7628 |  0.0481  0.0506 |  0.3423  0.4423 | 0.00708 0.01190 |  85.0   75 |  1.16  0.99
  II  heston |  2.0054  2.0876 |  0.0525  0.0581 |  0.6359  0.6214 | 0.04171 0.04030 |  49.0   49 |  0.65  0.73
  II     svj |  1.9738  1.9995 |  0.0481  0.0518 |  0.4903  0.4634 | 0.04109 0.04180 |  45.0   43 |  0.68  0.72
```

Max relative error in `u`: **3.9%**. `u` matches exactly for GBM case I
(0.5693 vs 0.5692) and SVJ case I (0.6551 vs 0.6550). Known weak spot: **JD `p`
and `f`** (see Open issues).

### Sepp Monte Carlo cross-check (`--mc --mc-paths 60000`)

Sharpe analytic vs MC: GBM 0.83/0.85, JD 0.76/0.64, Heston 0.43/0.46,
SVJ 0.43/0.46 (case I). Case II Heston 0.65/0.72, SVJ 0.68/0.72. The JD gap is
the same variance-rate issue.

### Deep hedging quick run (`--quick`, not a result, just a smoke test)

Frictionless (cost 0, 10 epochs, hidden 32): **correlation 0.9717, slope 1.090,
RMSE 0.0751**, mean learned 0.498 vs mean delta 0.505. Promising but needs a
real training budget before it is quoted.

### Comparison smoke test (`--quick --fixed-hyper`, 1 cell)

GBM cost 2bp: edge −0.114 with 10 epochs / 256 train paths — expected to be
negative at that budget, only proves the harness runs.

## Exact commands for the next session

```bash
# 0. commit this handoff state first
git add -A && git commit -m "Sepp replication, deep-hedging check, comparison harness, notebooks"

# 1. regression suite (fast, ~2s)
uv run pytest -q

# 2. Sepp tables + figures (~10s), optionally with MC (~1-2 min)
uv run python scripts/sepp_replication.py
uv run python scripts/sepp_replication.py --mc --mc-paths 60000

# 3. Deep hedging Heston replication (SLOW: 4 costs x 3 seeds; budget 20-40 min)
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3

# 4. Comparison: fast edge map with one fixed configuration per cell
uv run python scripts/comparison_study.py --fixed-hyper --epochs 80 \
    --costs 0.0 0.002 0.005 0.01 0.02
# 4b. Comparison: the Optuna addon, one (or few) cell(s) at a time
uv run python scripts/comparison_study.py --models heston --costs 0.002 0.01 \
    --trials 8 --epochs 80

# 5. regenerate notebooks after any API change
uv run python scripts/build_notebooks.py
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb  # needs jupyter
```

## Next steps, in priority order

1. **Run the full comparison map** (`--fixed-hyper`, all 4 models × 5 costs).
   This is the original contribution of the project. Expect ~30-60 min on CPU;
   reduce `--epochs`/`--val-paths` if needed and say so in the report.
2. **Run the Optuna addon** on at least Heston and one jump model, at 2-3 cost
   levels. Keep `SEARCH_SEEDS = (4001,4002,4003)` as the selection pool; never
   select on `val_seeds`.
3. **Run the deep-hedging notebook/script at full budget** and record the
   frictionless correlation/slope and the Sharpe-vs-cost curve.
4. **Write `REPORT.md`** covering: the two replications, the frictionless
   check, the win-map, and the guard-rail numbers (generation gap, win rate,
   per-seed min). Numbers live in `results/`.
5. **Update `README.md`** so the new scripts, `AGENTS.md`, and the `optuna`
   dependency are documented.
6. **Final holdout run** `uv run python scripts/holdout_eval.py` **once**, after
   all choices are frozen. Never read the holdout pool before then.

## Open issues / things not to forget

- **JD variance discrepancy.** `hedging/sepp.py` reproduces Table 2 poorly for
  the jump-diffusion `p` (0.1521 vs 0.2006, −24%) and `f` (0.00138 vs 0.00200),
  and the MC volatility for JD is correspondingly high. The likely cause is the
  equation (7.12)/(7.13) second-moment gamma convention (the `q2,q1` used in
  `q_jd` at `T/2`). This is documented as a known gap, not hidden. Fixing it
  would tighten the JD row; it does not affect the deep-vs-classical benchmark.
- **Paper equation (7.2) is mis-transcribed.** `z_integral` uses a re-derived
  exact completed square because the printed `B, C, A` do not reproduce a direct
  quadrature. See the docstring and `test_z_integral_matches_quadrature`.
- **Paper equation (8.4) `A_Θ/B_Θ`** is off by ~20× in the source text used.
  `_sv_autocorrelation_correction` instead solves the equivalent affine
  variational ODE (`_sv_conditional_expectation`). This is what makes the SV
  `u` match Table 2 (0.6621 vs 0.6703).
- **`u` convention.** Sepp's text subtracts the opening-trade cost, its Table 2
  does not. `SeppStats.u` includes it (matches the rollout); `SeppStats.u_raw`
  is the paper's value. Do not "fix" one to the other.
- **Statistics-vs-what.** `pytest` covers the Sepp formulas; it does **not**
  cover `scripts/comparison_study.py` or `scripts/deep_hedging_replication.py`
  beyond smoke tests. If you add a bug there, tests will not catch it.
- **Overfitting rule.** `comparison_study.py` selects Optuna trials on
  `SEARCH_SEEDS`, disjoint from train/val/tune/holdout. Keep it that way.
- **`.auto/`** still holds the original autoresearch session (best run: `alpha
  = 0.65`, `edge_mean +0.0297`, `win_rate 1.0`). The benchmark is frozen; do not
  re-run the optimisation loop against it while finishing the papers.

## Uncommitted file inventory

```
 M hedging/deep.py                 (additive helper only)
 M pyproject.toml                  (optuna, via uv add)
 M uv.lock
?? AGENTS.md
?? hedging/sepp.py
?? tests/test_sepp.py
?? scripts/sepp_replication.py
?? scripts/deep_hedging_replication.py
?? scripts/comparison_study.py
?? scripts/build_notebooks.py
?? notebooks/01_sepp_replication.ipynb
?? notebooks/02_deep_hedging_heston.ipynb
?? results/sepp/
?? results/figures/                (sepp_*.png)
?? results/deep_hedging/           (quick smoke-test output)
?? results/comparison/             (quick smoke-test output)
```
