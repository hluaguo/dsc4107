# Project plan: replications, frozen benchmark, and RQ-A

## Lean scope

The written report (`REPORT.md`) already holds the replication numbers and the exploratory 4×5 comparison / Optuna edge map. Those runs stay in the repository as historical evidence, but they are no longer the *main* research story. The course project now centres on:

1. **Sepp (2013) replication** — closed-form discrete-hedging statistics and Section 5 optima checked against Monte Carlo (`hedging/sepp.py`, `scripts/sepp_replication.py`, notebook 01).
2. **Buehler et al. (2019) Deep Hedging replication** — Heston experiment with proportional costs, frictionless delta check (`scripts/deep_hedging_replication.py`, notebook 02).
3. **One consolidated frozen benchmark** — `BenchConfig`: Heston ATM one-month call, proportional cost 0.002 (20 bp), `n_steps=21`; classical baseline pinned in `bench/classical_best.json` (band champion at `delta_band=0.1`).
4. **Original RQ-A** — the weekend Pareto experiment below (`scripts/rq_a_pareto.py`).

## Research questions

| ID | Question |
| --- | --- |
| RQ-S | Can Monte Carlo reproduce Sepp's formulas, tables and figures? |
| RQ-D | Can we reproduce Deep Hedging's main Heston claim (learned hedger vs Black–Scholes delta under costs)? |
| **RQ-A** | On the frozen benchmark cell, is the unconstrained deep hedger weakly dominated by the classical **cost–risk** frontier spanned by calendar delta hedging and band rules (including the pinned baseline)? |

Prior work differs in what it optimises. Sepp (2013), DOI [10.21314/jois.2013.023](https://doi.org/10.21314/jois.2013.023), maximises Sharpe over frequency and bands in analytic form. Leland (1985, *Journal of Finance*) and Ahn & Wilmott (DOI [10.1002/wilj.9](https://doi.org/10.1002/wilj.9)) adjust discrete classical hedges for costs. Li (2025, ICFTBA symposium — classical frequency sweeps only, no deep hedger) sits nearby but does not ask whether a Buehler-style network clears a Sepp-style classical **hull** that already contains the frozen band champion. RQ-A uses two objectives aligned with tail-aware hedging rather than Sharpe alone: **mean path proportional transaction cost** and **CVaR95 of terminal P&L** (5% tail loss from `hedging/metrics.py`).

## RQ-A: cost–CVaR Pareto on the frozen cell

**Setting.** Fixed `BenchConfig()`, shared out-of-sample paths = every `val_seeds` × `n_val_paths` path (same paths for every policy). Classical candidates (~10+ points):

- `TimeBasedDelta` with frequency ∈ {1, 3, 7, 21} (with `n_steps=21`);
- `BandDelta` with `delta_band` ∈ {0.05, 0.10, 0.20};
- `BandDelta` with `price_band` ∈ {0.01, 0.02, 0.05};
- plus the pinned `load_classical_best()` / `ClassicalStrategy` point.

**Learned policy.** Unconstrained deep hedger with default `Hyper()` (all `torch_seeds`), scored per seed and as the usual ensemble average.

**Accounting.** Mean cost from `RolloutResult.costs`; CVaR95 from pooled terminal P&L. Use `hedge.rollout`, `ClassicalStrategy`, and `bench.train_ensemble` / `NetworkStrategy` — no parallel P&L definitions.

**Runner.** `uv run python scripts/rq_a_pareto.py` writes JSON under `results/rq_a/`. Use `--quick` for a cheap smoke (mirrors `bench._quick_setup`).

**Acceptance readout** (automated in the script):

| Verdict | Criterion |
| --- | --- |
| `dominated` | Every deep seed (and the ensemble) is weakly dominated by at least one classical point: no seed lies strictly northwest of the classical scatter in the (mean cost, CVaR95) plane (lower is better on both axes). |
| `improves` | At least two of three training seeds beat `classical_best` on CVaR95 while mean cost is ≤ the baseline's mean cost (CVaR is primary). |
| `flat` | Otherwise. |

Full numerical results for the course report remain in `REPORT.md`; this plan only fixes the forward-looking question and how to re-run the cell.

## Replications (unchanged intent)

### Sepp (2013)

Simulator with proportional costs on four return processes; analytic P&L statistics checked against Appendix A; Section 5 Sharpe-maximising frequency and band results reproduced. Formulas are the reference, not the Monte Carlo engine.

### Buehler et al. (2019)

PyTorch hedger minimising a convex risk measure of terminal P&L; Heston replication with proportional costs. The frictionless run must recover Black–Scholes delta within tolerance or the implementation is wrong.

## Evaluation guard rails

- Frictionless deep hedger ≈ Black–Scholes delta.
- Sepp closed forms match Monte Carlo within simulation error (`tests/test_sepp.py`).
- Reported learned numbers average over `Hyper.torch_seeds`; holdout seeds are only read in `scripts/holdout_eval.py`.

## Deliverables

1. Simulator library and Sepp analytic layer with tests.
2. Notebook / script outputs for Sepp Section 5 and Deep Hedging Heston.
3. Frozen benchmark iteration (`scripts/benchmark_loop.py`) and pinned classical baseline.
4. **RQ-A** weekend script and `results/rq_a/` artifacts.
5. Written report (`REPORT.md`) covering replications and findings.

## Risks

Sepp's appendix is dense; we verify by simulation rather than re-derive. Deep hedging numbers carry training variance — multi-seed reporting and the frictionless check bound that uncertainty. All data are simulated; there is no external dataset risk.
