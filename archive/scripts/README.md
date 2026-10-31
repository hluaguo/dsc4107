# Archived scripts

Exploratory and historical one-offs moved off the grader path. Notebooks load
committed JSON under `submission/data/` and `experiments/*/results/` instead.

| Script | Role |
| --- | --- |
| `comparison_study.py` | 4×5 Sharpe edge map + Optuna add-on (`REPORT.md` §3) |
| `explore_*.py` | Objective, features, transfer, robustness probes |
| `diagnose_jd.py` | JD `q`-factor isolation for Sepp Table 2 |
| `sidequest_participation.py` | Participation frontier side quest |

Re-run from repo root, e.g. `uv run python archive/scripts/comparison_study.py --fixed-hyper`.
