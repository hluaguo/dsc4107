# Notebooks (grader entry)

Four notebooks tell the course story top to bottom. By default they call
`hedging/` and load committed artifacts under `submission/data/` and
`experiments/*/results/` — no multi-hour Optuna or full training unless you
opt in (notebook 02 smoke flag).

| # | Notebook | Story |
| --- | --- | --- |
| 01 | [`01_sepp_replication.ipynb`](01_sepp_replication.ipynb) | Sepp (2013) Section 5; MC uses **k/2** per eq. (2.5) |
| 02 | [`02_deep_hedging.ipynb`](02_deep_hedging.ipynb) | Buehler Heston replication from committed JSON |
| 03 | [`03_classical_vs_deep.ipynb`](03_classical_vs_deep.ipynb) | Frozen bench, 4×5 edge map, holdout display |
| 04 | [`04_research_path.ipynb`](04_research_path.ipynb) | RQ-A headline + liquid 10 bp footnote chain |

Regenerate `.ipynb` files after editing cell sources:

```bash
uv run python scripts/build_notebooks.py
```

Script entry points (same numbers as `REPORT.md`): `scripts/sepp_replication.py`,
`scripts/deep_hedging_replication.py`, `scripts/rq_a_pareto.py`, holdout
`scripts/holdout_eval.py` only.
