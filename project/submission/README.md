# DSC4107 final submission

**Classic hedging is all you need** — classical versus learned option hedging under
transaction costs. Replications of Sepp (2013) and Buehler et al. (2019) with a
head-to-head comparison written up in [`report/report.pdf`](report/report.pdf).

## Contents

```
submission/
├── README.md              this manifest
├── report/
│   ├── report.typ         Typst source of the written report
│   ├── report.pdf         12-page report, main body under 2000 words
│   └── figures/           figures embedded in the report
├── slides/
│   ├── slides.typ         Typst source of the presentation
│   ├── slides.pdf         13-slide deck (16:9), about 10 minutes
│   └── figures/           slide-only figures
├── code/                  source snapshot (runnable)
└── data/                  experimental data and figures
```

`code/` is a snapshot of the GitHub tree: the `hedging/` package, `scripts/`,
`archive/scripts/`, `experiments/` (including the RQ results cited in report
§4.1), the four notebooks and the unit tests (25; the repository's three extra
hand-in checks test this folder and are left out). The frozen band is
`bench/classical_best.json`. The reproduction commands below also run from
inside `code/`.

## Entry points

| Item | File |
| --- | --- |
| Written report | `report/report.pdf` |
| Presentation slides | `slides/slides.pdf` |
| Source code (GitHub) | https://github.com/hluaguo/dsc4107 |
| Source code (snapshot) | `code/` |
| Experimental data | `data/` |

The report main body is under 2000 words; references and appendices are
excluded. Appendix A holds the extended tables and figures, Appendix B links the
code repository, and Appendix C lists the reproduction commands.

## Rebuilding the documents

Typst 0.15 or newer is required. From this directory:

```bash
# written report
typst compile --root . report/report.typ report/report.pdf

# slides
typst compile --root . slides/slides.typ slides/slides.pdf
```

The slide figures are redrawn with plain-language labels from committed results
(nothing is retrained), from the repository root:

```bash
uv run python scripts/plot_slide_figures.py
```

The report reads its figures from `report/figures/` and does not read the code
tree, so it compiles from this package alone.

## Reproducing the results

The full pipeline runs from the repository root:

```bash
uv sync
uv run pytest -q                                          # 28 checks
uv run python scripts/sepp_replication.py                 # Sepp tables + figures
uv run python scripts/sepp_replication.py --mc --mc-paths 60000
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3
uv run python archive/scripts/comparison_study.py --fixed-hyper \
    --costs 0.0 0.002 0.005 0.01 0.02
uv run python archive/scripts/comparison_study.py --models heston jd \
    --costs 0.002 0.005 0.01 --trials 8
uv run python scripts/holdout_eval.py                     # final locked run
```

Training scripts select CUDA when present and fall back to the CPU. Raw results
are mirrored under `data/`.

## Notes

- Source of truth for code and committed experiment outputs: the public GitHub repo
  (root [`README.md`](../README.md),
  [`experiments/`](../experiments/)). This folder adds PDFs and a `data/` mirror.
- Holdout paths were simulated once, after choices were frozen, in
  `scripts/holdout_eval.py` only.
- Narrative spine (papers → frozen 20 bp cell → classical vs deep at 20 bp → liquid
  10 bp footnote only): root [`README.md`](../README.md), [`report/report.pdf`](report/report.pdf),
  and [`experiments/README.md`](../experiments/README.md). PDF title matches
  the locked headline in `report/report.pdf`.
