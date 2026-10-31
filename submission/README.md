# DSC4107 final submission

Classical versus learned option hedging under transaction costs. Replications of
Sepp (2013) and Buehler et al. (2019) with a head-to-head comparison.

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
│   └── slides.pdf         11-slide deck (16:9)
├── code/                  source snapshot (the same tree is on GitHub)
└── data/                  experimental data and figures
```

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

The report reads its figures from `report/figures/` and does not read the code
tree, so it compiles from this package alone.

## Reproducing the results

The full pipeline runs from the repository root:

```bash
uv sync
uv run pytest -q                                          # 24 checks
uv run python scripts/sepp_replication.py                 # Sepp tables + figures
uv run python scripts/sepp_replication.py --mc --mc-paths 60000
uv run python scripts/deep_hedging_replication.py --epochs 100 --hidden 64 --depth 3
uv run python scripts/comparison_study.py --fixed-hyper \
    --costs 0.0 0.002 0.005 0.01 0.02
uv run python scripts/comparison_study.py --models heston jd \
    --costs 0.002 0.005 0.01 --trials 8
uv run python scripts/holdout_eval.py                     # final locked run
```

Training scripts select CUDA when present and fall back to the CPU. Raw results
are mirrored under `data/`.

## Notes

- Source of truth for code and committed experiment outputs: the public GitHub repo
  (root [`README.md`](../README.md), [`REPORT.md`](../REPORT.md),
  [`experiments/`](../experiments/)). This folder adds PDFs and a `data/` mirror.
- Holdout paths were simulated once, after choices were frozen, in
  `scripts/holdout_eval.py` only.

## 研究脈絡（简报）

与仓库根目录 [`README.md`](../README.md) 中文段一致：Sepp (2013) 与 Buehler et al. (2019)
复现 → 冻结 Heston ATM 1M / 20 bp cell 上 classic vs deep → 主结论 **Classic hedging is all you need**
（见 `report/report.pdf` 与 [`REPORT.md`](../REPORT.md)）。Liquid 10 bp 实验链仅作 brittle 脚注，
详见 [`experiments/README.md`](../experiments/README.md)。
