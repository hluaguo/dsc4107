# Classical versus learned option hedging

CityU **DSC4107** — [hluaguo/dsc4107](https://github.com/hluaguo/dsc4107)

We replicate two papers on the same simulated desk (sell one ATM call, hedge with
the underlying, pay proportional transaction costs), then ask whether a neural
hedger beats a strong classical policy on a **fixed** benchmark cell.

**Title (locked):** **Classic hedging is all you need** at the course cell (Heston,
one-month ATM call, **20 bp** full-rate cost). Details and numbers: [`REPORT.md`](REPORT.md).
PDF for grading: [`submission/report/report.pdf`](submission/report/report.pdf).

## Quick start

```bash
uv sync
uv run pytest -q
```

Python version is pinned in `.python-version`. Scripts use GPU when available.

## Story spine (English)

1. **Papers.** Sepp (2013) — analytic discrete-hedging Sharpe, optimal frequency and
   bands. Buehler et al. (2019) — deep hedging with a CVaR-style objective under costs.
2. **Replications.** Sepp Section 5 tables/figures (`scripts/sepp_replication.py`);
   Buehler Heston experiment plus frictionless delta check
   (`scripts/deep_hedging_replication.py`). Reference PDFs: [`docs/papers/`](docs/papers/).
3. **Frozen cell.** `BenchConfig` in `hedging/bench.py`: Heston, ATM 1M call,
   `cost = 0.002` (20 bp **full-rate** ledger), 21 steps. Classical champion pinned in
   `bench/classical_best.json` (delta band 0.1).
4. **Classical vs deep on that cell.** Historical 4×5 comparison and holdout in
   [`REPORT.md`](REPORT.md) §2–3: at 20 bp the learned hedger does **not** beat the
   tuned band on Sharpe; RQ-A (`experiments/rq_a/`) shows the deep policy is **cheaper
   on mean cost but worse on CVaR95** versus the classical cost–tail hull → verdict
   `dominated`.
5. **Follow-up RQs (same cell, then liquid footnote).** Mandate, risk-matched α, train-time
   CVaR cap, cost stress, policy anatomy — indexed in
   [`experiments/README.md`](experiments/README.md). These sharpen *why* the headline
   holds; they do not replace it.
6. **Liquid 10 bp (footnote only).** Sepp Case I uses **k/2** in eq. (2.5); our simulator
   charges **full-rate** `cost`. Runs at `cost = 0.001` match Sepp’s **k** numerically but
   not Case I economics. A brittle soft path (do **not** promote to main claim):
   **H** `clears` (narrow) → **K** `dominated` → **L** `a05_only` → **I** `fails` → **J**
   `band_like`. See liquid rows in [`experiments/README.md`](experiments/README.md).

## 研究脈絡（给助教）

1. **基于哪两篇 paper。** Sepp (2013) 在比例成本下用解析式求离散对冲的 Sharpe、最优频率与
   band。Buehler et al. (2019) 用神经网络最小化终端 P&L 的 CVaR 类目标，在 Heston 等模型里
   与经典 delta 对冲比较。
2. **我们做了什么。** 在同一套模拟器里复现 Sepp §5 与 Buehler 的 Heston 实验（含零成本
   frictionless delta 自检），再在**冻结**的 `BenchConfig` 上比较经典规则与 deep hedger：
   Heston、ATM、一个月到期、`cost = 0.002`（账本为**全费率** 20 bp，不是 Sepp 的 k/2）。
3. **冻结 cell 上的结果。** 20 bp 下 tuned delta band（见 `bench/classical_best.json`）与
   deep 在 Sharpe 上接近或略优，但 deep 往往用更差的尾部换更低的平均成本；RQ-A 在
   （mean cost, CVaR95）平面上给出 `dominated`。完整表格与 holdout 见 [`REPORT.md`](REPORT.md)。
4. **由此推出的下一层问题（已跑完，见 `experiments/`）。** 在**不改动**主结论的前提下，
   我们追问：强制交易活跃度、训练 α 选择、训练期 CVaR 约束、多档成本、策略形态等能否让
   deep **同时**打败 baseline 的成本与尾部——索引见 [`experiments/README.md`](experiments/README.md)。
5. **标题结论（锁定）。** **Classic hedging is all you need** —— 主故事是 20 bp 冻结 cell
   上的经典 band，不是 liquid 上的 deep edge。
6. **Liquid 10 bp 仅作 brittle 脚注。** 账本仍是全费率；`cost=0.001` 只对齐 Sepp liquid **k**
   的数值，不是 Case I 的 k/2 经济学。脚注链：**H** 极窄 `clears` → **K** `dominated` → **L**
   `a05_only` → **I** `fails` → **J** `band_like`；勿写成「deep 全面优于经典」。

**代码与数据附录：** 本公开仓库即源码与结果路径；复现命令见 [`REPORT.md`](REPORT.md) §6，
已提交 JSON/图在 `experiments/*/results/` 与 [`submission/data/`](submission/data/)。

## Where to read

| Path | Role |
| --- | --- |
| [`REPORT.md`](REPORT.md) | Full report (replications, comparison, holdout) |
| [`experiments/README.md`](experiments/README.md) | RQ-A … RQ-L index and verdicts |
| [`submission/`](submission/) | PDF report, slides, mirrored artifacts |
| [`docs/papers/`](docs/papers/) | Sepp & Buehler PDFs |

Optional background notes (plan, extra probes) live under [`docs/`](docs/) and are not
required for grading.

## Layout

```
hedging/       simulators, Sepp formulas, classical + deep hedgers, bench config
bench/         pinned classical baseline (classical_best.json)
scripts/       replication and experiment entry points
experiments/   one folder per RQ + committed full-run results/
tests/         checks (frozen)
submission/    course PDFs and data mirror
```

Holdout seeds are used **only** in `scripts/holdout_eval.py`.

## Reproduce the headline cell

```bash
uv run python scripts/rq_a_pareto.py    # → experiments/rq_a/results/
uv run python scripts/benchmark_loop.py # → results/ (gitignored; see submission/data/)
```

Sepp / Buehler / 4×5 comparison commands: [`REPORT.md`](REPORT.md) §6.
