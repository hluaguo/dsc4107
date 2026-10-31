# Agent notes

## Documentation and comments: explain WH Y, not WHAT

Every docstring and every inline comment in this repository must explain the
*reasoning*, not restate the code. A reader can see what a line does by reading
it; what they cannot recover is why it is there. So:

- **Why this function exists** and what decision it encodes — not "returns the
  Sharpe ratio" but "the Sharpe ratio is the arbitrating score because Sepp
  maximises it and the comparison in this project is risk-adjusted P&L."
- **Why this approach** was chosen over the obvious alternative, and what the
  alternative would cost. Name the tradeoff.
- **Why this value / threshold / constant.** Every magic number, tolerance,
  truncation point, grid bound and quadrature limit needs a justification. If
  the value comes from a paper, cite the equation; if it comes from a measured
  constraint (runtime, memory, accuracy), say so.
- **Why the convention.** When two readings of a source disagree (e.g. an
  equation versus the table it generated, a per-trade cost versus a spread),
  state which one is implemented and how the choice was verified.
- **What was traded away.** Note the accuracy, generality or runtime that was
  given up for the choice.

Do not write filler such as "helper function", "computes X", "loop over
paths", or docstrings that simply expand the function name. Do not add a type
comment where the signature already carries it. Keep it short: one sentence of
why beats three of what.

## Dependency management

Never edit `pyproject.toml` by hand. Add dependencies with `uv add <pkg>` and
remove them with `uv remove <pkg>` so the lockfile, the environment and the
resolved versions stay in step. The only exceptions are non-dependency fields
(`name`, `description`, `readme`), and even those are better changed with
`uv` tooling when one exists.

## Project shape

- `hedging/market.py` — four simulators (GBM, JD, Heston, SVJ), frozen.
- `hedging/hedge.py` — the one rollout that turns positions into P&L, frozen.
- `hedging/sepp.py` — Sepp (2013) closed-form P&L statistics and Sharpe
  optimisation. The simulator is the ground truth; every formula here is
  checked against Monte Carlo in `tests/test_sepp.py`.
- `hedging/deep.py`, `hedging/hyper.py` — the learned hedger and its dials.
- `hedging/bench.py` — the frozen deep-vs-classical benchmark.
- `scripts/` — entry points; `notebooks/` — the two paper replications.

## Guard rails

- `hedging/market.py`, `hedging/bs.py`, `hedging/hedge.py`, `hedging/metrics.py`,
  `hedging/bench.py`, `bench/classical_best.json` and `tests/` are off limits for
  the optimisation loop. Do not weaken a check to make a number pass.
- Never read or simulate the holdout seed pool outside `scripts/holdout_eval.py`.
- Every reported number averages over `Hyper.torch_seeds`; do not shrink it.
