"""Sidequest: how much of the learned hedger's edge is just "hedge less"?

The comparison found large high-cost Sharpe edges whose P&L standard deviation
grows with cost: the network stops hedging and collects the volatility risk
premium. The classical family used there cannot express that, because every
candidate opens a full delta at ``t = 0``. This script adds the missing degree
of freedom so the interpretation can be tested instead of asserted.

A participation strategy holds ``lam * BS_delta`` and rebalances every
``frequency`` steps. ``lam = 1`` is the textbook time-scheduled delta, ``lam = 0``
is "sell and do nothing", and everything between is a partial hedge. If a single
scalar ``lam``, tuned on the same seed pool as every other classical baseline,
captures the network's high-cost Sharpe, then the network's extra freedom is
worth little beyond a leverage choice and the high-cost "win" is confirmed as
premium harvesting. If the network stays above the whole participation frontier,
it is doing something the scalar cannot.

This is a side probe. It imports the frozen modules read-only, reads the frozen
comparison output, and writes only under ``results/sidequest/``; it does not
change the benchmark, the plan or any earlier result.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hedging import bench as B  # noqa: E402
from hedging import bs, hedge, metrics  # noqa: E402

OUT = ROOT / "results" / "sidequest"
FIXED = ROOT / "results" / "comparison" / "edge_map_fixed.json"

MODELS = ("gbm", "jd", "heston", "svj")
COSTS = (0.0, 0.002, 0.005, 0.01, 0.02)

# A coarse grid is enough: the question is whether a partial hedge captures the
# edge at all, not the third decimal of the optimal ratio. 1.25 is included so a
# levered delta can win if it wants to, which makes the test conservative for
# the "under-hedging" story.
LAMS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25)


class ParticipationDelta:
    """Hold ``lam`` times the Black-Scholes delta on a fixed calendar.

    The frequency dial is kept because under-hedging and trading less often are
    two ways to pay less cost, and the comparison's network could in principle
    use either. Giving the scalar both makes it a stronger competitor and the
    test of the story correspondingly harder to pass.
    """

    def __init__(self, K, T, r, sigma_impl, kind, dt, frequency, lam):
        self.K = K
        self.T = T
        self.r = r
        self.sigma_impl = sigma_impl
        self.kind = kind
        self.dt = dt
        self.frequency = int(frequency)
        self.lam = float(lam)

    def __call__(self, j, spot, pos):
        if j % self.frequency:
            return pos
        tau = max(self.T - j * self.dt, 0.0)
        target = bs.bs_delta(spot, self.K, tau, self.r, self.sigma_impl, self.kind)
        return self.lam * target


def sharpe_on(cfg, paths, lam, frequency) -> float:
    fn = ParticipationDelta(
        cfg.K, cfg.T, cfg.r, cfg.sigma_impl, cfg.kind, cfg.dt, frequency, lam
    )
    result = hedge.rollout(
        paths, fn, cfg.K, cfg.kind, cfg.premium, cfg.cost, cfg.dt
    )
    return float(metrics.sharpe(result.pnl))


def load_deep_sharpes() -> dict:
    rows = json.loads(FIXED.read_text())
    return {(r["model"], float(r["cost"])): r for r in rows}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed = load_deep_sharpes()
    base = B.BenchConfig()

    rows = []
    frontier: dict[tuple[str, float], dict] = {}
    for model in MODELS:
        # Paths do not depend on cost, so simulate once per (model, seed) and
        # reuse across the whole cost grid. The alternative re-simulates 20 times
        # over and buys nothing but noise.
        tune_paths = [
            base.simulate(model, s, base.n_val_paths) for s in base.tune_seeds
        ]
        val_paths = [base.simulate(model, s, base.n_val_paths) for s in base.val_seeds]

        for cost in COSTS:
            cfg = replace(base, model=model, cost=float(cost))
            scored = []
            for lam in LAMS:
                for freq in B.FREQUENCIES:
                    mean_tune = float(
                        np.mean([sharpe_on(cfg, p, lam, freq) for p in tune_paths])
                    )
                    scored.append((mean_tune, lam, freq))
            scored.sort(key=lambda t: t[0], reverse=True)
            _, best_lam, best_freq = scored[0]
            best_val = float(
                np.mean([sharpe_on(cfg, p, best_lam, best_freq) for p in val_paths])
            )
            no_hedge = float(np.mean([sharpe_on(cfg, p, 0.0, 1) for p in val_paths]))
            ref = fixed[(model, cost)]
            row = {
                "model": model,
                "cost": float(cost),
                "best_lam": best_lam,
                "best_freq": best_freq,
                "participation_sharpe": best_val,
                "deep_sharpe": ref["deep_sharpe"],
                "classical_sharpe": ref["classical_sharpe"],
                "no_hedge_sharpe": no_hedge,
                "deep_minus_participation": ref["deep_sharpe"] - best_val,
            }
            rows.append(row)
            # Full-frequency frontier, only for the plot: one curve per model and
            # cost so the reader can see whether the network sits above it.
            frontier[(model, float(cost))] = {
                lam: float(np.mean([sharpe_on(cfg, p, lam, 1) for p in val_paths]))
                for lam in LAMS
            }
            print(
                f"{model:>6} {cost:<6} lam*={best_lam:<4} freq*={best_freq:<3} "
                f"part={best_val:+.3f} deep={ref['deep_sharpe']:+.3f} "
                f"class={ref['classical_sharpe']:+.3f} nohedge={no_hedge:+.3f} "
                f"deep-part={row['deep_minus_participation']:+.3f}"
            )

    (OUT / "participation.json").write_text(json.dumps(rows, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for ax, model in zip(axes.ravel(), MODELS):
        for cost, style in zip(COSTS, ("-", "--", "-.", ":", (0, (3, 1, 1, 1)))):
            fr = frontier[(model, float(cost))]
            xs = list(LAMS)
            ys = [fr[x] for x in xs]
            ax.plot(
                xs, ys, linestyle=style, marker=".", ms=4, label=f"{cost * 1e4:.0f} bp"
            )
            deep = fixed[(model, float(cost))]["deep_sharpe"]
            ax.axhline(deep, color=ax.lines[-1].get_color(), lw=0.6, alpha=0.5)
        ax.set_title(model)
        ax.set_xlabel("participation lambda")
        ax.set_ylabel("Sharpe (val)")
        ax.legend(fontsize=7)
    fig.suptitle(
        "Participation frontier (lines) versus learned hedger (thin horizontal lines)"
    )
    fig.tight_layout()
    path = OUT / "participation_frontier.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"\nwrote {path.relative_to(ROOT)} and participation.json")


if __name__ == "__main__":
    main()
