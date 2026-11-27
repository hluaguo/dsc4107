"""Slide figures with plain-language labels, redrawn from committed results.

Nothing is retrained: every number comes from JSON already in the repository.
"""

import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "submission" / "data"
OUT = ROOT / "submission" / "slides" / "figures"

NET = "#4c78a8"
BAND = "#e8833a"
DELTA = "#9d9d9d"
RED = "#b23a2e"
MODELS = {
    "gbm": ("Smooth (GBM)", "#4c78a8"),
    "jd": ("Jumps (JD)", "#e8833a"),
    "heston": ("Moving vol (Heston)", "#2e8b57"),
    "svj": ("Both (SVJ)", "#b23a2e"),
}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)


def frontier():
    data = json.loads(
        (ROOT / "experiments" / "rq_risk_matched" / "results" / "risk_matched.json").read_text()
    )
    classical = [r for r in data["classical"] if r["label"] != "classical_best"]
    deep = [r for r in data["deep"] if "_alpha_" in r["label"]]
    band = next(r for r in data["classical"] if r["label"] == "classical_best")

    fig, ax = plt.subplots(figsize=(8.0, 5.6), dpi=200)
    bx, by = band["mean_cost"], band["cvar95"]
    floor = min(r["cvar95"] for r in classical)
    x0, x1, y0, y1 = 0.33, 0.47, 0.90, 1.42

    ax.add_patch(plt.Rectangle((x0, y0), bx - x0, by - y0, color="#2e8b57", alpha=0.12, lw=0))
    ax.text(x0 + 0.004, y0 + 0.02, "cheaper AND safer\nthan the Sepp band:\nno network here",
            color="#1f6b42", fontsize=14, fontweight="bold", va="bottom")
    ax.hlines(floor, bx, x1, color=BAND, lw=1.5, ls="--")
    ax.text(x1 - 0.002, floor + 0.03, "safest classical rule", color="#c0651f",
            fontsize=13, ha="right")
    ax.annotate("only networks\nget this safe", xy=(0.4093, 0.9396), xytext=(0.437, 0.915),
                fontsize=13, color="#2f5d8a", va="bottom",
                arrowprops=dict(arrowstyle="->", color="#2f5d8a"))

    ax.scatter([r["mean_cost"] for r in deep], [r["cvar95"] for r in deep],
               s=55, color=NET, alpha=0.8, label="neural networks")
    ax.scatter([r["mean_cost"] for r in classical], [r["cvar95"] for r in classical],
               s=80, color=BAND, label="other classical rules")
    ax.scatter([bx], [by], s=420, marker="*", color=RED, zorder=5, label="Sepp band (benchmark)")

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_xlabel("trading cost  →  more expensive")
    ax.set_ylabel("tail loss (CVaR95)  →  riskier")
    ax.legend(loc="upper right", fontsize=13, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "frontier.png")
    plt.close(fig)


def edge_map():
    rows = json.loads((DATA / "comparison" / "edge_map_fixed.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), dpi=200, sharex=True, sharey=True)
    for ax, (key, (name, _)) in zip(axes.flat, MODELS.items()):
        pts = sorted(
            (r["cost"] * 1e4, r["deep_sharpe"], r["classical_sharpe"])
            for r in rows
            if r["model"] == key
        )
        x = [p[0] for p in pts]
        ax.axvspan(12, 28, color="#2e8b57", alpha=0.12, lw=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.plot(x, [p[2] for p in pts], "-o", color=BAND, lw=2.2, label="classical (best Sepp-style rule, re-tuned)")
        ax.plot(x, [p[1] for p in pts], "-o", color=NET, lw=2.2, label="neural network")
        ax.set_title(name, fontsize=14)
        ax.set_xticks([0, 20, 50, 100, 200])
        style(ax)
    for ax in axes[1]:
        ax.set_xlabel("trading fee (bp)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sharpe")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, fontsize=13,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.99, 0.005, "green = 20 bp benchmark", ha="right", va="bottom",
             fontsize=11, color="#1f6b42")
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(OUT / "edge_map.png")
    plt.close(fig)


def sharpe_vs_cost():
    res = json.loads((DATA / "deep_hedging" / "results.json").read_text())
    costs = [c["cost"] * 1e4 for c in res["costs"]]
    series = [
        ("deep_sharpe", "neural network", NET),
        ("band_sharpe", "Sepp band (tuned at 20 bp)", BAND),
        ("bs_delta_sharpe", "trade every day (textbook)", DELTA),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.4), dpi=200)
    for key, name, color in series:
        ys = [c["summary"][key]["mean"] for c in res["costs"]]
        ax.plot(costs, ys, "-o", color=color, label=name, lw=2.2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("trading fee (basis points; 100 bp = 1%)")
    ax.set_ylabel("Sharpe ratio  (higher is better)")
    ax.set_xticks(costs)
    ax.legend(loc="lower left", fontsize=13, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "sharpe_vs_cost.png")
    plt.close(fig)


def tail():
    d = json.loads((DATA / "exploration" / "distribution.json").read_text())
    groups = [("spread\n(standard deviation)", "std"), ("worst-5% loss\n(CVaR95)", "cvar95")]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
    w = 0.36
    for i, (_, key) in enumerate(groups):
        for j, (who, color, name) in enumerate(
            [("learned", NET, "neural network"), ("band", BAND, "Sepp band")]
        ):
            v = d[who][key]
            x = i + (j - 0.5) * w
            ax.bar(x, v, w * 0.95, color=color, label=name if i == 0 else None)
            ax.text(x, v + 0.02, f"${v:.2f}", ha="center", va="bottom", fontsize=14)
    ax.annotate("24% worse", xy=(1 - w / 2, d["learned"]["cvar95"] + 0.12), ha="center",
                fontsize=14, color=RED, fontweight="bold")
    ax.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=13)
    ax.set_ylabel("dollars per option")
    ax.set_ylim(0, 1.65)
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "tail.png")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({"font.size": 15})
    OUT.mkdir(parents=True, exist_ok=True)
    frontier()
    edge_map()
    sharpe_vs_cost()
    tail()


if __name__ == "__main__":
    main()
