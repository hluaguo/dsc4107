"""Exploration: can a larger search close the gap where the network loses?

The plan's Optuna add-on used eight trials. Two cells are the weak spots: Heston
and jump-diffusion at 20 bp, where the learned hedger only ties or trails the
tuned classical band. A weak result at eight trials could mean the family is
genuinely not better, or simply that the search was too small. This runs a
larger study (30 trials) on those two cells to find out.

Trials are ranked on the same disjoint ``SEARCH_SEEDS`` pool the plan used, and
the final configuration is reported on ``val_seeds``, so the answer is still
validation-free. The holdout pool is never touched.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import replace

ROOT = pathlib.Path(__file__).resolve().parents[2]
ARCH_SCRIPTS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ARCH_SCRIPTS))

import optuna  # noqa: E402

import comparison_study as C  # noqa: E402
from hedging import bench as B  # noqa: E402
from hedging.hyper import Hyper  # noqa: E402

OUT = ROOT / "results" / "exploration"
CELLS = (("heston", 0.002), ("jd", 0.002))
N_TRIALS = 30


def run_cell(model: str, cost: float, n_trials: int) -> dict:
    cfg = replace(B.BenchConfig(), model=model, cost=float(cost))
    classical = B.ClassicalStrategy(cfg, C.tune_classical(cfg))

    def objective(trial: optuna.Trial) -> float:
        hyper = Hyper(
            epochs=80,
            batch_size=1024,
            lr=float(trial.suggest_float("lr", 1e-4, 3e-3, log=True)),
            hidden=int(trial.suggest_int("hidden", 32, 256, step=32)),
            depth=int(trial.suggest_int("depth", 2, 4)),
            alpha=float(trial.suggest_float("alpha", 0.2, 0.9)),
            torch_seeds=(0,),
        )
        nets, _ = B.train_ensemble(cfg, hyper, device="cuda")
        return C.deep_edge(cfg, classical, nets, C.SEARCH_SEEDS, "cuda")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=2024),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    final_hyper = Hyper(
        epochs=80,
        lr=float(best["lr"]),
        hidden=int(best["hidden"]),
        depth=int(best["depth"]),
        alpha=float(best["alpha"]),
        torch_seeds=(0, 1, 2),
    )
    nets, _ = B.train_ensemble(cfg, final_hyper, device="cuda")
    result = C.report(cfg, classical, nets, cfg.val_seeds, "cuda")
    result.update(
        {
            "model": model,
            "cost": float(cost),
            "n_trials": n_trials,
            "search_edge": study.best_value,
            "hyper": best,
        }
    )
    print(
        f"{model:>6} {cost:<6} trials={n_trials} search={study.best_value:+.4f} "
        f"val_edge={result['edge_mean']:+.4f} win={result['win_rate']:.2f} "
        f"deep={result['deep_sharpe']:+.3f} class={result['classical_sharpe']:+.3f} "
        f"deep_std={result['deep_std']:.3f} hyper={best}"
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [run_cell(model, cost, N_TRIALS) for model, cost in CELLS]
    (OUT / "optimise_large.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}/optimise_large.json")


if __name__ == "__main__":
    main()
