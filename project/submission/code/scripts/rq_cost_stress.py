"""Stable CLI entrypoint; implementation lives under ``experiments/rq_cost_stress/``."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rq_cost_stress.run_cost_stress import main

if __name__ == "__main__":
    main()
