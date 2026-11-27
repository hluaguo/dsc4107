"""Stable CLI entrypoint; implementation lives under ``experiments/rq_hard_cvar/``."""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.rq_hard_cvar.run_hard_cvar import main

if __name__ == "__main__":
    main()
