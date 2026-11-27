"""Stable CLI entrypoint; implementation lives under ``experiments/rq_mandate/``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.rq_mandate.run_mandate import main

if __name__ == "__main__":
    main()
