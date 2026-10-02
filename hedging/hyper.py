"""Training hyperparameters for the learned hedger.

These are the dials the experiment is allowed to turn, kept apart from
``bench.BenchConfig`` on purpose. The benchmark definition (market, contract,
seed pools, cost level) has to stay fixed for the comparison to mean anything,
while the optimiser is free to search the training setup.

``torch_seeds`` carries more than one value so every reported number averages
over training randomness. A single seed would let a lucky initialisation pass as
a real improvement, which is the failure mode this harness exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hyper:
    epochs: int = 200
    batch_size: int = 1024
    lr: float = 1e-3
    hidden: int = 128
    depth: int = 3
    alpha: float = 0.2
    torch_seeds: tuple[int, ...] = (0, 1, 2)
