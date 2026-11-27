"""Classical versus learned option hedging: replication and comparison.

The package is split so that the benchmark harness and market simulators are
frozen, while the strategy under test lives in ``deep`` (and, for the
classical replication, in ``hedge``).
"""

from . import bs, market, hedge, deep, hyper, metrics, bench

__all__ = ["bs", "market", "hedge", "deep", "hyper", "metrics", "bench"]
