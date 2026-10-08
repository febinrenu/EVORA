"""A stand-in for the `lap` package, built on scipy, used only when `lap` is not installed.

Ultralytics' trackers import `lap` at module level and, when it is missing, try to `pip install` it. A `uv` environment has
no pip, so tracking would crash on first use. The trackers need exactly one function, `lap.lapjv(cost, extend_cost, cost_limit)`;
this module provides it with the same semantics (assignments costlier than `cost_limit` are rejected) using
`scipy.optimize.linear_sum_assignment`, and registers itself as `lap` before Ultralytics is imported.
"""
from __future__ import annotations

import sys
import types

import numpy as np

_FORBIDDEN = 1e9


def lapjv(cost: np.ndarray, extend_cost: bool = False, cost_limit: float = float("inf")):
    """Returns (total cost, x, y): x[i] is the column assigned to row i (-1 if none), y[j] the row of column j."""
    from scipy.optimize import linear_sum_assignment

    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    x = np.full(n, -1, dtype=np.intp)
    y = np.full(m, -1, dtype=np.intp)
    if n == 0 or m == 0:
        return 0.0, x, y
    if np.isfinite(cost_limit):
        # rows and columns may stay unassigned at a price of cost_limit / 2 each, so a pair costing more than
        # cost_limit is never worth assigning
        half = cost_limit / 2.0
        big = np.full((n + m, n + m), _FORBIDDEN)
        big[:n, :m] = np.minimum(cost, _FORBIDDEN)
        big[np.arange(n), m + np.arange(n)] = half
        big[n + np.arange(m), np.arange(m)] = half
        big[n:, m:] = 0.0
        rows, cols = linear_sum_assignment(big)
        pairs = [(r, c) for r, c in zip(rows, cols, strict=True) if r < n and c < m]
    else:
        rows, cols = linear_sum_assignment(np.minimum(cost, _FORBIDDEN))
        pairs = list(zip(rows, cols, strict=True))
    total = 0.0
    for r, c in pairs:
        x[r], y[c] = c, r
        total += cost[r, c]
    return float(total), x, y


def install_if_missing() -> bool:
    """Register the stand-in as `lap` unless a real, working `lap` is importable. Returns True when the stand-in is active."""
    try:
        import lap

        if getattr(lap, "__version__", None) and hasattr(lap, "lapjv"):
            return False
    except (ImportError, AttributeError):
        pass
    shim = types.ModuleType("lap")
    shim.__version__ = "0.0.0+evora-shim"
    shim.lapjv = lapjv
    sys.modules["lap"] = shim
    return True
