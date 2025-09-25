"""Regression tests for optimization helpers. 优化模块的回归测试。"""

from __future__ import annotations

import numpy as np

from numma.optimize import sortedroots

def test_sortedroots_on_cos() -> None:
    """Validate root finding on cos(x) - x.

    Compare to the known fixed point of ``cos(x)``. 对比 ``cos(x)`` 的已知不动点。
    """
    def func(x: float) -> float:
        return float(np.cos(x) - x)

    roots = sortedroots(func, -10.0, 10.0, 0.5)
    expected = [0.7390851332151607]
    for root, reference in zip(roots, expected):
        assert abs(root - reference) < 1e-5, f"Expected {reference}, got {root}"

#just to update