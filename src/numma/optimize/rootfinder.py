"""Root finding helpers. 根寻找工具集。"""

from __future__ import annotations

from collections.abc import Callable
from typing import List, cast

from scipy.optimize import brentq

def sortedroots(func: Callable[[float], float], start: float, stop: float, step: float) -> List[float]:
    """Return sorted real roots detected via sign changes.

    Scan the interval with a fixed step, bracket sign switches, and refine each root with
    ``scipy.optimize.brentq``. 扫描区间并用固定步长寻找符号变化, 再用 ``brentq`` 精修根。
    """
    if step <= 0:
        raise ValueError('step must be positive (步长必须为正数)')
    if start >= stop:
        raise ValueError('start must be smaller than stop (起始点必须小于终点)')

    roots: List[float] = []
    left = start
    while left < stop:
        right = min(left + step, stop)
        try:
            f_left = func(left)
            f_right = func(right)
        except Exception as exc:
            raise RuntimeError('function evaluation failed (函数求值失败)') from exc

        if f_left == 0.0:
            roots.append(left)
        elif f_right == 0.0:
            roots.append(right)
        elif f_left * f_right < 0:
            try:
                root = cast(float, brentq(func, left, right))  # narrow SciPy union type (收窄 SciPy 返回的联合类型)
            except ValueError:
                # Skip intervals where Brent fails to converge (若 Brent 失败则跳过该区间)
                pass
            else:
                roots.append(root)

        left = right

    roots.sort()
    return roots



