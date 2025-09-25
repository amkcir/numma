"""Optimization helpers. 数值优化辅助模块。"""

from __future__ import annotations

from .rootfinder import sortedroots  # Consistent re-export of the public API (统一导出对外 API)

__all__ = ["sortedroots"]

