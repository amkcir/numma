"""numma core exports. 核心导出。"""

from __future__ import annotations

from importlib import metadata

# Attempt to read the installed distribution version (优先读取已安装发行版的版本号)
try:
    __version__ = metadata.version('numma')
except metadata.PackageNotFoundError:
    # Fallback to setuptools_scm when running from a source tree (源代码环境下回退到 setuptools_scm)
    try:
        from setuptools_scm import get_version  # type: ignore
    except Exception:
        __version__ = '0.0.0'  # Use a neutral placeholder when everything else fails (最坏情况下回退占位符)
    else:
        try:
            __version__ = get_version(root='../..', relative_to=__file__)
        except LookupError:
            __version__ = '0.0.0'

from .optimize import sortedroots  # Re-export optimisation helper (再次导出优化工具函数)

__all__ = ["__version__", "sortedroots"]

