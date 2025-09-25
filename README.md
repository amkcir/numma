# numma

Common numerical tools for scientific and simulation workflows.
面向科学与仿真工作的常用数值工具集。

## Features 功能亮点
- Root-finding utilities based on SciPy (基于 SciPy 的求根工具)
- Clean ``src`` layout for packaging best practices (符合最佳实践的 ``src`` 目录结构)

## Installation 安装
```bash
pip install numma
```

## Development 开发指南
1. Create a virtual environment and install the project in editable mode:
   创建虚拟环境并以可编辑模式安装项目:
   ```bash
   pip install -e .[dev]
   ```
2. Run the test suite 运行测试集:
   ```bash
   pytest
   ```

## Versioning 版本控制
The project relies on ``setuptools_scm`` to derive versions from Git tags.
项目通过 ``setuptools_scm`` 从 Git 标签推导版本号。
Generate release artifacts directly from the tagged commit to avoid dev suffixes.
从打好标签的提交生成发版产物, 以避免开发版后缀。
