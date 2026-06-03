"""
evalscope_ext — Cerebras benchmark pruning extension for evalscope.

Importing this package registers two new benchmark datasets:
  - ``live_code_bench_pruned``  (difficulty-stratified subset of LCB v5)
  - ``aa_lcr_pruned``           (difficulty + source-count stratified subset of AA-LCR)

Usage:
    import evalscope_ext  # registers benchmarks as a side effect
    from evalscope import run_task
    ...
"""

from . import benchmarks  # noqa: F401 — triggers registration of pruned benchmarks

__version__ = "0.1.0"
__all__ = ["benchmarks"]