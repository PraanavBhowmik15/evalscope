"""
evalscope_ext — Cerebras benchmark pruning extension for evalscope.

Provides DifficultyDiversityPruner, PrunedMixin, and pre-computed index files.
Benchmark registration (live_code_bench_pruned, aa_lcr_pruned, mmmu_pruned) is
handled by the fork's auto-discovery mechanism in evalscope/benchmarks/.
"""

from evalscope_ext.pruning import DifficultyDiversityPruner, PrunedMixin  # noqa: F401

__version__ = "0.1.0"
__all__ = ["DifficultyDiversityPruner", "PrunedMixin"]