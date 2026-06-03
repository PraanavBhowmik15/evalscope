"""
evalscope_ext/pruning/strategy.py

DifficultyDiversityPruner — the core pruning algorithm.

Loads pre-computed index lists from bundled JSON files (generated offline
by scripts/compute_lcb_indices.py and scripts/compute_aalcr_indices.py).

Generalises to unseen models because selection criteria are problem
properties (difficulty bucket estimated from pass rates, message length,
source-document count) — not any individual model's outputs.
"""

import json
import os
from typing import Optional

_INDICES_DIR = os.path.join(os.path.dirname(__file__), "indices")

_DATASET_TO_FILE = {
    "live_code_bench":         "lcb_indices.json",
    "live_code_bench_pruned":  "lcb_indices.json",
    "aa_lcr":                  "aalcr_indices.json",
    "aa_lcr_pruned":           "aalcr_indices.json",
}


class DifficultyDiversityPruner:
    """
    Selects a representative subset via difficulty-stratified diversity sampling.

    Usage::

        pruner = DifficultyDiversityPruner(dataset_name="live_code_bench", prune_ratio=0.1)
        keep = pruner.index_set()   # set of int IDs to keep
        filtered = [s for s in samples if s["id"] in keep]
    """

    def __init__(self, dataset_name: str, prune_ratio: float = 0.1,
                 strategy: str = "difficulty_diversity"):
        self.dataset_name = dataset_name
        self.prune_ratio  = prune_ratio
        self.strategy     = strategy
        self._indices: Optional[list] = None
        self._meta: dict = {}
        self._load()

    def _load(self) -> None:
        fname = _DATASET_TO_FILE.get(self.dataset_name)
        if not fname:
            raise ValueError(
                f"No pre-computed indices for dataset '{self.dataset_name}'. "
                f"Known datasets: {list(_DATASET_TO_FILE)}"
            )
        path = os.path.join(_INDICES_DIR, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Pruning index file not found: {path}. "
                "Run scripts/compute_lcb_indices.py or scripts/compute_aalcr_indices.py first."
            )
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._meta = data

        # Find the closest available ratio
        available = sorted(float(k) for k in data["indices"])
        closest = min(available, key=lambda r: abs(r - self.prune_ratio))
        if abs(closest - self.prune_ratio) > 0.01:
            import warnings
            warnings.warn(
                f"prune_ratio={self.prune_ratio} not found; using closest={closest}. "
                f"Available ratios: {available}"
            )
        self._indices = data["indices"][str(closest)]

    def index_set(self) -> set:
        """Return the set of sample IDs (0-based row indices) to keep."""
        return set(self._indices)

    @property
    def n_selected(self) -> int:
        return len(self._indices) if self._indices else 0

    @property
    def total_samples(self) -> int:
        return self._meta.get("total_samples", -1)

    def __repr__(self) -> str:
        return (f"DifficultyDiversityPruner(dataset={self.dataset_name!r}, "
                f"prune_ratio={self.prune_ratio}, "
                f"selected={self.n_selected}/{self.total_samples})")