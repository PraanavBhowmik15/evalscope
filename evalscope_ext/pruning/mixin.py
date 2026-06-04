"""
PrunedMixin — universal drop-in for any evalscope adapter.

Usage::

    class MyPrunedAdapter(PrunedMixin, MyBaseAdapter):
        _DATASET_NAME = 'my_dataset'   # key in pruning/indices/<name>_indices.json
"""

import logging

logger = logging.getLogger(__name__)


class PrunedMixin:
    """
    Mixin that adds difficulty-stratified pruning to any evalscope adapter.

    Reads ``prune_ratio`` and ``pruning_strategy`` from ``extra_params``,
    stamps a row counter onto ``sample.id`` during loading (because evalscope's
    ``reindex()`` runs *after* ``sample_filter``, so ``sample.id`` is None at
    filter time unless we set it here), and gates ``sample_filter`` to keep
    only the pre-computed index set.

    Subclass must set:
        _DATASET_NAME (str): key used to load the bundled index file,
                             e.g. 'live_code_bench', 'aa_lcr', 'mmmu'.
    """

    _DATASET_NAME: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        strategy    = self.extra_params.get('pruning_strategy', 'difficulty_diversity')
        prune_ratio = float(self.extra_params.get('prune_ratio', 0.1))

        from evalscope_ext.pruning.strategy import DifficultyDiversityPruner
        self._pruner = DifficultyDiversityPruner(
            dataset_name=self._DATASET_NAME,
            prune_ratio=prune_ratio,
            strategy=strategy,
        )
        self._keep_ids       = self._pruner.index_set()
        self._record_counter = 0

        logger.info(
            f'[{type(self).__name__}] strategy={strategy!r}, '
            f'prune_ratio={prune_ratio}, '
            f'selected={self._pruner.n_selected}/{self._pruner.total_samples} samples'
        )

    def record_to_sample(self, record):
        # evalscope calls reindex() AFTER sample_filter, so sample.id is None
        # at filter time. Stamp the row counter here so sample_filter can use it.
        sample    = super().record_to_sample(record)
        sample.id = self._record_counter
        self._record_counter += 1
        return sample

    def sample_filter(self, sample) -> bool:
        return sample.id in self._keep_ids