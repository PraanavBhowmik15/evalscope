"""
evalscope_ext/benchmarks/aa_lcr_pruned.py

Pruned variant of AA-LCR (Artificial Analysis Long Context Retrieval).

Registered as dataset name ``aa_lcr_pruned``.

Extra params (passed via --dataset-args or dataset_args dict):

    pruning_strategy  (str, default "difficulty_diversity")
        Which pruning strategy to use.

    prune_ratio  (float, default 0.2)
        Fraction of samples to retain.  Supported values: 0.1, 0.2, 0.3.
        AA-LCR has only 100 samples; 0.2 (≈20 samples) is recommended as
        the minimum reliable subset given LLM-judge noise.

    text_dir  (str | null, default null)
        Local directory with extracted AA-LCR text files.  Omit to auto-download.

CLI example::

    evalscope eval \\
        --model <model> \\
        --api-url <url> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"aa_lcr_pruned": {"extra_params": {"prune_ratio": 0.2}}}'

Python example::

    from evalscope import run_task
    from evalscope.config import TaskConfig
    import evalscope_ext  # noqa: F401 — registers pruned benchmarks

    run_task(TaskConfig(
        model="<model>",
        datasets=["aa_lcr_pruned"],
        dataset_args={"aa_lcr_pruned": {"extra_params": {"prune_ratio": 0.2}}},
    ))
"""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.strategy import DifficultyDiversityPruner

logger = get_logger()

PROMPT_TEMPLATE = """
BEGIN INPUT DOCUMENTS

{documents_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


@register_benchmark(
    BenchmarkMeta(
        name="aa_lcr_pruned",
        pretty_name="AA-LCR (Pruned)",
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

Pruned variant of AA-LCR using difficulty-stratified diversity sampling.
Stratifies across two axes: difficulty bucket (d0–d3 by observed pass rate)
and source-document count (few / medium / many).  Retains a configurable
fraction of the 100-sample benchmark while preserving coverage at every
difficulty level.

## Why prune_ratio=0.2 by default

AA-LCR is scored by an LLM judge (non-deterministic). Very small subsets
(< 10 samples) have high variance from judge noise alone.  20 samples
(ratio=0.2) balances evaluation speed against noise floor.

## Extra Parameters

| Parameter          | Type    | Default                | Description                            |
|--------------------|---------|------------------------|----------------------------------------|
| `pruning_strategy` | `str`   | `difficulty_diversity` | Which strategy to apply                |
| `prune_ratio`      | `float` | `0.2`                  | Fraction of samples to retain          |
| `text_dir`         | `str`   | `null`                 | Local extracted-text directory         |
""",
        dataset_id="evalscope/AA-LCR",
        metric_list=["acc"],
        few_shot_num=0,
        train_split=None,
        eval_split="test",
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            "pruning_strategy": {
                "type": "str",
                "description": "Pruning strategy.  Only 'difficulty_diversity' supported.",
                "value": "difficulty_diversity",
            },
            "prune_ratio": {
                "type": "float",
                "description": "Fraction of samples to retain (0 < ratio ≤ 1).",
                "value": 0.2,
            },
            "text_dir": {
                "type": "str | null",
                "description": "Local directory with extracted AA-LCR text files; null to auto-download.",
                "value": None,
            },
        },
    )
)
class AALCRPrunedAdapter(AALCRAdapter):
    """
    Pruned AA-LCR.  Inherits the full evaluation pipeline from AALCRAdapter
    (including document loading and LLM judging) and adds an index-based
    sample filter on top.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        strategy    = self.extra_params.get("pruning_strategy", "difficulty_diversity")
        prune_ratio = float(self.extra_params.get("prune_ratio", 0.2))

        self._pruner = DifficultyDiversityPruner(
            dataset_name="aa_lcr",
            prune_ratio=prune_ratio,
            strategy=strategy,
        )
        self._keep_ids = self._pruner.index_set()

        logger.info(
            f"[AALCRPruned] strategy={strategy!r}, "
            f"prune_ratio={prune_ratio}, "
            f"selected={self._pruner.n_selected}/{self._pruner.total_samples} samples"
        )

    def sample_filter(self, sample) -> bool:
        """Keep a sample only if its ID is in the pre-computed pruned set."""
        return sample.id in self._keep_ids