# flake8: noqa: E501
"""
Pruned variant of AA-LCR (Artificial Analysis Long Context Retrieval).

Registered as dataset name ``aa_lcr_pruned``.
Requires the ``evalscope_ext`` package to be installed::

    pip install -e ./evalscope_ext

Extra params:

    pruning_strategy  (str, default "difficulty_diversity")
    prune_ratio       (float, default 0.2)  -- 0.2 recommended minimum given LLM-judge noise
    text_dir          (str | null)          -- local extracted-text directory; null to auto-download

CLI example::

    evalscope eval \\
        --model <model> --api-url <url> \\
        --datasets aa_lcr_pruned \\
        --dataset-args '{"aa_lcr_pruned": {"extra_params": {"prune_ratio": 0.2}}}'
"""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.aa_lcr.aa_lcr_adapter import AALCRAdapter
from evalscope.constants import Tags
from evalscope_ext.pruning.mixin import PrunedMixin

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
        name='aa_lcr_pruned',
        pretty_name='AA-LCR (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.REASONING, Tags.LONG_CONTEXT],
        description="""
## Overview

Pruned variant of AA-LCR using difficulty-stratified diversity sampling.
Stratifies across two axes: difficulty bucket (d0-d3 by observed pass rate)
and source-document count (few / medium / many). Retains a configurable
fraction of the 100-sample benchmark while preserving coverage at every
difficulty level.

## Why prune_ratio=0.2 by default

AA-LCR is scored by an LLM judge (non-deterministic). Very small subsets
(< 10 samples) have high variance from judge noise alone. 20 samples
(ratio=0.2) balances evaluation speed against noise floor.

## Extra Parameters

| Parameter          | Type    | Default                | Description                   |
|--------------------|---------|------------------------|-------------------------------|
| `pruning_strategy` | `str`   | `difficulty_diversity` | Which strategy to apply       |
| `prune_ratio`      | `float` | `0.2`                  | Fraction of samples to retain |
| `text_dir`         | `str`   | `null`                 | Local extracted-text dir      |
""",
        dataset_id='evalscope/AA-LCR',
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='test',
        prompt_template=PROMPT_TEMPLATE,
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': "Pruning strategy. Only 'difficulty_diversity' supported.",
                'value': 'difficulty_diversity',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to retain (0 < ratio <= 1).',
                'value': 0.2,
            },
            'text_dir': {
                'type': 'str | null',
                'description': 'Local directory with extracted AA-LCR text files; null to auto-download.',
                'value': None,
            },
        },
    )
)
class AALCRPrunedAdapter(PrunedMixin, AALCRAdapter):
    _DATASET_NAME = 'aa_lcr'