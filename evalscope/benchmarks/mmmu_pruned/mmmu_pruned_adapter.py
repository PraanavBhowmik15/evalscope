# flake8: noqa: E501
"""
Pruned variant of MMMU (Massive Multitask Multimodal Understanding).

Registered as dataset name ``mmmu_pruned``.
Requires the ``evalscope_ext`` package to be installed::

    pip install -e ./evalscope_ext

Pruning strategy:
  Stratifies by topic_difficulty (Easy/Medium/Hard) × discipline_group × img_type_group.
  Overweights Hard and Medium samples (the multimodal probe targets real encoder failures,
  not easy text-dominated questions).

  Budget weights: Easy 15% / Medium 45% / Hard 40%

Extra params:

    pruning_strategy  (str,   default "difficulty_diversity")
    prune_ratio       (float, default 0.1)

CLI example::

    evalscope eval \\
        --model <model> --api-url <url> \\
        --datasets mmmu_pruned \\
        --dataset-args '{"mmmu_pruned": {"extra_params": {"prune_ratio": 0.1}}}'
"""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.mmmu.mmmu_adapter import MMMUAdapter
from evalscope.constants import Tags
from evalscope_ext.pruning.mixin import PrunedMixin

# The 22 MMMU subjects covered by the shipped glm-4.5v-fp8 evaluation data.
# Kept in alphabetical order to match the row-counter assignment in the index file.
_SUBSET_LIST = [
    'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
    'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
    'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics',
    'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage',
    'Marketing', 'Materials',
]


@register_benchmark(
    BenchmarkMeta(
        name='mmmu_pruned',
        pretty_name='MMMU (Pruned)',
        tags=[Tags.KNOWLEDGE, Tags.MULTI_MODAL],
        description="""
## Overview

Pruned variant of MMMU using difficulty-stratified diversity sampling.
Covers 22 subjects across 5 discipline groups (Art, Science, Engineering,
Business, Other) and 7 image-type categories.  Designed as a multimodal probe:
Hard and Medium samples are overweighted so that encoder-quality differences
are amplified rather than washed out by easy, text-dominated questions.

## Pruning strategy: difficulty_diversity

Samples are stratified by topic_difficulty × discipline_group, then a secondary
diversity pass maximises image-type coverage within each stratum (ensuring
Diagrams, Tables, Photographs, Plots, Paintings, and Scientific images are
all represented).

## Budget weights

| Difficulty | Weight | Rationale |
|---|---|---|
| Easy | 15% | 71% overall acc means Easy items have low discrimination power |
| Medium | 45% | Primary evaluation zone |
| Hard | 40% | Over-represented to surface encoder failures |

## Extra Parameters

| Parameter          | Type    | Default                | Description                   |
|--------------------|---------|------------------------|-------------------------------|
| `pruning_strategy` | `str`   | `difficulty_diversity` | Which strategy to apply       |
| `prune_ratio`      | `float` | `0.1`                  | Fraction of samples to retain |
""",
        dataset_id='AI-ModelScope/MMMU',
        subset_list=_SUBSET_LIST,
        metric_list=['acc'],
        few_shot_num=0,
        train_split=None,
        eval_split='validation',
        extra_params={
            'pruning_strategy': {
                'type': 'str',
                'description': "Pruning strategy. Only 'difficulty_diversity' supported.",
                'value': 'difficulty_diversity',
            },
            'prune_ratio': {
                'type': 'float',
                'description': 'Fraction of samples to retain (0 < ratio <= 1).',
                'value': 0.1,
            },
        },
    )
)
class MMMUPrunedAdapter(PrunedMixin, MMMUAdapter):
    _DATASET_NAME = 'mmmu'