# flake8: noqa: E501
"""
Pruned variant of LiveCodeBench v5.

Registered as dataset name ``live_code_bench_pruned``.
Requires the ``evalscope_ext`` package to be installed::

    pip install -e ./evalscope_ext

Extra params (passed via --dataset-args or dataset_args dict):

    pruning_strategy  (str, default "difficulty_diversity")
    prune_ratio       (float, default 0.1)  -- supported: 0.1, 0.2, 0.3

CLI example::

    evalscope eval \\
        --model <model> --api-url <url> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.1}}}' \\
        --sandbox '{"enabled": true}'
"""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope_ext.pruning.mixin import PrunedMixin


@register_benchmark(
    BenchmarkMeta(
        name='live_code_bench_pruned',
        pretty_name='Live-Code-Bench (Pruned)',
        tags=[Tags.CODING],
        description="""
## Overview

Pruned variant of LiveCodeBench v5 using difficulty-stratified diversity sampling.
Retains a configurable fraction of problems (default 10%) while preserving signal
for model ranking.  Designed for fast go/no-go evaluation without full benchmark cost.

## Pruning strategy: difficulty_diversity

Problems are binned into 4 difficulty buckets by observed pass rate, then
within each bucket stratified by agreement pattern (PPF, PFP, etc.) and
representatives are selected to maximise coverage of problem complexity.

## Extra Parameters

| Parameter          | Type    | Default                | Description                   |
|--------------------|---------|------------------------|-------------------------------|
| `pruning_strategy` | `str`   | `difficulty_diversity` | Which strategy to apply       |
| `prune_ratio`      | `float` | `0.1`                  | Fraction of samples to retain |
""",
        dataset_id='evalscope/livecodebench_code_generation_lite_parquet',
        subset_list=['v5'],
        metric_list=['acc'],
        aggregation='mean_and_pass_at_k',
        eval_split='test',
        prompt_template=(
            '### Question:\n{question_content}\n\n'
            '{format_prompt} ### Answer: (use the provided format with backticks)\n\n'
        ),
        review_timeout=6,
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
            'start_date': {
                'type': 'str | null',
                'description': 'Filter problems from this date onward (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'end_date': {
                'type': 'str | null',
                'description': 'Filter problems up to this date (YYYY-MM-DD). Null keeps all.',
                'value': None,
            },
            'debug': {
                'type': 'bool',
                'description': 'Enable verbose debug logging.',
                'value': False,
            },
        },
        sandbox_config={
            'image': 'python:3.11-slim',
            'tools_config': {'shell_executor': {}, 'python_executor': {}},
        },
    )
)
class LiveCodeBenchPrunedAdapter(PrunedMixin, LiveCodeBenchAdapter):
    _DATASET_NAME = 'live_code_bench'