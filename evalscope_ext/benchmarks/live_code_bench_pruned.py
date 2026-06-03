"""
evalscope_ext/benchmarks/live_code_bench_pruned.py

Pruned variant of LiveCodeBench v5.

Registered as dataset name ``live_code_bench_pruned``.

Extra params (passed via --dataset-args or dataset_args dict):

    pruning_strategy  (str, default "difficulty_diversity")
        Which pruning strategy to use.  Only "difficulty_diversity" is
        currently implemented.

    prune_ratio  (float, default 0.1)
        Fraction of samples to retain.  Supported values: 0.1, 0.2, 0.3.
        Nearest pre-computed ratio is used if an exact match is not found.

CLI example::

    evalscope eval \\
        --model <model> \\
        --api-url <url> \\
        --datasets live_code_bench_pruned \\
        --dataset-args '{"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.1}}}' \\
        --sandbox '{"enabled": true}'

Python example::

    from evalscope import run_task
    from evalscope.config import TaskConfig
    import evalscope_ext  # noqa: F401 — registers pruned benchmarks

    run_task(TaskConfig(
        model="<model>",
        datasets=["live_code_bench_pruned"],
        dataset_args={"live_code_bench_pruned": {"extra_params": {"prune_ratio": 0.1}}},
    ))
"""

from evalscope.api.benchmark import BenchmarkMeta
from evalscope.api.registry import register_benchmark
from evalscope.benchmarks.live_code_bench.live_code_bench_adapter import LiveCodeBenchAdapter
from evalscope.constants import Tags
from evalscope.utils.logger import get_logger

from evalscope_ext.pruning.strategy import DifficultyDiversityPruner

logger = get_logger()


@register_benchmark(
    BenchmarkMeta(
        name="live_code_bench_pruned",
        pretty_name="Live-Code-Bench (Pruned)",
        tags=[Tags.CODING],
        description="""
## Overview

Pruned variant of LiveCodeBench v5 using difficulty-stratified diversity sampling.
Retains a configurable fraction of problems (default 10%) while preserving signal
for model ranking.  Designed for fast go/no-go evaluation without full benchmark cost.

## Pruning strategy: difficulty_diversity

Problems are binned into 4 difficulty buckets by observed pass rate, then
representatives are selected within each bucket to maximise coverage of
problem complexity (message length).  Budget allocation overweights the
informative middle buckets (d1, d2) and underweights trivial extremes (d0, d3).

The selection is driven by problem properties, not any specific model's outputs,
so it generalises to models not seen during index construction.

## Extra Parameters

| Parameter          | Type    | Default                | Description                            |
|--------------------|---------|------------------------|----------------------------------------|
| `pruning_strategy` | `str`   | `difficulty_diversity` | Which strategy to apply                |
| `prune_ratio`      | `float` | `0.1`                  | Fraction of samples to retain          |
""",
        dataset_id="evalscope/livecodebench_code_generation_lite_parquet",
        subset_list=["v5"],
        metric_list=["acc"],
        aggregation="mean_and_pass_at_k",
        eval_split="test",
        prompt_template=(
            "### Question:\n{question_content}\n\n"
            "{format_prompt} ### Answer: (use the provided format with backticks)\n\n"
        ),
        review_timeout=6,
        extra_params={
            "pruning_strategy": {
                "type": "str",
                "description": "Pruning strategy.  Only 'difficulty_diversity' supported.",
                "value": "difficulty_diversity",
            },
            "prune_ratio": {
                "type": "float",
                "description": "Fraction of samples to retain (0 < ratio ≤ 1).",
                "value": 0.1,
            },
            "start_date": {
                "type": "str | null",
                "description": "Filter problems from this date onward (YYYY-MM-DD). Null keeps all.",
                "value": None,
            },
            "end_date": {
                "type": "str | null",
                "description": "Filter problems up to this date (YYYY-MM-DD). Null keeps all.",
                "value": None,
            },
            "debug": {
                "type": "bool",
                "description": "Enable verbose debug logging.",
                "value": False,
            },
        },
        sandbox_config={
            "image": "python:3.11-slim",
            "tools_config": {"shell_executor": {}, "python_executor": {}},
        },
    )
)
class LiveCodeBenchPrunedAdapter(LiveCodeBenchAdapter):
    """
    Pruned LiveCodeBench.  Inherits the full evaluation pipeline from
    LiveCodeBenchAdapter and adds an index-based sample filter on top.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        strategy   = self.extra_params.get("pruning_strategy", "difficulty_diversity")
        prune_ratio = float(self.extra_params.get("prune_ratio", 0.1))

        self._pruner = DifficultyDiversityPruner(
            dataset_name="live_code_bench",
            prune_ratio=prune_ratio,
            strategy=strategy,
        )
        self._keep_ids = self._pruner.index_set()

        logger.info(
            f"[LiveCodeBenchPruned] strategy={strategy!r}, "
            f"prune_ratio={prune_ratio}, "
            f"selected={self._pruner.n_selected}/{self._pruner.total_samples} samples"
        )

    def sample_filter(self, sample) -> bool:
        """
        Keep a sample only if its ID is in the pre-computed pruned set AND
        it passes the parent class's date filter.
        """
        if sample.id not in self._keep_ids:
            return False
        return super().sample_filter(sample)