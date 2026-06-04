"""
evalscope_ext/tools/compare_runs.py

Compares a full benchmark run against a pruned run and reports:
  - Per-model scores on full vs pruned
  - Score delta per model
  - Spearman rank correlation between full and pruned rankings
  - Verdict: is the pruned subset reliable for go/no-go?

Usage::

    python -m evalscope_ext.tools.compare_runs \\
        --full  ./results_full/ \\
        --pruned ./results_pruned/

The tool auto-detects which benchmarks are present in the result directories
and reports them all.

Expected directory structure (produced by evalscope eval)::

    results_full/
        <model_name>/
            <benchmark_name>/
                <subset>/
                    review_summary.json   <- contains overall accuracy

    results_pruned/
        <model_name>/
            <benchmark_name>/
                <subset>/
                    review_summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _find_report_files(root: Path):
    """
    Yield (model, benchmark, path) for every evalscope report JSON found.

    Evalscope 1.8+ output layout::

        <work_dir>/
            <timestamp>/          <- optional; auto-detected
                reports/
                    <model>/
                        <benchmark>.json
    """
    # If root contains a single timestamp dir (auto-created by evalscope), descend into it
    candidates = [root]
    timestamp_dirs = [d for d in root.iterdir() if d.is_dir() and d.name != "configs"]
    if timestamp_dirs and all((d / "reports").is_dir() for d in timestamp_dirs):
        candidates = timestamp_dirs  # multiple runs; use all

    for base in candidates:
        reports_dir = base / "reports"
        if not reports_dir.is_dir():
            # Fallback: maybe user pointed directly at the reports/ dir
            reports_dir = base
        for model_dir in reports_dir.iterdir():
            if not model_dir.is_dir():
                continue
            for report_file in model_dir.glob("*.json"):
                if report_file.name == "report.json":
                    continue  # aggregate, skip
                benchmark = report_file.stem
                yield model_dir.name, benchmark, report_file


def _parse_score(report_path: Path) -> Optional[float]:
    """Extract the primary accuracy score from an evalscope report JSON."""
    try:
        with open(report_path, encoding="utf-8") as f:
            data = json.load(f)
        # evalscope 1.8+ format: top-level "score" field
        if "score" in data and isinstance(data["score"], (int, float)):
            return float(data["score"])
        # Fallback: first numeric score in metrics list
        for metric in data.get("metrics", []):
            if "score" in metric and isinstance(metric["score"], (int, float)):
                return float(metric["score"])
        # Legacy / other formats
        for key in ("acc", "pass@1", "accuracy", "mean"):
            if key in data and isinstance(data[key], (int, float)):
                return float(data[key])
    except Exception as e:
        print(f"  Warning: could not parse {report_path}: {e}", file=sys.stderr)
    return None


def load_scores(results_dir: Path) -> Dict[Tuple[str, str, str], float]:
    """Return {(model, benchmark, 'all'): score} for all reports found."""
    scores = {}
    for model, bench, path in _find_report_files(results_dir):
        score = _parse_score(path)
        if score is not None:
            scores[(model, bench, "all")] = score
    return scores


# ---------------------------------------------------------------------------
# Rank correlation
# ---------------------------------------------------------------------------

def spearman_rho(full_scores: list, pruned_scores: list) -> float:
    """
    Compute Spearman's rho between two matched score lists.
    Uses a simple rank-difference formula (no external deps needed).
    """
    n = len(full_scores)
    if n < 2:
        return float("nan")

    def rank(lst):
        sorted_idx = sorted(range(n), key=lambda i: lst[i], reverse=True)
        r = [0] * n
        for rank_val, idx in enumerate(sorted_idx, 1):
            r[idx] = rank_val
        return r

    r_full   = rank(full_scores)
    r_pruned = rank(pruned_scores)
    d_sq     = sum((r_full[i] - r_pruned[i]) ** 2 for i in range(n))
    rho      = 1 - 6 * d_sq / (n * (n ** 2 - 1))
    return round(rho, 4)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _header(text: str):
    print(f"\n{'-' * 60}")
    print(f"  {text}")
    print(f"{'-' * 60}")


def _table(rows, headers):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    sep = "  ".join("-" * w for w in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))


def compare(full_dir: Path, pruned_dir: Path):
    full_scores   = load_scores(full_dir)
    pruned_scores = load_scores(pruned_dir)

    if not full_scores:
        print(f"No result summaries found in {full_dir}", file=sys.stderr)
        sys.exit(1)
    if not pruned_scores:
        print(f"No result summaries found in {pruned_dir}", file=sys.stderr)
        sys.exit(1)

    # Group by (benchmark, subset)
    benchmarks = sorted(set(
        (bench, subset)
        for (model, bench, subset) in full_scores
    ))

    all_verdicts_pass = True

    for bench, subset in benchmarks:
        _header(f"Benchmark: {bench}  /  Subset: {subset}")

        full_models   = {m: s for (m, b, ss), s in full_scores.items()   if b == bench and ss == subset}
        pruned_models = {m: s for (m, b, ss), s in pruned_scores.items() if b == bench and ss == subset}

        # Models present in both
        common_models = sorted(set(full_models) & set(pruned_models))
        only_full     = sorted(set(full_models) - set(pruned_models))
        only_pruned   = sorted(set(pruned_models) - set(full_models))

        if only_full:
            print(f"  Note: models only in full run: {only_full}")
        if only_pruned:
            print(f"  Note: models only in pruned run: {only_pruned}")

        if not common_models:
            print("  No models in common — cannot compare.")
            continue

        rows = []
        f_vals = []
        p_vals = []
        for model in common_models:
            f  = full_models[model]
            p  = pruned_models[model]
            d  = p - f
            sign = "+" if d >= 0 else ""
            rows.append((model, f"{f:.1%}", f"{p:.1%}", f"{sign}{d:.1%}"))
            f_vals.append(f)
            p_vals.append(p)

        _table(rows, ["Model", "Full", "Pruned", "Delta"])

        rho = spearman_rho(f_vals, p_vals)
        max_delta = max(abs(p - f) for f, p in zip(f_vals, p_vals))

        print(f"\n  Rank correlation (Spearman rho): {rho:.3f}")
        print(f"  Max score delta: {max_delta:.1%}  "
              f"(expected: pruned set overweights hard problems, deflating absolute scores)")

        # Rank preservation (rho) is the primary go/no-go signal.
        # Absolute score delta is informational -- the pruned set intentionally
        # over-represents hard problems, deflating scores while preserving rankings.
        PASS_RHO = 0.85

        rho_ok       = rho >= PASS_RHO or len(common_models) < 3
        verdict_pass = rho_ok
        all_verdicts_pass = all_verdicts_pass and verdict_pass

        if verdict_pass:
            verdict = f"PASS  rho={rho:.3f} >= {PASS_RHO}  (rankings reliable for go/no-go)"
        else:
            verdict = f"WARN  rho={rho:.3f} < {PASS_RHO}  (rankings not preserved)"

        print(f"  VERDICT: {verdict}")

    _header("Overall")
    print(f"  {'All benchmarks: RELIABLE' if all_verdicts_pass else 'One or more benchmarks: UNRELIABLE — review deltas above'}")
    return 0 if all_verdicts_pass else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare a full benchmark run against a pruned run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--full",   required=True, type=Path,
                        help="Directory containing full benchmark results.")
    parser.add_argument("--pruned", required=True, type=Path,
                        help="Directory containing pruned benchmark results.")
    args = parser.parse_args()

    if not args.full.exists():
        print(f"Error: --full directory not found: {args.full}", file=sys.stderr)
        sys.exit(1)
    if not args.pruned.exists():
        print(f"Error: --pruned directory not found: {args.pruned}", file=sys.stderr)
        sys.exit(1)

    sys.exit(compare(args.full, args.pruned))


if __name__ == "__main__":
    main()