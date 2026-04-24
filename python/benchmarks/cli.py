"""CLI entrypoint for benchmark harness."""

from __future__ import annotations

import argparse
import re
from dataclasses import replace
from pathlib import Path

from benchmarks.harness import (
    BenchmarkRunner,
    compare_against_baseline,
    default_strategy_configs,
    expand_benchmark_jobs,
    load_baseline_csv,
    render_comparison_report,
    render_console_table,
    write_csv,
)
from benchmarks.harness.models import BenchmarkJob
from benchmarks.kernels import select_benchmark_cases

MAX_LOGICAL_QUBITS = 10


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    case_filter = _parse_filter(args.cases)
    strategy_filter = _parse_filter(args.strategies)

    try:
        cases = select_benchmark_cases(case_filter)
    except ValueError as exc:
        parser.error(str(exc))

    strategies = default_strategy_configs()
    jobs = expand_benchmark_jobs(
        cases,
        strategies,
        strategy_filter=strategy_filter,
    )
    jobs = _apply_architecture_mode(jobs, architecture_mode=args.architecture)
    if args.architecture == "logical":
        jobs = _filter_jobs_for_logical_capacity(
            jobs, max_logical_qubits=MAX_LOGICAL_QUBITS
        )
    if not jobs:
        parser.error("No benchmark jobs selected after applying filters.")

    runner = BenchmarkRunner(repeats=args.repeats, warmup=args.warmup)
    on_job_start = _print_debug_job_start if args.debug_progress else None
    rows = runner.run_jobs(jobs, on_job_start=on_job_start)

    output_path = _resolve_output_path(args.output, architecture_mode=args.architecture)
    baseline_rows = None
    if args.compare:
        baseline_path = Path(args.compare)
        try:
            baseline_rows = load_baseline_csv(baseline_path)
        except FileNotFoundError:
            print(
                "\nWARNING: compare baseline CSV was not found; "
                f"continuing without comparison: {baseline_path}"
            )
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot load compare baseline '{baseline_path}': {exc}")

    print(render_console_table(rows))
    exit_code = 0
    if baseline_rows is not None:
        report = compare_against_baseline(
            rows,
            baseline_rows,
        )
        print()
        print(render_comparison_report(report))
        if report.has_differences:
            exit_code = 1
    else:
        print(
            "\nWARNING: no baseline CSV provided via --compare; "
            "results were written without comparison."
        )

    write_csv(rows, output_path)
    print(f"\nCSV written to {output_path}")
    return exit_code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark harness comparisons.")
    parser.add_argument(
        "--output",
        default="",
        help=(
            "CSV output path. Defaults to architecture-specific files: "
            "'python/benchmarks/harness/latest_physical.csv' or "
            "'python/benchmarks/harness/latest_logical.csv'."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of timed repeats per job",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Warmup iterations per job",
    )
    parser.add_argument(
        "--cases",
        default="",
        help=(
            "Comma-separated case selectors. Use kernel names (e.g. trotter_rand_35), "
            "size buckets (small, medium, large), or mix both."
        ),
    )
    parser.add_argument(
        "--strategies",
        default="",
        help="Comma-separated strategy IDs to include",
    )
    parser.add_argument(
        "--architecture",
        choices=("logical", "physical"),
        required=True,
        help="Execution mode: force logical or physical globally",
    )
    parser.add_argument(
        "--debug-progress",
        action="store_true",
        help="Print kernel and strategy progress while jobs execute",
    )
    parser.add_argument(
        "--compare",
        default="",
        help=(
            "Baseline CSV path used for comparison. "
            "When provided, any difference from baseline returns non-zero."
        ),
    )
    return parser


def _parse_filter(value: str) -> set[str] | None:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        return None
    return set(pieces)


def _resolve_output_path(output: str, *, architecture_mode: str) -> Path:
    if output:
        return Path(output)
    if architecture_mode == "physical":
        return Path("python/benchmarks/harness/latest_physical.csv")
    return Path("python/benchmarks/harness/latest_logical.csv")


def _apply_architecture_mode(
    jobs: list[BenchmarkJob], architecture_mode: str
) -> list[BenchmarkJob]:
    logical_initialize = architecture_mode == "logical"
    updated_jobs: list[BenchmarkJob] = []
    for job in jobs:
        updated_jobs.append(
            BenchmarkJob(
                case=replace(job.case, logical_initialize=logical_initialize),
                strategy=job.strategy,
            )
        )
    return updated_jobs


def _print_debug_job_start(job: BenchmarkJob, new_case: bool) -> None:
    if new_case:
        print(f"[debug] kernel: {job.case.case_id}")
    print(
        "[debug] start strategy: "
        f"{job.strategy.strategy_id} "
        f"(backend={job.strategy.backend}, generator={job.strategy.generator_id})"
    )


def _filter_jobs_for_logical_capacity(
    jobs: list[BenchmarkJob], *, max_logical_qubits: int
) -> list[BenchmarkJob]:
    filtered_jobs: list[BenchmarkJob] = []
    skipped_case_ids: set[str] = set()
    for job in jobs:
        qubit_count = _infer_case_qubit_count(job.case.case_id)
        if qubit_count is not None and qubit_count > max_logical_qubits:
            skipped_case_ids.add(job.case.case_id)
            continue
        filtered_jobs.append(job)

    if skipped_case_ids:
        skipped = ", ".join(sorted(skipped_case_ids))
        print(
            "[warn] logical mode skipped cases exceeding logical qubit capacity "
            f"({max_logical_qubits}): {skipped}"
        )
    return filtered_jobs


def _infer_case_qubit_count(case_id: str) -> int | None:
    match = re.search(r"(\d+)$", case_id)
    if match is None:
        return None
    return int(match.group(1))


if __name__ == "__main__":
    raise SystemExit(main())
