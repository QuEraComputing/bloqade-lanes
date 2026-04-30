"""CLI entrypoint for benchmark harness."""

from __future__ import annotations

import argparse
import re
from collections import Counter
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
from benchmarks.harness.models import BUILTIN_ARCH_SPEC_ID, BenchmarkJob
from benchmarks.kernels import select_benchmark_cases

from bloqade.lanes.arch import ArchSpec
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec

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

    try:
        arch_spec_pairs = _resolve_arch_specs(args.arch_spec)
    except ValueError as exc:
        parser.error(str(exc))

    strategies = tuple(
        cfg
        for (arch_id, arch) in arch_spec_pairs
        for cfg in default_strategy_configs(
            arch_spec=(arch_id, (lambda arch=arch: arch)),
        )
    )
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
    parser.add_argument(
        "--arch-spec",
        nargs="+",
        default=None,
        help=(
            "Path(s) to archspec JSON file(s). When omitted, uses the built-in "
            f"physical archspec (id='{BUILTIN_ARCH_SPEC_ID}'). Each file's "
            "filename stem is used as its id in output rows; stem collisions "
            f"across paths and the reserved id '{BUILTIN_ARCH_SPEC_ID}' are an "
            "error."
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


def _resolve_arch_specs(
    paths: list[str] | None,
) -> list[tuple[str, ArchSpec]]:
    """Resolve --arch-spec paths into (id, ArchSpec) pairs.

    Returns a single (BUILTIN_ARCH_SPEC_ID, built-in archspec) pair when paths
    is None or empty. Each path's identifier is its filename stem. Paths are
    loaded eagerly; schema errors, stem collisions, and use of the reserved
    BUILTIN_ARCH_SPEC_ID stem surface before any benchmark runs.
    """
    if not paths:
        return [(BUILTIN_ARCH_SPEC_ID, physical.get_arch_spec())]

    stems = [Path(p).stem for p in paths]
    reserved = sorted(
        {
            path
            for path, stem in zip(paths, stems, strict=True)
            if stem == BUILTIN_ARCH_SPEC_ID
        }
    )
    if reserved:
        raise ValueError(
            f"--arch-spec filename stem '{BUILTIN_ARCH_SPEC_ID}' is reserved "
            f"for the built-in physical archspec; rename the file: {reserved}"
        )

    counts = Counter(stems)
    duplicates = sorted({stem for stem, n in counts.items() if n > 1})
    if duplicates:
        colliding = [
            path for path, stem in zip(paths, stems, strict=True) if stem in duplicates
        ]
        raise ValueError(
            "duplicate --arch-spec filename stem across paths: "
            f"{sorted(set(colliding))}"
        )

    resolved: list[tuple[str, ArchSpec]] = []
    for path_str, stem in zip(paths, stems, strict=True):
        path = Path(path_str)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Cannot read --arch-spec file '{path}': {exc}") from exc
        try:
            rust_spec = _RustArchSpec.from_json(text)
        except Exception as exc:
            raise ValueError(
                f"Failed to parse --arch-spec file '{path}': {exc}"
            ) from exc
        resolved.append((stem, ArchSpec(rust_spec)))
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
