"""Dynamic benchmark case discovery and selection."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from kirin import ir


@dataclass(frozen=True)
class BenchmarkCase:
    """Reproducible benchmark input circuit."""

    case_id: str
    kernel: ir.Method
    tags: tuple[str, ...] = ()
    logical_initialize: bool = True


SIZE_BUCKETS = ("small", "medium", "large")
_KERNELS_DIR = Path(__file__).parent


def _discover_case_kernel(module_name: str, case_id: str) -> ir.Method:
    module = import_module(module_name)
    kernels = [value for value in vars(module).values() if isinstance(value, ir.Method)]
    if len(kernels) != 1:
        raise RuntimeError(
            f"Expected exactly one squin kernel in module '{module_name}' for case "
            f"'{case_id}', found {len(kernels)}."
        )
    return kernels[0]


def _discover_cases_for_bucket(bucket: str) -> tuple[BenchmarkCase, ...]:
    bucket_dir = _KERNELS_DIR / bucket
    cases: list[BenchmarkCase] = []
    for path in sorted(bucket_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        case_id = path.stem
        module_name = f"{__name__}.{bucket}.{case_id}"
        kernel = _discover_case_kernel(module_name, case_id)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                kernel=kernel,
                tags=tuple(case_id.split("_")),
                logical_initialize="physical" not in case_id,
            )
        )
    return tuple(cases)


def _discover_benchmark_cases() -> tuple[BenchmarkCase, ...]:
    cases: list[BenchmarkCase] = []
    for bucket in SIZE_BUCKETS:
        cases.extend(_discover_cases_for_bucket(bucket))
    cases.sort(key=lambda case: case.case_id)
    return tuple(cases)


BENCHMARK_CASES: tuple[BenchmarkCase, ...] = _discover_benchmark_cases()
_CASES_BY_ID = {case.case_id: case for case in BENCHMARK_CASES}
_CASES_BY_BUCKET = {
    bucket: _discover_cases_for_bucket(bucket) for bucket in SIZE_BUCKETS
}


def select_benchmark_cases(case_filter: set[str] | None) -> tuple[BenchmarkCase, ...]:
    """Resolve case selection from explicit names and/or size buckets."""
    if case_filter is None:
        return BENCHMARK_CASES

    selected: dict[str, BenchmarkCase] = {}
    unknown: list[str] = []
    for selector in case_filter:
        if selector in _CASES_BY_BUCKET:
            for case in _CASES_BY_BUCKET[selector]:
                selected[case.case_id] = case
            continue

        case = _CASES_BY_ID.get(selector)
        if case is None:
            unknown.append(selector)
            continue
        selected[case.case_id] = case

    if unknown:
        supported = ", ".join(sorted((*SIZE_BUCKETS, *_CASES_BY_ID.keys())))
        requested = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown case selector(s): {requested}. Supported selectors: {supported}"
        )

    return tuple(sorted(selected.values(), key=lambda case: case.case_id))


__all__ = ["BENCHMARK_CASES", "BenchmarkCase", "SIZE_BUCKETS", "select_benchmark_cases"]
