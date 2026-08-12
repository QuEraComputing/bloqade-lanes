"""Baseline CSV loading and comparison helpers for benchmark harness."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from benchmarks.harness.models import BUILTIN_ARCH_SPEC_ID, BenchmarkRow
from benchmarks.harness.output import CSV_COLUMNS

BenchmarkKey = tuple[str, str, str]
FLOAT_COMPARE_DECIMALS = 6


@dataclass(frozen=True)
class BenchmarkDiff:
    """One detected difference between baseline and current rows."""

    case_id: str
    strategy_id: str
    arch_spec_id: str
    field: str
    baseline: object
    current: object
    kind: str

    @property
    def key(self) -> BenchmarkKey:
        return (self.case_id, self.strategy_id, self.arch_spec_id)


@dataclass(frozen=True)
class BenchmarkComparisonReport:
    """Comparison result for a full benchmark matrix."""

    missing_from_baseline: tuple[BenchmarkKey, ...]
    missing_from_current: tuple[BenchmarkKey, ...]
    diffs: tuple[BenchmarkDiff, ...]

    @property
    def matrix_mismatch(self) -> bool:
        return bool(self.missing_from_baseline or self.missing_from_current)

    @property
    def has_differences(self) -> bool:
        return self.matrix_mismatch or bool(self.diffs)


# Columns a baseline may omit, with the value used when it does. Keeping this
# explicit means adding a column does not invalidate every committed baseline,
# while an unknown or missing *required* column is still an error.
OPTIONAL_COLUMN_DEFAULTS: dict[str, str] = {
    # Predates arch_spec_id: such rows describe the built-in archspec.
    "arch_spec_id": BUILTIN_ARCH_SPEC_ID,
    # Predate branch-and-bound instrumentation; blank means "not measured".
    "cuts_by_g": "",
    "cuts_by_h": "",
    "cut_depth_sum": "",
    "cut_depth_g_only_sum": "",
    "max_optimality_gap": "",
}


def load_baseline_csv(path: Path) -> list[BenchmarkRow]:
    """Load benchmark rows from a CSV file produced by harness output.

    Columns listed in :data:`OPTIONAL_COLUMN_DEFAULTS` may be absent, so
    baselines committed before a column existed still load. Any other missing
    column, or any column this harness does not know, is an error.
    """
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Baseline CSV '{path}' is missing a header row.")
        fieldnames = tuple(reader.fieldnames)
        unknown = [c for c in fieldnames if c not in CSV_COLUMNS]
        missing = [
            c
            for c in CSV_COLUMNS
            if c not in fieldnames and c not in OPTIONAL_COLUMN_DEFAULTS
        ]
        if unknown or missing:
            expected = ", ".join(CSV_COLUMNS)
            actual = ", ".join(fieldnames)
            raise ValueError(
                f"Baseline CSV '{path}' has unexpected columns. "
                f"Expected: {expected}. Found: {actual}."
            )
        defaults = {
            column: value
            for column, value in OPTIONAL_COLUMN_DEFAULTS.items()
            if column not in fieldnames
        }
        return [_parse_row({**defaults, **row}) for row in reader]


def compare_against_baseline(
    current_rows: list[BenchmarkRow],
    baseline_rows: list[BenchmarkRow],
) -> BenchmarkComparisonReport:
    """Compare benchmark rows by key and classify row-level differences."""
    current_by_key = _rows_by_key(current_rows, source="current run")
    baseline_by_key = _rows_by_key(baseline_rows, source="baseline")

    current_keys = set(current_by_key)
    baseline_keys = set(baseline_by_key)

    missing_from_baseline = tuple(sorted(current_keys - baseline_keys))
    missing_from_current = tuple(sorted(baseline_keys - current_keys))

    diffs: list[BenchmarkDiff] = []
    for key in sorted(current_keys & baseline_keys):
        current = current_by_key[key]
        baseline = baseline_by_key[key]
        diffs.extend(
            _compare_rows(
                current=current,
                baseline=baseline,
            )
        )

    return BenchmarkComparisonReport(
        missing_from_baseline=missing_from_baseline,
        missing_from_current=missing_from_current,
        diffs=tuple(diffs),
    )


def render_comparison_report(report: BenchmarkComparisonReport) -> str:
    """Render plain-text comparison output with improvement/degradation markers."""
    lines: list[str] = ["comparison results"]
    if report.missing_from_baseline:
        lines.append("matrix mismatch: rows missing from baseline")
        for case_id, strategy_id, arch_spec_id in report.missing_from_baseline:
            lines.append(
                f"  ! {case_id}/{strategy_id}/{arch_spec_id} only in current run"
            )
    if report.missing_from_current:
        lines.append("matrix mismatch: rows missing from current run")
        for case_id, strategy_id, arch_spec_id in report.missing_from_current:
            lines.append(f"  ! {case_id}/{strategy_id}/{arch_spec_id} only in baseline")

    if report.diffs:
        lines.append("row differences")
        for diff in report.diffs:
            marker = {"improved": "+", "degraded": "-", "changed": "!"}.get(
                diff.kind, "!"
            )
            lines.append(
                f"  {marker} {diff.case_id}/{diff.strategy_id}/{diff.arch_spec_id} "
                f"{diff.field}: "
                f"baseline={diff.baseline!r} current={diff.current!r} ({diff.kind})"
            )

    if not report.has_differences:
        lines.append("no differences found")
    return "\n".join(lines)


def _rows_by_key(
    rows: list[BenchmarkRow], *, source: str
) -> dict[BenchmarkKey, BenchmarkRow]:
    indexed: dict[BenchmarkKey, BenchmarkRow] = {}
    for row in rows:
        key = (row.case_id, row.strategy_id, row.arch_spec_id)
        if key in indexed:
            case_id, strategy_id, arch_spec_id = key
            raise ValueError(
                f"Duplicate row for case_id='{case_id}', strategy_id='{strategy_id}', "
                f"arch_spec_id='{arch_spec_id}' in {source}."
            )
        indexed[key] = row
    return indexed


def _compare_rows(
    *,
    current: BenchmarkRow,
    baseline: BenchmarkRow,
) -> list[BenchmarkDiff]:
    diffs: list[BenchmarkDiff] = []
    case_id = current.case_id
    strategy_id = current.strategy_id
    arch_spec_id = current.arch_spec_id

    _add_strict_diff(
        diffs,
        case_id,
        strategy_id,
        arch_spec_id,
        "backend",
        baseline.backend,
        current.backend,
    )
    _add_strict_diff(
        diffs,
        case_id,
        strategy_id,
        arch_spec_id,
        "generator_id",
        baseline.generator_id,
        current.generator_id,
    )

    if baseline.success != current.success:
        status_kind = (
            "degraded" if baseline.success and not current.success else "improved"
        )
        diffs.append(
            BenchmarkDiff(
                case_id=case_id,
                strategy_id=strategy_id,
                arch_spec_id=arch_spec_id,
                field="success",
                baseline=baseline.success,
                current=current.success,
                kind=status_kind,
            )
        )
        return diffs

    if not baseline.success and not current.success:
        return diffs

    _add_numeric_delta(
        diffs,
        case_id=case_id,
        strategy_id=strategy_id,
        arch_spec_id=arch_spec_id,
        field="move_count_events",
        baseline=baseline.move_count_events,
        current=current.move_count_events,
        lower_is_better=True,
    )
    _add_numeric_delta(
        diffs,
        case_id=case_id,
        strategy_id=strategy_id,
        arch_spec_id=arch_spec_id,
        field="move_count_lanes",
        baseline=baseline.move_count_lanes,
        current=current.move_count_lanes,
        lower_is_better=True,
    )
    _add_numeric_delta(
        diffs,
        case_id=case_id,
        strategy_id=strategy_id,
        arch_spec_id=arch_spec_id,
        field="estimated_fidelity",
        baseline=baseline.estimated_fidelity,
        current=current.estimated_fidelity,
        lower_is_better=False,
        float_compare_decimals=FLOAT_COMPARE_DECIMALS,
    )
    _add_numeric_delta(
        diffs,
        case_id=case_id,
        strategy_id=strategy_id,
        arch_spec_id=arch_spec_id,
        field="nodes_explored",
        baseline=baseline.nodes_explored,
        current=current.nodes_explored,
        lower_is_better=True,
        percent_tolerance=0.10,
    )
    _add_numeric_delta(
        diffs,
        case_id=case_id,
        strategy_id=strategy_id,
        arch_spec_id=arch_spec_id,
        field="max_depth_reached",
        baseline=baseline.max_depth_reached,
        current=current.max_depth_reached,
        lower_is_better=True,
    )
    # Pruning counters are deterministic, so compare them exactly. Neither
    # direction is inherently better -- fewer cuts can mean a tighter bound
    # reached the incumbent sooner, more cuts can mean it is doing more work --
    # so these go through `_add_strict_diff`, which reports "changed" rather
    # than judging. `_add_numeric_delta` has no neutral verdict: either setting
    # of `lower_is_better` would label one direction an improvement.
    for field_name, base_value, cur_value in (
        ("cuts_by_g", baseline.cuts_by_g, current.cuts_by_g),
        ("cuts_by_h", baseline.cuts_by_h, current.cuts_by_h),
        ("cut_depth_sum", baseline.cut_depth_sum, current.cut_depth_sum),
        (
            "cut_depth_g_only_sum",
            baseline.cut_depth_g_only_sum,
            current.cut_depth_g_only_sum,
        ),
    ):
        _add_strict_diff(
            diffs,
            case_id,
            strategy_id,
            arch_spec_id,
            field_name,
            base_value,
            cur_value,
        )
    _add_numeric_delta(
        diffs,
        case_id=case_id,
        strategy_id=strategy_id,
        arch_spec_id=arch_spec_id,
        field="max_optimality_gap",
        baseline=baseline.max_optimality_gap,
        current=current.max_optimality_gap,
        lower_is_better=True,
        float_compare_decimals=FLOAT_COMPARE_DECIMALS,
    )
    _add_strict_diff(
        diffs,
        case_id,
        strategy_id,
        arch_spec_id,
        "notes",
        baseline.notes,
        current.notes,
    )
    return diffs


def _add_strict_diff(
    diffs: list[BenchmarkDiff],
    case_id: str,
    strategy_id: str,
    arch_spec_id: str,
    field: str,
    baseline: object,
    current: object,
) -> None:
    if baseline == current:
        return
    diffs.append(
        BenchmarkDiff(
            case_id=case_id,
            strategy_id=strategy_id,
            arch_spec_id=arch_spec_id,
            field=field,
            baseline=baseline,
            current=current,
            kind="changed",
        )
    )


def _add_numeric_delta(
    diffs: list[BenchmarkDiff],
    *,
    case_id: str,
    strategy_id: str,
    arch_spec_id: str,
    field: str,
    baseline: float | None,
    current: float | None,
    lower_is_better: bool,
    float_compare_decimals: int | None = None,
    percent_tolerance: float | None = None,
) -> None:
    if baseline is None or current is None:
        _add_strict_diff(
            diffs, case_id, strategy_id, arch_spec_id, field, baseline, current
        )
        return
    baseline_cmp = baseline
    current_cmp = current
    if float_compare_decimals is not None:
        baseline_cmp = _round_float_for_compare(
            baseline, decimals=float_compare_decimals
        )
        current_cmp = _round_float_for_compare(current, decimals=float_compare_decimals)
    if baseline_cmp == current_cmp:
        return
    if (
        percent_tolerance is not None
        and baseline != 0
        and abs(current - baseline) / abs(baseline) <= percent_tolerance
    ):
        return
    if lower_is_better:
        kind = "degraded" if current > baseline else "improved"
    else:
        kind = "degraded" if current < baseline else "improved"
    diffs.append(
        BenchmarkDiff(
            case_id=case_id,
            strategy_id=strategy_id,
            arch_spec_id=arch_spec_id,
            field=field,
            baseline=baseline,
            current=current,
            kind=kind,
        )
    )


def _round_float_for_compare(value: float, *, decimals: int) -> float:
    return round(float(value), decimals)


def _parse_row(raw: dict[str, str]) -> BenchmarkRow:
    return BenchmarkRow(
        case_id=raw["case_id"],
        strategy_id=raw["strategy_id"],
        backend=raw["backend"],  # type: ignore[arg-type]
        generator_id=raw["generator_id"],
        success=_parse_bool(raw["success"]),
        wall_time_ms=_parse_optional_float(raw["wall_time_ms"]),
        move_count_events=_parse_optional_int(raw["move_count_events"]),
        move_count_lanes=_parse_optional_int(raw["move_count_lanes"]),
        estimated_fidelity=_parse_optional_float(raw["estimated_fidelity"]),
        nodes_explored=_parse_optional_int(raw["nodes_explored"]),
        max_depth_reached=_parse_optional_int(raw["max_depth_reached"]),
        cuts_by_g=_parse_optional_int(raw["cuts_by_g"]),
        cuts_by_h=_parse_optional_int(raw["cuts_by_h"]),
        cut_depth_sum=_parse_optional_int(raw["cut_depth_sum"]),
        cut_depth_g_only_sum=_parse_optional_int(raw["cut_depth_g_only_sum"]),
        max_optimality_gap=_parse_optional_float(raw["max_optimality_gap"]),
        arch_spec_id=raw["arch_spec_id"],
        notes=raw["notes"],
    )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Invalid boolean in CSV: {value!r}")


def _parse_optional_float(value: str) -> float | None:
    stripped = value.strip()
    if not stripped:
        return None
    return float(stripped)


def _parse_optional_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)
