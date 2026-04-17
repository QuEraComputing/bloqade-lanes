from pathlib import Path

import pytest
from benchmarks.harness.compare import (
    BenchmarkComparisonReport,
    compare_against_baseline,
    load_baseline_csv,
)
from benchmarks.harness.models import BenchmarkRow
from benchmarks.harness.output import write_csv


def _row(
    *,
    case_id: str = "ghz_4",
    strategy_id: str = "python_entropy",
    success: bool = True,
    wall_time_ms: float | None = 10.0,
    move_count_events: int | None = 10,
    move_count_lanes: int | None = 10,
    estimated_fidelity: float | None = 0.9,
    nodes_explored: int | None = 10,
    max_depth_reached: int | None = 10,
) -> BenchmarkRow:
    return BenchmarkRow(
        case_id=case_id,
        strategy_id=strategy_id,
        backend="python",
        generator_id="heuristic",
        success=success,
        wall_time_ms=wall_time_ms,
        move_count_events=move_count_events,
        move_count_lanes=move_count_lanes,
        estimated_fidelity=estimated_fidelity,
        nodes_explored=nodes_explored,
        max_depth_reached=max_depth_reached,
    )


def test_load_baseline_csv_round_trips_rows(tmp_path: Path):
    rows = [_row()]
    path = tmp_path / "baseline.csv"
    write_csv(rows, path)

    loaded = load_baseline_csv(path)
    assert loaded == rows


def test_compare_detects_matrix_mismatch():
    baseline_rows = [_row(case_id="ghz_4")]
    current_rows = [_row(case_id="ghz_4"), _row(case_id="ghz_6")]
    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )

    assert report.matrix_mismatch is True
    assert report.has_differences is True
    assert report.missing_from_baseline == (("ghz_6", "python_entropy"),)
    assert report.missing_from_current == ()


def test_compare_allows_both_fail_without_diff():
    baseline_rows = [_row(success=False, wall_time_ms=None)]
    current_rows = [_row(success=False, wall_time_ms=None)]
    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )

    assert report == BenchmarkComparisonReport(
        missing_from_baseline=(),
        missing_from_current=(),
        diffs=(),
    )
    assert report.has_differences is False


def test_compare_detects_success_to_failure_transition():
    baseline_rows = [_row(success=True)]
    current_rows = [_row(success=False, wall_time_ms=None)]
    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )

    assert report.has_differences is True
    assert len(report.diffs) == 1
    assert report.diffs[0].field == "success"
    assert report.diffs[0].kind == "degraded"


def test_compare_wall_time_tolerance_is_applied():
    baseline_rows = [_row(wall_time_ms=100.0)]
    current_rows = [_row(wall_time_ms=299.0)]
    within_tolerance = compare_against_baseline(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
        wall_time_tolerance_ratio=2.0,
    )
    outside_tolerance = compare_against_baseline(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
        wall_time_tolerance_ratio=1.5,
    )

    assert within_tolerance.has_differences is False
    assert outside_tolerance.has_differences is True
    assert outside_tolerance.diffs[0].field == "wall_time_ms"


def test_compare_ignores_wall_time_improvements():
    baseline_rows = [_row(wall_time_ms=1000.0)]
    current_rows = [_row(wall_time_ms=10.0)]

    report = compare_against_baseline(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
        wall_time_tolerance_ratio=0.0,
    )

    assert report.has_differences is False


def test_load_baseline_csv_rejects_invalid_bool(tmp_path: Path):
    path = tmp_path / "invalid.csv"
    path.write_text(
        ",".join(
            (
                "case_id",
                "strategy_id",
                "backend",
                "generator_id",
                "success",
                "wall_time_ms",
                "move_count_events",
                "move_count_lanes",
                "estimated_fidelity",
                "nodes_explored",
                "max_depth_reached",
                "notes",
            )
        )
        + "\n"
        + "ghz_4,python_entropy,python,heuristic,notabool,1.0,1,1,0.9,1,1,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid boolean"):
        load_baseline_csv(path)
