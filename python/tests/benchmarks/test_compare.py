from dataclasses import replace
from pathlib import Path

import pytest
from benchmarks.harness.compare import (
    BenchmarkComparisonReport,
    compare_against_baseline,
    load_baseline_csv,
    render_comparison_report,
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


def test_load_baseline_csv_defaults_missing_arch_spec_id_to_builtin(tmp_path: Path):
    path = tmp_path / "legacy_baseline.csv"
    # Legacy schema: no arch_spec_id column.
    path.write_text(
        "case_id,strategy_id,backend,generator_id,success,wall_time_ms,"
        "move_count_events,move_count_lanes,estimated_fidelity,nodes_explored,"
        "max_depth_reached,notes\n"
        "ghz_4,python_entropy,python,heuristic,True,10.0,10,10,0.9,10,10,\n",
        encoding="utf-8",
    )
    loaded = load_baseline_csv(path)
    assert len(loaded) == 1
    assert loaded[0].arch_spec_id == "builtin"


def test_load_baseline_csv_preserves_arch_spec_id_when_present(tmp_path: Path):
    rows = [_row()]
    # Force a non-default arch_spec_id by replacing on the dataclass.
    from dataclasses import replace

    rows = [replace(rows[0], arch_spec_id="full")]
    path = tmp_path / "baseline.csv"
    write_csv(rows, path)
    loaded = load_baseline_csv(path)
    assert loaded[0].arch_spec_id == "full"


def test_compare_detects_matrix_mismatch():
    baseline_rows = [_row(case_id="ghz_4")]
    current_rows = [_row(case_id="ghz_4"), _row(case_id="ghz_6")]
    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )

    assert report.matrix_mismatch is True
    assert report.has_differences is True
    assert report.missing_from_baseline == (("ghz_6", "python_entropy", "builtin"),)
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


def test_compare_never_diffs_wall_time_regressions():
    baseline_rows = [_row(wall_time_ms=100.0)]
    current_rows = [_row(wall_time_ms=299.0)]
    report = compare_against_baseline(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
    )

    assert report.has_differences is False
    assert report.diffs == ()


def test_compare_ignores_wall_time_improvements():
    baseline_rows = [_row(wall_time_ms=1000.0)]
    current_rows = [_row(wall_time_ms=10.0)]

    report = compare_against_baseline(
        current_rows=current_rows,
        baseline_rows=baseline_rows,
    )

    assert report.has_differences is False


def test_compare_distinguishes_rows_by_arch_spec_id():
    baseline_rows = [
        replace(_row(), arch_spec_id="builtin"),
        replace(_row(), arch_spec_id="full"),
    ]
    # Current only has the "full" archspec — the "builtin" entry is missing.
    current_rows = [replace(_row(), arch_spec_id="full")]

    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )

    assert report.missing_from_current == (("ghz_4", "python_entropy", "builtin"),)
    assert report.missing_from_baseline == ()


def test_compare_treats_same_case_strategy_across_archspecs_as_distinct():
    baseline_rows = [
        replace(_row(move_count_events=10), arch_spec_id="full"),
        replace(_row(move_count_events=10), arch_spec_id="simple"),
    ]
    current_rows = [
        replace(_row(move_count_events=10), arch_spec_id="full"),
        replace(_row(move_count_events=20), arch_spec_id="simple"),
    ]

    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )

    assert len(report.diffs) == 1
    assert report.diffs[0].arch_spec_id == "simple"
    assert report.diffs[0].field == "move_count_events"


def test_render_comparison_report_includes_arch_spec_id():
    baseline_rows = [replace(_row(move_count_events=10), arch_spec_id="full")]
    current_rows = [replace(_row(move_count_events=20), arch_spec_id="full")]

    report = compare_against_baseline(
        current_rows=current_rows, baseline_rows=baseline_rows
    )
    rendered = render_comparison_report(report)
    assert "full" in rendered


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
