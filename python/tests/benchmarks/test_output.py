from pathlib import Path

from benchmarks.harness.models import BenchmarkRow
from benchmarks.harness.output import CSV_COLUMNS, sort_rows, write_csv


def test_sort_rows_is_deterministic():
    rows = [
        BenchmarkRow(
            case_id="b_case",
            strategy_id="z_strategy",
            backend="python",
            generator_id="heuristic",
            success=True,
            wall_time_ms=1.0,
            move_count_events=1,
            move_count_lanes=1,
            estimated_fidelity=0.9,
            nodes_explored=5,
            max_depth_reached=2,
        ),
        BenchmarkRow(
            case_id="a_case",
            strategy_id="a_strategy",
            backend="rust",
            generator_id="rust_solver",
            success=True,
            wall_time_ms=2.0,
            move_count_events=2,
            move_count_lanes=3,
            estimated_fidelity=0.8,
            nodes_explored=None,
            max_depth_reached=None,
        ),
    ]
    sorted_rows = sort_rows(rows)
    assert [row.case_id for row in sorted_rows] == ["a_case", "b_case"]


def test_csv_schema_column_order(tmp_path: Path):
    out = tmp_path / "bench.csv"
    rows = [
        BenchmarkRow(
            case_id="ghz_4",
            strategy_id="python_entropy",
            backend="python",
            generator_id="heuristic",
            success=True,
            wall_time_ms=3.14,
            move_count_events=2,
            move_count_lanes=4,
            estimated_fidelity=0.99,
            nodes_explored=7,
            max_depth_reached=3,
            notes="ok",
        )
    ]
    write_csv(rows, out)
    first_line = out.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == ",".join(CSV_COLUMNS)
