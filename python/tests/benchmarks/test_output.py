from pathlib import Path

from benchmarks.harness.models import BenchmarkRow
from benchmarks.harness.output import (
    CSV_COLUMNS,
    EXTRA_COLUMNS,
    render_console_table,
    sort_rows,
    write_csv,
)


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


def test_csv_columns_include_arch_spec_id_after_strategy_id():
    strategy_idx = CSV_COLUMNS.index("strategy_id")
    assert CSV_COLUMNS[strategy_idx + 1] == "arch_spec_id"


def test_csv_writes_arch_spec_id_value(tmp_path: Path):
    out = tmp_path / "bench.csv"
    rows = [
        BenchmarkRow(
            case_id="ghz_4",
            strategy_id="python_entropy",
            backend="python",
            generator_id="heuristic",
            success=True,
            wall_time_ms=1.0,
            move_count_events=0,
            move_count_lanes=0,
            estimated_fidelity=1.0,
            nodes_explored=1,
            max_depth_reached=1,
            arch_spec_id="full",
        )
    ]
    write_csv(rows, out)
    lines = out.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    data = lines[1].split(",")
    assert data[header.index("arch_spec_id")] == "full"


def test_sort_rows_breaks_ties_by_arch_spec_id():
    rows = [
        BenchmarkRow(
            case_id="c",
            strategy_id="s",
            backend="python",
            generator_id="heuristic",
            success=True,
            wall_time_ms=1.0,
            move_count_events=0,
            move_count_lanes=0,
            estimated_fidelity=1.0,
            nodes_explored=1,
            max_depth_reached=1,
            arch_spec_id="z",
        ),
        BenchmarkRow(
            case_id="c",
            strategy_id="s",
            backend="python",
            generator_id="heuristic",
            success=True,
            wall_time_ms=1.0,
            move_count_events=0,
            move_count_lanes=0,
            estimated_fidelity=1.0,
            nodes_explored=1,
            max_depth_reached=1,
            arch_spec_id="a",
        ),
    ]
    sorted_rows = sort_rows(rows)
    assert [row.arch_spec_id for row in sorted_rows] == ["a", "z"]


def test_benchmark_row_arch_spec_id_defaults_to_builtin():
    row = BenchmarkRow(
        case_id="ghz_4",
        strategy_id="python_entropy",
        backend="python",
        generator_id="heuristic",
        success=True,
        wall_time_ms=1.0,
        move_count_events=0,
        move_count_lanes=0,
        estimated_fidelity=1.0,
        nodes_explored=1,
        max_depth_reached=1,
    )
    assert row.arch_spec_id == "builtin"


def _row(case_id: str, extra: dict[str, object] | None = None) -> BenchmarkRow:
    return BenchmarkRow(
        case_id=case_id,
        strategy_id="rust_entropy_bounded",
        backend="rust",
        generator_id="rust_solver",
        success=True,
        wall_time_ms=1.0,
        move_count_events=4,
        move_count_lanes=6,
        estimated_fidelity=0.5,
        nodes_explored=9,
        max_depth_reached=3,
        extra=dict(extra or {}),
    )


def test_console_table_renders_the_root_bound_block():
    """A row carrying ``extra`` gains the bound columns, values and all."""
    table = render_console_table(
        [
            _row(
                "bounded",
                {
                    "h_root_sum": "99.000",
                    "cost_sum": "101.000",
                    "gap_pct": "2%",
                    "certificates": "12/36",
                    "proven_solves": 12,
                },
            )
        ]
    )
    header = table.splitlines()[0]
    for name, _ in EXTRA_COLUMNS:
        assert name in header, f"{name!r} missing from {header!r}"
    assert "99.000" in table and "101.000" in table
    assert "12/36" in table and "2%" in table


def test_console_table_omits_the_block_without_extras():
    """A run with no bound rows renders exactly as it did before."""
    header = render_console_table([_row("plain")]).splitlines()[0]
    for name, _ in EXTRA_COLUMNS:
        assert name not in header


def test_console_table_pads_rows_missing_an_extra_key():
    """One bounded row does not force values onto the unbounded rows.

    The block is rendered whenever *any* row has extras, so the rows without
    them must still produce a cell per column — otherwise the table's columns
    stop lining up with its header.
    """
    table = render_console_table(
        [_row("bounded", {"h_root_sum": "5.000"}), _row("unbounded")]
    )
    lines = [line for line in table.splitlines() if "|" in line]
    counts = {line.count("|") for line in lines}
    assert len(counts) == 1, f"ragged table: {counts}"
