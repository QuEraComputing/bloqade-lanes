"""CSV and console output helpers for benchmark results."""

from __future__ import annotations

import csv
from pathlib import Path

from benchmarks.harness.models import BenchmarkRow

CSV_COLUMNS: tuple[str, ...] = (
    "case_id",
    "strategy_id",
    "arch_spec_id",
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

FLOAT_DECIMALS = 6
FIDELITY_DECIMALS = 10


def sort_rows(rows: list[BenchmarkRow]) -> list[BenchmarkRow]:
    """Sort rows deterministically for stable diffs."""
    return sorted(
        rows, key=lambda row: (row.case_id, row.strategy_id, row.arch_spec_id)
    )


def write_csv(rows: list[BenchmarkRow], output_path: Path) -> None:
    """Write benchmark rows to CSV with stable schema and order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_rows(rows)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "strategy_id": row.strategy_id,
                    "arch_spec_id": row.arch_spec_id,
                    "backend": row.backend,
                    "generator_id": row.generator_id,
                    "success": row.success,
                    "wall_time_ms": _fmt_float(row.wall_time_ms),
                    "move_count_events": row.move_count_events,
                    "move_count_lanes": row.move_count_lanes,
                    "estimated_fidelity": _fmt_fidelity(row.estimated_fidelity),
                    "nodes_explored": row.nodes_explored,
                    "max_depth_reached": row.max_depth_reached,
                    "notes": row.notes,
                }
            )


def render_console_table(rows: list[BenchmarkRow]) -> str:
    """Render a compact plain-text comparison table."""
    sorted_rows = sort_rows(rows)
    headers = (
        "case_id",
        "strategy_id",
        "ok",
        "wall_ms",
        "move_events",
        "move_lanes",
        "fidelity",
        "nodes_explored",
    )
    table_rows = [
        (
            row.case_id,
            row.strategy_id,
            str(row.success),
            _fmt_float(row.wall_time_ms),
            _fmt_int(row.move_count_events),
            _fmt_int(row.move_count_lanes),
            _fmt_fidelity(row.estimated_fidelity),
            _fmt_int(row.nodes_explored),
        )
        for row in sorted_rows
    ]
    widths = [len(h) for h in headers]
    for table_row in table_rows:
        for idx, value in enumerate(table_row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values: tuple[str, ...]) -> str:
        return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [fmt_row(headers), "-+-".join("-" * width for width in widths)]
    lines.extend(fmt_row(row) for row in table_rows)
    return "\n".join(lines)


def _fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.{FLOAT_DECIMALS}f}"


def _fmt_fidelity(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.{FIDELITY_DECIMALS}f}"


def _fmt_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(value)
