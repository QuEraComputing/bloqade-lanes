from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import benchmarks.cli as cli_module
import pytest
from benchmarks.cli import (
    _apply_architecture_mode,
    _build_parser,
    _filter_jobs_for_logical_capacity,
    _infer_case_qubit_count,
    _print_debug_job_start,
    _resolve_arch_specs,
    _resolve_output_path,
)
from benchmarks.harness.models import (
    BenchmarkCase,
    BenchmarkJob,
    BenchmarkRow,
    StrategyConfig,
)
from benchmarks.harness.output import write_csv
from benchmarks.harness.runner import BenchmarkRunner
from kirin import ir

from bloqade.lanes.analysis.placement import PlacementStrategyABC


def _build_job(case_id: str, strategy_id: str) -> BenchmarkJob:
    return BenchmarkJob(
        case=BenchmarkCase(case_id=case_id, kernel=cast(ir.Method, object())),
        strategy=StrategyConfig(
            strategy_id=strategy_id,
            backend="python",
            generator_id="heuristic",
            build_placement_strategy=lambda: cast(PlacementStrategyABC, object()),
        ),
    )


def test_print_debug_job_start_emits_kernel_and_strategy(capsys):
    job = _build_job("ghz_6", "python_entropy")
    _print_debug_job_start(job=job, new_case=True)
    captured = capsys.readouterr()

    assert captured.out.splitlines() == [
        "[debug] kernel: ghz_6",
        "[debug] start strategy: python_entropy (backend=python, generator=heuristic)",
    ]


def test_runner_run_jobs_emits_case_boundary_to_progress_callback(monkeypatch):
    jobs = [
        _build_job("ghz_4", "python_bfs"),
        _build_job("ghz_4", "python_entropy"),
        _build_job("ghz_6", "python_bfs"),
    ]
    runner = BenchmarkRunner()
    events: list[tuple[str, str, bool]] = []

    def fake_run_one(self, job):
        return BenchmarkRow(
            case_id=job.case.case_id,
            strategy_id=job.strategy.strategy_id,
            backend=job.strategy.backend,
            generator_id=job.strategy.generator_id,
            success=True,
            wall_time_ms=1.0,
            move_count_events=0,
            move_count_lanes=0,
            estimated_fidelity=1.0,
            nodes_explored=1,
            max_depth_reached=1,
        )

    monkeypatch.setattr(BenchmarkRunner, "_run_one", fake_run_one)
    rows = runner.run_jobs(
        jobs,
        on_job_start=lambda job, new_case: events.append(
            (job.case.case_id, job.strategy.strategy_id, new_case)
        ),
    )

    assert len(rows) == 3
    assert events == [
        ("ghz_4", "python_bfs", True),
        ("ghz_4", "python_entropy", False),
        ("ghz_6", "python_bfs", True),
    ]


def test_cli_architecture_defaults_to_case_default():
    parser = _build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args([])
    assert exc_info.value.code == 2


def test_apply_architecture_mode_overrides_jobs_to_logical():
    logical_job = _build_job("ghz_4", "python_bfs")
    physical_job = BenchmarkJob(
        case=replace(
            logical_job.case,
            case_id="steane_physical_35",
            logical_initialize=False,
        ),
        strategy=logical_job.strategy,
    )

    updated_jobs = _apply_architecture_mode(
        [logical_job, physical_job], architecture_mode="logical"
    )

    assert updated_jobs[0].case.logical_initialize is True
    assert updated_jobs[1].case.logical_initialize is True


def test_apply_architecture_mode_overrides_jobs_to_physical():
    logical_job = _build_job("ghz_4", "python_bfs")
    physical_job = BenchmarkJob(
        case=replace(
            logical_job.case, case_id="steane_physical_35", logical_initialize=False
        ),
        strategy=logical_job.strategy,
    )

    updated_jobs = _apply_architecture_mode(
        [logical_job, physical_job], architecture_mode="physical"
    )

    assert all(job.case.logical_initialize is False for job in updated_jobs)


def test_infer_case_qubit_count_from_case_id():
    assert _infer_case_qubit_count("ghz_6") == 6
    assert _infer_case_qubit_count("steane_physical_35") == 35
    assert _infer_case_qubit_count("custom_case") is None


def test_filter_jobs_for_logical_capacity_skips_large_cases(capsys):
    jobs = [
        _build_job("ghz_6", "python_entropy"),
        _build_job("adder_64", "python_entropy"),
        _build_job("qpe_9", "python_entropy"),
    ]
    filtered_jobs = _filter_jobs_for_logical_capacity(jobs, max_logical_qubits=10)

    assert [job.case.case_id for job in filtered_jobs] == ["ghz_6", "qpe_9"]
    captured = capsys.readouterr()
    assert (
        "logical mode skipped cases exceeding logical qubit capacity (10)"
        in captured.out
    )
    assert "adder_64" in captured.out


def test_resolve_output_path_uses_architecture_specific_defaults():
    assert _resolve_output_path("", architecture_mode="physical") == Path(
        "python/benchmarks/harness/latest_physical.csv"
    )
    assert _resolve_output_path("", architecture_mode="logical") == Path(
        "python/benchmarks/harness/latest_logical.csv"
    )
    assert _resolve_output_path(
        "custom/output.csv", architecture_mode="physical"
    ) == Path("custom/output.csv")


def _row(
    *,
    success: bool = True,
    wall_time_ms: float | None = 1.0,
    move_count_events: int | None = 0,
    move_count_lanes: int | None = 0,
) -> BenchmarkRow:
    return BenchmarkRow(
        case_id="ghz_4",
        strategy_id="python_bfs",
        backend="python",
        generator_id="heuristic",
        success=success,
        wall_time_ms=wall_time_ms,
        move_count_events=move_count_events,
        move_count_lanes=move_count_lanes,
        estimated_fidelity=1.0 if success else None,
        nodes_explored=1 if success else None,
        max_depth_reached=1 if success else None,
    )


def _mock_cli_run(monkeypatch, rows: list[BenchmarkRow]):
    case = BenchmarkCase(case_id="ghz_4", kernel=cast(ir.Method, object()))
    strategy = StrategyConfig(
        strategy_id="python_bfs",
        backend="python",
        generator_id="heuristic",
        build_placement_strategy=lambda: cast(PlacementStrategyABC, object()),
    )
    jobs = [BenchmarkJob(case=case, strategy=strategy)]

    monkeypatch.setattr(
        cli_module, "select_benchmark_cases", lambda case_filter: (case,)
    )
    monkeypatch.setattr(
        cli_module,
        "default_strategy_configs",
        lambda arch_spec=None: (strategy,),
    )
    monkeypatch.setattr(
        cli_module,
        "expand_benchmark_jobs",
        lambda cases, strategies, strategy_filter: jobs,
    )
    monkeypatch.setattr(
        BenchmarkRunner, "run_jobs", lambda self, jobs, on_job_start: rows
    )


def test_cli_parser_accepts_compare_flag():
    parser = _build_parser()
    args = parser.parse_args(
        ["--compare", "baseline.csv", "--architecture", "physical"]
    )
    assert args.compare == "baseline.csv"


def test_main_warns_and_continues_when_compare_file_missing(
    monkeypatch, capsys, tmp_path: Path
):
    _mock_cli_run(monkeypatch, rows=[_row()])
    output_path = tmp_path / "latest.csv"
    missing_baseline = tmp_path / "missing.csv"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.cli",
            "--output",
            str(output_path),
            "--compare",
            str(missing_baseline),
            "--architecture",
            "physical",
        ],
    )

    assert cli_module.main() == 0
    captured = capsys.readouterr()
    assert "WARNING: compare baseline CSV was not found" in captured.out
    assert str(missing_baseline) in captured.out
    assert output_path.exists()


def test_main_without_compare_emits_warning(monkeypatch, capsys, tmp_path: Path):
    _mock_cli_run(monkeypatch, rows=[_row()])
    output_path = tmp_path / "latest.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmarks.cli", "--output", str(output_path), "--architecture", "physical"],
    )

    assert cli_module.main() == 0
    captured = capsys.readouterr()
    assert "WARNING: no baseline CSV provided via --compare" in captured.out


def test_main_compare_returns_nonzero_when_diffs_found(
    monkeypatch, capsys, tmp_path: Path
):
    _mock_cli_run(monkeypatch, rows=[_row(move_count_events=2)])
    output_path = tmp_path / "latest.csv"
    baseline_path = tmp_path / "baseline.csv"
    baseline_rows = [_row(move_count_events=1)]

    write_csv(baseline_rows, baseline_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.cli",
            "--output",
            str(output_path),
            "--compare",
            str(baseline_path),
            "--architecture",
            "physical",
        ],
    )

    assert cli_module.main() == 1
    captured = capsys.readouterr()
    assert "row differences" in captured.out


def test_main_compare_can_use_same_file_for_output_and_baseline(
    monkeypatch, capsys, tmp_path: Path
):
    _mock_cli_run(monkeypatch, rows=[_row(move_count_events=2)])
    baseline_path = tmp_path / "latest.csv"
    baseline_rows = [_row(move_count_events=1)]

    write_csv(baseline_rows, baseline_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.cli",
            "--output",
            str(baseline_path),
            "--compare",
            str(baseline_path),
            "--architecture",
            "physical",
        ],
    )

    assert cli_module.main() == 1
    captured = capsys.readouterr()
    assert "row differences" in captured.out


def test_main_logical_mode_skips_large_cases(monkeypatch, capsys, tmp_path: Path):
    strategy = StrategyConfig(
        strategy_id="python_bfs",
        backend="python",
        generator_id="heuristic",
        build_placement_strategy=lambda: cast(PlacementStrategyABC, object()),
    )
    cases = (
        BenchmarkCase(case_id="ghz_6", kernel=cast(ir.Method, object())),
        BenchmarkCase(case_id="adder_64", kernel=cast(ir.Method, object())),
    )
    jobs = [BenchmarkJob(case=case, strategy=strategy) for case in cases]
    seen_jobs: list[BenchmarkJob] = []

    monkeypatch.setattr(cli_module, "select_benchmark_cases", lambda case_filter: cases)
    monkeypatch.setattr(
        cli_module,
        "default_strategy_configs",
        lambda arch_spec=None: (strategy,),
    )
    monkeypatch.setattr(
        cli_module,
        "expand_benchmark_jobs",
        lambda cases, strategies, strategy_filter: jobs,
    )

    def fake_run_jobs(self, jobs, on_job_start):
        seen_jobs.extend(jobs)
        return [_row()]

    monkeypatch.setattr(BenchmarkRunner, "run_jobs", fake_run_jobs)

    output_path = tmp_path / "logical-latest.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.cli",
            "--output",
            str(output_path),
            "--architecture",
            "logical",
        ],
    )

    assert cli_module.main() == 0
    assert [job.case.case_id for job in seen_jobs] == ["ghz_6"]
    captured = capsys.readouterr()
    assert "adder_64" in captured.out


def test_resolve_arch_specs_none_returns_builtin_pair():
    pairs = _resolve_arch_specs(None)
    assert len(pairs) == 1
    arch_id, _arch = pairs[0]
    assert arch_id == "builtin"


def test_resolve_arch_specs_empty_list_returns_builtin_pair():
    pairs = _resolve_arch_specs([])
    assert len(pairs) == 1
    assert pairs[0][0] == "builtin"


def test_resolve_arch_specs_uses_filename_stem_as_id():
    # Use the real archspec fixture since it's guaranteed to parse.
    simple_path = Path("examples/arch/simple.json")
    pairs = _resolve_arch_specs([str(simple_path)])
    assert len(pairs) == 1
    assert pairs[0][0] == "simple"


def test_resolve_arch_specs_raises_on_missing_path(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ValueError, match="does_not_exist"):
        _resolve_arch_specs([str(missing)])


def test_resolve_arch_specs_raises_on_malformed_json(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.json"):
        _resolve_arch_specs([str(bad)])


def test_resolve_arch_specs_raises_on_reserved_builtin_stem(tmp_path):
    src = Path("examples/arch/full.json").read_text(encoding="utf-8")
    reserved = tmp_path / "builtin.json"
    reserved.write_text(src, encoding="utf-8")
    with pytest.raises(ValueError, match="reserved"):
        _resolve_arch_specs([str(reserved)])


def test_resolve_arch_specs_raises_on_duplicate_stem(tmp_path):
    sub_a = tmp_path / "a"
    sub_b = tmp_path / "b"
    sub_a.mkdir()
    sub_b.mkdir()
    # Both files have stem "full".
    src = Path("examples/arch/full.json").read_text(encoding="utf-8")
    (sub_a / "full.json").write_text(src, encoding="utf-8")
    (sub_b / "full.json").write_text(src, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _resolve_arch_specs([str(sub_a / "full.json"), str(sub_b / "full.json")])


def test_resolve_arch_specs_preserves_argument_order():
    pairs = _resolve_arch_specs(
        ["examples/arch/simple.json", "examples/arch/full.json"]
    )
    assert [pair[0] for pair in pairs] == ["simple", "full"]


def test_cli_parser_accepts_arch_spec_flag():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--architecture",
            "physical",
            "--arch-spec",
            "a.json",
            "b.json",
        ]
    )
    assert args.arch_spec == ["a.json", "b.json"]


def test_cli_parser_arch_spec_default_is_none():
    parser = _build_parser()
    args = parser.parse_args(["--architecture", "physical"])
    assert args.arch_spec is None


def test_main_arch_spec_expands_matrix_across_archspecs(
    monkeypatch, capsys, tmp_path: Path
):
    case = BenchmarkCase(case_id="ghz_4", kernel=cast(ir.Method, object()))
    # Rebuild strategies each invocation so arch_spec_id is unique per call.
    captured_strategies: list[tuple[StrategyConfig, ...]] = []

    def fake_default_strategy_configs(arch_spec=None):
        arch_spec_id = "builtin" if arch_spec is None else arch_spec[0]
        strategy = StrategyConfig(
            strategy_id="python_bfs",
            backend="python",
            generator_id="heuristic",
            build_placement_strategy=lambda: cast(PlacementStrategyABC, object()),
            arch_spec_id=arch_spec_id,
        )
        captured_strategies.append((strategy,))
        return (strategy,)

    seen_jobs: list[BenchmarkJob] = []

    def fake_run_jobs(self, jobs, on_job_start):
        seen_jobs.extend(jobs)
        return [
            BenchmarkRow(
                case_id=job.case.case_id,
                strategy_id=job.strategy.strategy_id,
                backend=job.strategy.backend,
                generator_id=job.strategy.generator_id,
                success=True,
                wall_time_ms=1.0,
                move_count_events=0,
                move_count_lanes=0,
                estimated_fidelity=1.0,
                nodes_explored=1,
                max_depth_reached=1,
                arch_spec_id=job.strategy.arch_spec_id,
            )
            for job in jobs
        ]

    monkeypatch.setattr(
        cli_module, "select_benchmark_cases", lambda case_filter: (case,)
    )
    monkeypatch.setattr(
        cli_module, "default_strategy_configs", fake_default_strategy_configs
    )
    monkeypatch.setattr(BenchmarkRunner, "run_jobs", fake_run_jobs)

    output_path = tmp_path / "bench.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.cli",
            "--output",
            str(output_path),
            "--architecture",
            "physical",
            "--arch-spec",
            "examples/arch/simple.json",
            "examples/arch/full.json",
        ],
    )

    assert cli_module.main() == 0
    # One case × one strategy × two archspecs = 2 jobs.
    assert len(seen_jobs) == 2
    assert sorted(job.strategy.arch_spec_id for job in seen_jobs) == ["full", "simple"]

    # CSV output should contain both archspec ids.
    csv_text = output_path.read_text(encoding="utf-8")
    assert "simple" in csv_text
    assert "full" in csv_text


def test_main_without_arch_spec_uses_builtin_id(monkeypatch, tmp_path: Path):
    case = BenchmarkCase(case_id="ghz_4", kernel=cast(ir.Method, object()))

    def fake_default_strategy_configs(arch_spec=None):
        arch_spec_id = "builtin" if arch_spec is None else arch_spec[0]
        return (
            StrategyConfig(
                strategy_id="python_bfs",
                backend="python",
                generator_id="heuristic",
                build_placement_strategy=lambda: cast(PlacementStrategyABC, object()),
                arch_spec_id=arch_spec_id,
            ),
        )

    seen_jobs: list[BenchmarkJob] = []

    def fake_run_jobs(self, jobs, on_job_start):
        seen_jobs.extend(jobs)
        return [
            BenchmarkRow(
                case_id=job.case.case_id,
                strategy_id=job.strategy.strategy_id,
                backend=job.strategy.backend,
                generator_id=job.strategy.generator_id,
                success=True,
                wall_time_ms=1.0,
                move_count_events=0,
                move_count_lanes=0,
                estimated_fidelity=1.0,
                nodes_explored=1,
                max_depth_reached=1,
                arch_spec_id=job.strategy.arch_spec_id,
            )
            for job in jobs
        ]

    monkeypatch.setattr(
        cli_module, "select_benchmark_cases", lambda case_filter: (case,)
    )
    monkeypatch.setattr(
        cli_module, "default_strategy_configs", fake_default_strategy_configs
    )
    monkeypatch.setattr(BenchmarkRunner, "run_jobs", fake_run_jobs)

    output_path = tmp_path / "bench.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmarks.cli",
            "--output",
            str(output_path),
            "--architecture",
            "physical",
        ],
    )

    assert cli_module.main() == 0
    assert [job.strategy.arch_spec_id for job in seen_jobs] == ["builtin"]
