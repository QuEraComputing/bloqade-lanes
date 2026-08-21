from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from benchmarks.harness.models import BenchmarkCase, BenchmarkJob, StrategyConfig
from benchmarks.harness.runner import BenchmarkRunner, _count_moves
from kirin import ir

from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import SiteLaneAddress
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.physical.placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)
from bloqade.lanes.prelude import kernel as move_kernel


def test_estimate_fidelity_runs_for_logical_mode(monkeypatch):
    runner = BenchmarkRunner()
    expected_fidelity = 0.72
    calls: dict[str, object] = {}

    @dataclass
    class _FakePlacement:
        arch_spec: object

    class _FakeMoveMethod:
        pass

    class _FakePhysicalSquin:
        dialects = object()

    class _FakeGateFidelity:
        def __init__(self, min_value: float):
            self.min = min_value

    class _FakeFidelityAnalysis:
        def __init__(self, dialects):
            calls["analysis_dialects"] = dialects
            self.gate_fidelities = [_FakeGateFidelity(0.9), _FakeGateFidelity(0.8)]

        def run(self, kernel):
            calls["analysis_kernel"] = kernel

    class _FakeMoveToSquinLogical:
        def __init__(self, *, arch_spec, noise_model, add_noise, aggressive_unroll):
            calls["transform_arch_spec"] = arch_spec
            calls["transform_noise_model"] = noise_model
            calls["transform_add_noise"] = add_noise
            calls["transform_aggressive_unroll"] = aggressive_unroll

        def emit(self, move_mt):
            calls["emit_move_mt"] = move_mt
            return _FakePhysicalSquin()

    fake_move_mt = _FakeMoveMethod()
    fake_noise_model = object()

    def _fake_squin_to_move(*args, **kwargs):
        calls["squin_to_move_kwargs"] = kwargs
        return fake_move_mt

    def _fake_transversal_rewrites(move_mt):
        calls["transversal_input"] = move_mt
        return move_mt

    monkeypatch.setattr("benchmarks.harness.runner._squin_to_move", _fake_squin_to_move)
    monkeypatch.setattr(
        "benchmarks.harness.runner.transversal_rewrites", _fake_transversal_rewrites
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.generate_logical_noise_model",
        lambda: fake_noise_model,
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.MoveToSquinLogical", _FakeMoveToSquinLogical
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.FidelityAnalysis", _FakeFidelityAnalysis
    )

    job = BenchmarkJob(
        case=BenchmarkCase(
            case_id="ghz_6",
            kernel=cast(ir.Method, object()),
            logical_initialize=True,
        ),
        strategy=StrategyConfig(
            strategy_id="python_entropy",
            backend="python",
            generator_id="heuristic",
            build_placement_strategy=lambda: cast(
                PlacementStrategyABC, _FakePlacement(arch_spec=object())
            ),
        ),
    )

    fidelity = runner._estimate_fidelity(job)
    assert fidelity == pytest.approx(expected_fidelity)
    squin_to_move_kwargs = cast(dict[str, object], calls["squin_to_move_kwargs"])
    assert squin_to_move_kwargs["logical_initialize"] is True
    assert "insert_return_moves" not in squin_to_move_kwargs
    assert calls["transversal_input"] is fake_move_mt
    assert calls["emit_move_mt"] is fake_move_mt
    assert calls["transform_noise_model"] is fake_noise_model
    assert calls["transform_add_noise"] is True
    # Broadcasted state-prep loops must be fully unrolled so FidelityAnalysis
    # can see the per-qubit noise channels.
    assert calls["transform_aggressive_unroll"] is True


def test_estimate_fidelity_runs_for_physical_mode(monkeypatch):
    runner = BenchmarkRunner()
    expected_fidelity = 0.61

    class _FakeGateFidelity:
        min = expected_fidelity

    class _FakeFidelityAnalysis:
        def __init__(self, dialects):
            self.gate_fidelities = [_FakeGateFidelity()]

        def run(self, kernel):
            return None

    @dataclass
    class _FakePlacement:
        arch_spec: object

    class _FakePhysicalSquin:
        dialects = object()

    class _FakePhysicalPipeline:
        def __init__(self, **kwargs):
            pass

        def emit(self, kernel, **kwargs):
            return cast(ir.Method, object())

    class _FakeMoveToSquinPhysical:
        def __init__(self, **kwargs):
            pass

        def emit(self, move_mt, **kwargs):
            return _FakePhysicalSquin()

    monkeypatch.setattr(
        "benchmarks.harness.runner.PhysicalPipeline",
        _FakePhysicalPipeline,
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.MoveToSquinPhysical",
        _FakeMoveToSquinPhysical,
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.FidelityAnalysis", _FakeFidelityAnalysis
    )

    job = BenchmarkJob(
        case=BenchmarkCase(
            case_id="steane_physical_35",
            kernel=cast(ir.Method, object()),
            logical_initialize=False,
        ),
        strategy=StrategyConfig(
            strategy_id="rust_astar",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: cast(
                PlacementStrategyABC, _FakePlacement(arch_spec=object())
            ),
        ),
    )

    fidelity = runner._estimate_fidelity(job)
    assert fidelity == expected_fidelity


def test_run_jobs_stamps_arch_spec_id_from_strategy(monkeypatch):
    runner = BenchmarkRunner()

    class _FakeRegion:
        def walk(self):
            return ()

    class _FakeMoveMethod:
        callable_region = _FakeRegion()

    def _fake_squin_to_move(*args, **kwargs):
        return _FakeMoveMethod()

    monkeypatch.setattr("benchmarks.harness.runner._squin_to_move", _fake_squin_to_move)
    monkeypatch.setattr(BenchmarkRunner, "_estimate_fidelity", lambda self, job: 1.0)

    @dataclass
    class _FakePlacement:
        arch_spec: object

    jobs = [
        BenchmarkJob(
            case=BenchmarkCase(
                case_id="ghz_4",
                kernel=cast(ir.Method, object()),
                logical_initialize=False,
            ),
            strategy=StrategyConfig(
                strategy_id="rust_astar",
                backend="rust",
                generator_id="rust_solver",
                build_placement_strategy=lambda: cast(
                    PlacementStrategyABC, _FakePlacement(arch_spec=object())
                ),
                arch_spec_id=arch_id,
            ),
        )
        for arch_id in ("full", "simple")
    ]

    rows = runner.run_jobs(jobs)
    assert [row.arch_spec_id for row in rows] == ["full", "simple"]


def test_count_moves_ignores_empty_filler_lanes():
    # Physical-spec site bus 0 (src=[0,2,4,6], dst=[1,3,5,7]) on word 1:
    # the lane at site 0 carries the atom; the lane at site 2 is a valid
    # empty-source filler completing the AOD rectangle. Only the mover
    # counts toward move_count_lanes.
    @move_kernel
    def main():
        state0 = move.load()
        state1 = move.fill(state0, location_addresses=(move.LocationAddress(1, 0),))
        state2 = move.move(
            state1,
            lanes=(
                SiteLaneAddress(1, 0, 0),
                SiteLaneAddress(1, 2, 0),
            ),
        )
        move.store(state2)

    move_count_events, move_count_lanes = _count_moves(main, physical.get_arch_spec())

    assert move_count_events == 1
    assert move_count_lanes == 1


def test_compile_reads_rust_nodes_from_strategy(monkeypatch):
    runner = BenchmarkRunner()

    class _FakeRegion:
        def walk(self):
            return ()

    class _FakeMoveMethod:
        callable_region = _FakeRegion()

    def _fake_squin_to_move(*args, **kwargs):
        placement_strategy = kwargs["placement_strategy"]
        placement_strategy._rust_nodes_expanded_total = 321
        return _FakeMoveMethod()

    monkeypatch.setattr("benchmarks.harness.runner._squin_to_move", _fake_squin_to_move)

    job = BenchmarkJob(
        case=BenchmarkCase(
            case_id="steane_physical_35", kernel=cast(ir.Method, object())
        ),
        strategy=StrategyConfig(
            strategy_id="rust_astar",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: PhysicalPlacementStrategy(
                arch_spec=physical.get_arch_spec(),
                traversal=RustPlacementTraversal(),
            ),
        ),
    )

    artifacts = runner._compile(job)
    assert artifacts.nodes_explored == 321
    assert artifacts.notes == ""


def _fake_job(*, arch_spec: object, logical_initialize: bool) -> BenchmarkJob:
    @dataclass
    class _FakePlacement:
        arch_spec: object

    return BenchmarkJob(
        case=BenchmarkCase(
            case_id="case_x",
            kernel=cast(ir.Method, object()),
            logical_initialize=logical_initialize,
        ),
        strategy=StrategyConfig(
            strategy_id="rust_astar",
            backend="rust",
            generator_id="rust_solver",
            build_placement_strategy=lambda: cast(
                PlacementStrategyABC, _FakePlacement(arch_spec=arch_spec)
            ),
        ),
    )


@pytest.mark.parametrize("logical_initialize", [False, True])
def test_build_layout_heuristic_uses_the_strategy_arch_spec(logical_initialize):
    """The layout stage must target the strategy's arch, not a bundled default.

    Regression test: both heuristics used to be constructed with no arguments,
    so they silently fell back to the bundled Gemini specs and ``--arch-spec``
    only reached the placement stage.
    """
    sentinel = object()
    runner = BenchmarkRunner()
    job = _fake_job(arch_spec=sentinel, logical_initialize=logical_initialize)

    heuristic = runner._build_layout_heuristic(job, cast(ArchSpec, sentinel))

    assert heuristic.arch_spec is sentinel


def test_compile_threads_strategy_arch_spec_into_the_layout_heuristic(monkeypatch):
    """``_compile`` must hand the layout heuristic the strategy's arch spec."""
    sentinel = object()
    seen: dict[str, object] = {}

    def _fake_squin_to_move(mt, *, layout_heuristic, placement_strategy, **kwargs):
        seen["layout_arch_spec"] = layout_heuristic.arch_spec
        seen["placement_arch_spec"] = placement_strategy.arch_spec
        return cast(ir.Method, object())

    monkeypatch.setattr("benchmarks.harness.runner._squin_to_move", _fake_squin_to_move)
    monkeypatch.setattr(
        "benchmarks.harness.runner._assert_move_lowering_complete", lambda mt: None
    )

    artifacts = BenchmarkRunner()._compile(
        _fake_job(arch_spec=sentinel, logical_initialize=False)
    )

    assert seen["layout_arch_spec"] is sentinel
    assert seen["placement_arch_spec"] is sentinel
    assert artifacts.arch_spec is sentinel


def test_estimate_fidelity_physical_mode_uses_the_strategy_arch_spec(monkeypatch):
    """Fidelity must be measured against the strategy's arch, not bundled Gemini.

    The pipeline and the noise-insertion step both used a hardcoded
    ``get_physical_arch_spec()``, so a custom-arch run reported a fidelity for
    the wrong architecture (and failed lowering with a misleading
    "SSAValue ... stmt: fill ... not found").
    """
    sentinel = object()
    seen: dict[str, object] = {}

    class _FakeGateFidelity:
        min = 0.5

    class _FakeFidelityAnalysis:
        def __init__(self, dialects):
            self.gate_fidelities = [_FakeGateFidelity()]

        def run(self, kernel):
            return None

    class _FakePhysicalPipeline:
        def __init__(self, **kwargs):
            seen["pipeline_arch_spec"] = kwargs["arch_spec"]
            seen["pipeline_layout_arch_spec"] = kwargs["layout_heuristic"].arch_spec

        def emit(self, kernel, **kwargs):
            return cast(ir.Method, object())

    class _FakeMoveToSquinPhysical:
        def __init__(self, **kwargs):
            seen["noise_arch_spec"] = kwargs["arch_spec"]

        def emit(self, move_mt, **kwargs):
            class _S:
                dialects = object()

            return _S()

    monkeypatch.setattr(
        "benchmarks.harness.runner.PhysicalPipeline", _FakePhysicalPipeline
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.MoveToSquinPhysical", _FakeMoveToSquinPhysical
    )
    monkeypatch.setattr(
        "benchmarks.harness.runner.FidelityAnalysis", _FakeFidelityAnalysis
    )

    fidelity = BenchmarkRunner()._estimate_fidelity(
        _fake_job(arch_spec=sentinel, logical_initialize=False)
    )

    assert fidelity == 0.5
    # All three consumers must agree on the arch, or the plan and the noise
    # model describe different machines.
    assert seen["pipeline_arch_spec"] is sentinel
    assert seen["pipeline_layout_arch_spec"] is sentinel
    assert seen["noise_arch_spec"] is sentinel
