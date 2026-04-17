from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
from benchmarks.harness.models import BenchmarkCase, BenchmarkJob, StrategyConfig
from benchmarks.harness.runner import BenchmarkRunner
from kirin import ir

from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.arch.gemini import physical
from bloqade.lanes.heuristics.physical_placement import (
    PhysicalPlacementStrategy,
    RustPlacementTraversal,
)


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
            return None

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

    monkeypatch.setattr("benchmarks.harness.runner.squin_to_move", _fake_squin_to_move)
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
    assert squin_to_move_kwargs["insert_return_moves"] is True
    assert squin_to_move_kwargs["logical_initialize"] is True
    assert calls["transversal_input"] is fake_move_mt
    assert calls["emit_move_mt"] is fake_move_mt
    assert calls["transform_noise_model"] is fake_noise_model
    assert calls["transform_add_noise"] is True
    assert calls["transform_aggressive_unroll"] is False


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

    monkeypatch.setattr(
        "benchmarks.harness.runner.compile_physical_noise_model",
        lambda *args, **kwargs: _FakePhysicalSquin(),
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

    monkeypatch.setattr("benchmarks.harness.runner.squin_to_move", _fake_squin_to_move)

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
