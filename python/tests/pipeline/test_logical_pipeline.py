"""Tests for LogicalPipeline."""

import bloqade.squin as squin
from bloqade.squin.gate.stmts import S, SqrtX

import bloqade.gemini as gemini
from bloqade.lanes.dialects import move, place
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.pipeline import LogicalPipeline
from bloqade.lanes.pipeline.logical import _LogicalNativeToPlace


def test_logical_pipeline_smoke():
    """2-qubit Bell kernel compiles end-to-end via LogicalPipeline."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    out = LogicalPipeline().emit(kernel)
    assert out is not None
    fills = [s for s in out.callable_region.walk() if isinstance(s, move.Fill)]
    assert len(fills) == 1


def test_logical_pre_native_rewrites_steane_transversal_adjoints():
    """With transversal_rewrite=True, pre-native rewrites swap Steane transversal
    Clifford adjoints."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.sqrt_x(reg[0])
        squin.sqrt_x_adj(reg[0])
        squin.s(reg[0])
        squin.s_adj(reg[0])
        gemini.logical.terminal_measure(reg)

    out = kernel.similar(kernel.dialects.add(place))
    _LogicalNativeToPlace(transversal_rewrite=True)._pre_native_rewrites(
        kernel, out, no_raise=True
    )

    sqrt_x_gates = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, SqrtX)
    ]
    s_gates = [stmt for stmt in out.callable_region.walk() if isinstance(stmt, S)]

    assert [stmt.adjoint for stmt in sqrt_x_gates] == [True, False]
    assert [stmt.adjoint for stmt in s_gates] == [True, False]


def test_logical_pipeline_produces_logical_initialize():
    """Logical pipeline inserts LogicalInitialize (not just Fill)."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    out = LogicalPipeline().emit(kernel)
    inits = [
        s for s in out.callable_region.walk() if isinstance(s, move.LogicalInitialize)
    ]
    assert len(inits) >= 1


def test_logical_pipeline_no_return_moves():
    """Passing a bare (non-palindrome) strategy disables return moves."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        gemini.logical.terminal_measure(reg)

    from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_arch_spec

    arch_spec = get_logical_arch_spec()
    out = LogicalPipeline(
        arch_spec=arch_spec,
        placement_strategy=LogicalPlacementStrategyNoHome(arch_spec=arch_spec),
    ).emit(kernel)
    assert out is not None


def test_logical_pipeline_layout_heuristic_default_is_none():
    """LogicalPipeline.layout_heuristic defaults to None."""
    pipeline = LogicalPipeline()
    assert pipeline.layout_heuristic is None


def test_logical_pipeline_resolves_none_to_logical_defaults(monkeypatch):
    """When layout_heuristic is None, LogicalPipeline passes LogicalLayoutHeuristic
    to the place→move stage."""
    from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
    from bloqade.lanes.pipeline.base import _PlaceToMove

    captured: dict = {}
    _orig_emit = _PlaceToMove.emit

    def spy_emit(self_inner, mt, no_raise=True):
        captured["layout_heuristic_type"] = type(self_inner.layout_heuristic)
        return _orig_emit(self_inner, mt, no_raise=no_raise)

    monkeypatch.setattr(_PlaceToMove, "emit", spy_emit)

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    LogicalPipeline().emit(kernel)
    assert captured["layout_heuristic_type"] is LogicalLayoutHeuristic


def test_logical_pipeline_no_raise_suppresses_validation():
    """no_raise=True does not raise even when pre-native validation fails."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def invalid_kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cx(reg[0], reg[1])
        # missing terminal_measure — violates GeminiTerminalMeasurementValidation

    out = LogicalPipeline().emit(invalid_kernel, no_raise=True)
    assert out is not None
