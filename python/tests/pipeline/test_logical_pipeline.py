"""Tests for LogicalPipeline."""

import bloqade.squin as squin
import pytest
from bloqade.squin.gate.stmts import S, SqrtX

import bloqade.gemini as gemini
from bloqade.lanes.dialects import move, place
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.pipeline import LogicalPipeline
from bloqade.lanes.pipeline.logical import _LogicalNativeToPlace, transversal_rewrites


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


def test_logical_pipeline_layout_heuristic_mismatch_warns():
    """resolved_layout_heuristic warns when the explicit heuristic carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )

    logical_arch = LogicalPipeline().arch_spec
    physical_arch = get_physical_arch_spec()
    assert logical_arch != physical_arch

    mismatched_heuristic = LogicalLayoutHeuristic(arch_spec=physical_arch)
    pipeline = LogicalPipeline(layout_heuristic=mismatched_heuristic)

    with pytest.warns(
        UserWarning, match="layout_heuristic was constructed with a different"
    ):
        result = pipeline.resolved_layout_heuristic

    assert result is mismatched_heuristic


def test_logical_pipeline_placement_strategy_mismatch_warns():
    """resolved_placement_strategy warns when the explicit strategy carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )

    logical_arch = LogicalPipeline().arch_spec
    physical_arch = get_physical_arch_spec()
    assert logical_arch != physical_arch

    mismatched_strategy = LogicalPlacementStrategyNoHome(arch_spec=physical_arch)
    pipeline = LogicalPipeline(placement_strategy=mismatched_strategy)

    with pytest.warns(
        UserWarning, match="placement_strategy was constructed with a different"
    ):
        result = pipeline.resolved_placement_strategy

    assert result is mismatched_strategy


def test_transversal_rewrites_direct():
    """transversal_rewrites() rewrites the method in place and returns it."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = squin.qalloc(1)
        squin.h(reg[0])
        gemini.logical.terminal_measure(reg)

    result = transversal_rewrites(kernel, rewrite_logical_initialize=False)
    assert result is kernel
