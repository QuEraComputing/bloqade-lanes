"""Tests for LogicalPipeline."""

import bloqade.squin as squin

import bloqade.gemini as gemini
from bloqade.lanes.dialects import move
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.pipeline import LogicalPipeline


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

    out = LogicalPipeline(placement_strategy=LogicalPlacementStrategyNoHome()).emit(
        kernel
    )
    assert out is not None


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
