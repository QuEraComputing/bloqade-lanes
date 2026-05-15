from bloqade.decoders.dialects.annotate.stmts import SetDetector
from bloqade.squin.gate.stmts import U3
from bloqade.test_utils import assert_nodes
from bloqade.types import MeasurementResultType
from kirin import ir, rewrite, types
from kirin.dialects import ilist, py

from bloqade import squin
from bloqade.gemini import logical
from bloqade.gemini.logical.dialects.operations.stmts import Initialize
from bloqade.gemini.logical.rewrite.initialize import _RewriteU3ToInitialize
from bloqade.gemini.logical.rewrite.remove_postprocessing import (
    RemovePostProcessing,
)

TerminalMeasureRetType = ilist.IListType[
    ilist.IListType[MeasurementResultType, types.Any], types.Any
]


def test_rewrite_u3_to_initialize():
    theta = ir.TestValue()
    phi = ir.TestValue()
    qubits = ir.TestValue()
    test_block = ir.Block(
        [
            lam_stmt := py.Constant(1.0),
            U3(theta, phi, lam_stmt.result, qubits),
        ]
    )

    expected_block = ir.Block(
        [
            lam_stmt := py.Constant(1.0),
            Initialize(theta, phi, lam_stmt.result, qubits),
        ]
    )

    rewrite.Walk(_RewriteU3ToInitialize()).rewrite(test_block)

    assert_nodes(test_block, expected_block)


def test_remove_postprocessing():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        squin.broadcast.h(q)

        logical.terminal_measure(q)

    result = RemovePostProcessing(main.dialects)(main)

    assert result.has_done_something

    # check that calling twice doesn't do anything
    result = RemovePostProcessing(main.dialects)(main)
    assert not result.has_done_something

    assert main.return_type.is_subseteq(TerminalMeasureRetType)


def test_remove_postprocessing_with_uses():
    @logical.kernel(aggressive_unroll=True)
    def main():
        q = squin.qalloc(2)
        m = logical.terminal_measure(q)
        det = squin.set_detector(ilist.IList([m[0][0], m[1][0]]), [0, 1])
        return det

    # check that we have a detector there
    assert any(isinstance(stmt, SetDetector) for stmt in main.callable_region.stmts())

    result = RemovePostProcessing(main.dialects)(main)
    assert result.has_done_something

    assert main.return_type.is_subseteq(TerminalMeasureRetType)

    # check that calling twice doesn't do anything
    result = RemovePostProcessing(main.dialects)(main)
    assert not result.has_done_something

    assert not any(
        isinstance(stmt, SetDetector) for stmt in main.callable_region.stmts()
    )
