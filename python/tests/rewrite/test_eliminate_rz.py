"""Tests for the EliminateRz rewrite rule.

Each case builds the input block and the expected output block explicitly and
compares them whole with ``assert_nodes``, so a stray statement the rule leaves
behind fails the test rather than slipping past a spot-check.
"""

import pytest
from bloqade.native.dialects import gate as native_gate
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite, types as kirin_types
from kirin.dialects import func, ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.gemini.logical.dialects.operations import stmts as operations
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz, EliminateRzError

QUBIT = bloqade_types.QubitType


def _rz_is_deleted():
    """A lone Rz is absorbed into the frame and removed. Its angle constant is
    left behind as dead code -- DCE later in the pipeline sweeps it, and this
    rule does not do reference counting."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    test = ir.Block([q, reg, quarter, native_gate.stmts.Rz(quarter.result, reg.result)])

    q2 = squin_qubit.stmts.New()
    reg2 = ilist.New(values=(q2.result,), elem_type=QUBIT)
    expected = ir.Block([q2, reg2, py.Constant(0.25)])
    return test, expected


def _r_axis_shifts_by_the_frame():
    """R(0) after Rz(0.25) becomes R(0 - 0.25). The rule emits a fresh register
    and gate before the original and deletes it, so the new pair lands where the
    old R was."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    q2 = squin_qubit.stmts.New()
    reg2 = ilist.New(values=(q2.result,), elem_type=QUBIT)
    quarter2 = py.Constant(0.25)
    zero2 = py.Constant(0.0)
    shifted = py.Constant(-0.25)
    expected = ir.Block(
        [
            q2,
            reg2,
            quarter2,
            zero2,
            shifted,
            native_gate.stmts.R(shifted.result, quarter2.result, reg2.result),
        ]
    )
    return test, expected


def _frames_accumulate():
    """Two Rz on one qubit sum before shifting: 0.0 - (0.25 + 0.5) = -0.75."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    half = py.Constant(0.5)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            half,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            native_gate.stmts.Rz(half.result, reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    q2 = squin_qubit.stmts.New()
    reg2 = ilist.New(values=(q2.result,), elem_type=QUBIT)
    quarter2 = py.Constant(0.25)
    half2 = py.Constant(0.5)
    zero2 = py.Constant(0.0)
    summed = py.Constant(0.75)
    shifted = py.Constant(-0.75)
    expected = ir.Block(
        [
            q2,
            reg2,
            quarter2,
            half2,
            zero2,
            summed,
            shifted,
            native_gate.stmts.R(shifted.result, quarter2.result, reg2.result),
        ]
    )
    return test, expected


def _untouched_qubit_is_left_alone():
    """An R on a qubit no Rz ever reached keeps its original statement -- the
    rule must not rebuild it."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    reg0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    reg1 = ilist.New(values=(q1.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            reg0,
            reg1,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg0.result),
            native_gate.stmts.R(zero.result, quarter.result, reg1.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    areg0 = ilist.New(values=(a0.result,), elem_type=QUBIT)
    areg1 = ilist.New(values=(a1.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    expected = ir.Block(
        [
            a0,
            a1,
            areg0,
            areg1,
            aquarter,
            azero,
            native_gate.stmts.R(azero.result, aquarter.result, areg1.result),
        ]
    )
    return test, expected


def _r_splits_when_frames_differ():
    """One pulse cannot carry two axis angles. Only q0 saw an Rz, so the shared
    R splits: q0 gets the shifted axis, q1 keeps the original."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    only_q0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    both = ilist.New(values=(q0.result, q1.result), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            only_q0,
            both,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, only_q0.result),
            native_gate.stmts.R(zero.result, quarter.result, both.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    a_only = ilist.New(values=(a0.result,), elem_type=QUBIT)
    a_both = ilist.New(values=(a0.result, a1.result), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    shifted = py.Constant(-0.25)
    shifted_reg = ilist.New(values=(a0.result,), elem_type=QUBIT)
    plain_reg = ilist.New(values=(a1.result,), elem_type=QUBIT)
    expected = ir.Block(
        [
            a0,
            a1,
            a_only,
            a_both,
            aquarter,
            azero,
            shifted,
            shifted_reg,
            native_gate.stmts.R(shifted.result, aquarter.result, shifted_reg.result),
            plain_reg,
            native_gate.stmts.R(azero.result, aquarter.result, plain_reg.result),
        ]
    )
    return test, expected


def _equal_frames_produce_equal_axes():
    """Two R statements shifted by the same frame get the same axis *value*,
    each as its own constant. Deduplicating them is not this rule's job --
    py.Constant is Pure, so the CommonSubexpressionElimination already in
    NativeToPlaceBase.emit merges them, and the fusion this protects is asserted
    end-to-end by test_equal_frames_still_fuse_after_lowering_to_place."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    both = ilist.New(values=(q0.result, q1.result), elem_type=QUBIT)
    reg0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    reg1 = ilist.New(values=(q1.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q0,
            q1,
            both,
            reg0,
            reg1,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, both.result),
            native_gate.stmts.R(zero.result, quarter.result, reg0.result),
            native_gate.stmts.R(zero.result, quarter.result, reg1.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    a_both = ilist.New(values=(a0.result, a1.result), elem_type=QUBIT)
    areg0 = ilist.New(values=(a0.result,), elem_type=QUBIT)
    areg1 = ilist.New(values=(a1.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    shifted0 = py.Constant(-0.25)
    shifted1 = py.Constant(-0.25)
    expected = ir.Block(
        [
            a0,
            a1,
            a_both,
            areg0,
            areg1,
            aquarter,
            azero,
            shifted0,
            native_gate.stmts.R(shifted0.result, aquarter.result, areg0.result),
            shifted1,
            native_gate.stmts.R(shifted1.result, aquarter.result, areg1.result),
        ]
    )
    return test, expected


def _cz_passes_through():
    """CZ is diagonal, so the frame commutes past it and it is left alone."""
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    reg0 = ilist.New(values=(q0.result,), elem_type=QUBIT)
    reg1 = ilist.New(values=(q1.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    test = ir.Block(
        [
            q0,
            q1,
            reg0,
            reg1,
            quarter,
            native_gate.stmts.Rz(quarter.result, reg0.result),
            native_gate.stmts.CZ(reg0.result, reg1.result),
        ]
    )

    a0 = squin_qubit.stmts.New()
    a1 = squin_qubit.stmts.New()
    areg0 = ilist.New(values=(a0.result,), elem_type=QUBIT)
    areg1 = ilist.New(values=(a1.result,), elem_type=QUBIT)
    expected = ir.Block(
        [
            a0,
            a1,
            areg0,
            areg1,
            py.Constant(0.25),
            native_gate.stmts.CZ(areg0.result, areg1.result),
        ]
    )
    return test, expected


def _measurement_discards_the_frame():
    """The residual is diagonal and the readout is in the Z basis, so the
    measurement drops it and a later R on that qubit is unshifted."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    test = ir.Block(
        [
            q,
            reg,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            operations.TerminalLogicalMeasurement(reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aquarter = py.Constant(0.25)
    azero = py.Constant(0.0)
    expected = ir.Block(
        [
            a,
            areg,
            aquarter,
            azero,
            operations.TerminalLogicalMeasurement(areg.result),
            native_gate.stmts.R(azero.result, aquarter.result, areg.result),
        ]
    )
    return test, expected


def _non_constant_angle_stays_symbolic():
    """A parametrized angle has no literal to fold, so the shift is emitted as
    py.Sub over the SSA values -- this is why the frame holds SSA values rather
    than floats."""
    theta = ir.TestValue(type=kirin_types.Float)
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    zero = py.Constant(0.0)
    quarter = py.Constant(0.25)
    test = ir.Block(
        [
            q,
            reg,
            zero,
            quarter,
            native_gate.stmts.Rz(theta, reg.result),
            native_gate.stmts.R(zero.result, quarter.result, reg.result),
        ]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    azero = py.Constant(0.0)
    aquarter = py.Constant(0.25)
    sub = py.Sub(azero.result, theta)
    expected = ir.Block(
        [
            a,
            areg,
            azero,
            aquarter,
            sub,
            native_gate.stmts.R(sub.result, aquarter.result, areg.result),
        ]
    )
    return test, expected


CASES = {
    "rz_is_deleted": _rz_is_deleted,
    "r_axis_shifts_by_the_frame": _r_axis_shifts_by_the_frame,
    "frames_accumulate": _frames_accumulate,
    "untouched_qubit_is_left_alone": _untouched_qubit_is_left_alone,
    "r_splits_when_frames_differ": _r_splits_when_frames_differ,
    "equal_frames_produce_equal_axes": _equal_frames_produce_equal_axes,
    "cz_passes_through": _cz_passes_through,
    "measurement_discards_the_frame": _measurement_discards_the_frame,
    "non_constant_angle_stays_symbolic": _non_constant_angle_stays_symbolic,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_rewrite(case):
    test_block, expected_block = CASES[case]()
    rewrite.Walk(EliminateRz()).rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_rewrite_is_idempotent():
    """A second walk finds no Rz and reports no change."""
    test_block, _ = _r_axis_shifts_by_the_frame()
    rewrite.Walk(EliminateRz()).rewrite(test_block)
    assert not rewrite.Walk(EliminateRz()).rewrite(test_block).has_done_something


# ---------------------------------------------------------------------------
# Preconditions. Dispatch is exhaustive, so anything unregistered is an error
# rather than a silent pass -- that is what keeps an Rz from being left behind.
# ---------------------------------------------------------------------------


def test_unregistered_statement_raises():
    """An unrecognised statement is one whose phase behaviour is unknown, so it
    is an error rather than a guess."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    block = ir.Block([q, reg, ilist.Push(reg.result, q.result)])
    with pytest.raises(EliminateRzError, match="no EliminateRz handler"):
        rewrite.Walk(EliminateRz()).rewrite(block)


def test_multi_block_region_raises():
    """A phase frame has no IR representation that could cross a block."""
    region = ir.Region(ir.Block())
    region.blocks.append(ir.Block())
    with pytest.raises(EliminateRzError, match="single-block"):
        EliminateRz().rewrite_Region(region)


def test_nested_func_function_raises():
    """A nested function's region would reset the frame mid-walk."""
    inner = func.Function(
        sym_name="inner",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(ir.Block([func.ConstantNone(), func.Return()])),
    )
    outer = func.Function(
        sym_name="outer",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(ir.Block([inner])),
    )
    with pytest.raises(EliminateRzError, match="nested func.Function"):
        rewrite.Walk(EliminateRz()).rewrite(outer)


def test_duplicate_qubit_in_one_register_raises():
    q = squin_qubit.stmts.New()
    doubled = ilist.New(values=(q.result, q.result), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    block = ir.Block(
        [q, doubled, quarter, native_gate.stmts.Rz(quarter.result, doubled.result)]
    )
    with pytest.raises(EliminateRzError, match="more than once"):
        rewrite.Walk(EliminateRz()).rewrite(block)


def test_non_ilist_register_raises():
    opaque = ir.TestValue(type=ilist.IListType[QUBIT, kirin_types.Any])
    quarter = py.Constant(0.25)
    block = ir.Block([quarter, native_gate.stmts.Rz(quarter.result, opaque)])
    with pytest.raises(EliminateRzError, match="ilist.New"):
        rewrite.Walk(EliminateRz()).rewrite(block)


def test_initialize_with_a_pending_frame_raises():
    """Initialize sits at the head of a wire; a mid-wire one would drop a phase."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    quarter = py.Constant(0.25)
    zero = py.Constant(0.0)
    block = ir.Block(
        [
            q,
            reg,
            quarter,
            zero,
            native_gate.stmts.Rz(quarter.result, reg.result),
            operations.Initialize(zero.result, zero.result, zero.result, reg.result),
        ]
    )
    with pytest.raises(EliminateRzError, match="Initialize"):
        rewrite.Walk(EliminateRz()).rewrite(block)


def test_star_rz_is_left_alone_and_warns():
    """STAR puts an Rz on a subset of a block's physical qubits, which this
    rule's per-logical-qubit frame does not model. It passes through -- the
    commutation still holds -- but the phase reaches the backend, so the rule
    says so rather than letting it look eliminated."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    theta = py.Constant(0.03)
    star = operations.StarRz(theta.result, reg.result)
    block = ir.Block([q, reg, theta, star])

    with pytest.warns(UserWarning, match="STAR gadget"):
        rewrite.Walk(EliminateRz()).rewrite(block)

    assert star in list(block.stmts)
