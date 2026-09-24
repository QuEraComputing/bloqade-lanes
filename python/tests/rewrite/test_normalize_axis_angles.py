"""Tests for the NormalizeGateAxisAngles rewrite rule.

Like ``test_eliminate_rz.py``, each case builds the input and expected blocks
explicitly and compares them whole, so a stray statement fails the test.
"""

import pytest
from bloqade.native.dialects import gate as native_gate
from bloqade.test_utils import assert_nodes
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.lanes.rewrite.normalize_axis_angles import NormalizeGateAxisAngles

QUBIT = bloqade_types.QubitType


def _axis_above_one_turn_is_reduced():
    """1.25 turns and 0.25 turns are the same axis; only the latter can share."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    axis = py.Constant(1.25)
    turn = py.Constant(0.5)
    test = ir.Block(
        [q, reg, axis, turn, native_gate.stmts.R(axis.result, turn.result, reg.result)]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aaxis = py.Constant(1.25)
    aturn = py.Constant(0.5)
    reduced = py.Constant(0.25)
    expected = ir.Block(
        [
            a,
            areg,
            aaxis,
            aturn,
            reduced,
            native_gate.stmts.R(reduced.result, aturn.result, areg.result),
        ]
    )
    return test, expected


def _negative_axis_wraps_into_range():
    """`EliminateRz` emits `phi - alpha`, so negative axes are the common case."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    axis = py.Constant(-0.25)
    turn = py.Constant(0.5)
    test = ir.Block(
        [q, reg, axis, turn, native_gate.stmts.R(axis.result, turn.result, reg.result)]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aaxis = py.Constant(-0.25)
    aturn = py.Constant(0.5)
    reduced = py.Constant(0.75)
    expected = ir.Block(
        [
            a,
            areg,
            aaxis,
            aturn,
            reduced,
            native_gate.stmts.R(reduced.result, aturn.result, areg.result),
        ]
    )
    return test, expected


def _axis_already_in_range_is_untouched():
    """No churn on the gates that do not need it -- and no fresh constant."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    axis = py.Constant(0.25)
    turn = py.Constant(0.5)
    test = ir.Block(
        [q, reg, axis, turn, native_gate.stmts.R(axis.result, turn.result, reg.result)]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aaxis = py.Constant(0.25)
    aturn = py.Constant(0.5)
    expected = ir.Block(
        [
            a,
            areg,
            aaxis,
            aturn,
            native_gate.stmts.R(aaxis.result, aturn.result, areg.result),
        ]
    )
    return test, expected


def _rotation_angle_is_left_alone():
    """Only phi has period 1. `theta -> theta + 1` gives `-R(phi, theta)`, so
    reducing it is not an identity -- and nothing here ever grows it."""
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    axis = py.Constant(0.25)
    turn = py.Constant(2.5)
    test = ir.Block(
        [q, reg, axis, turn, native_gate.stmts.R(axis.result, turn.result, reg.result)]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    aaxis = py.Constant(0.25)
    aturn = py.Constant(2.5)
    expected = ir.Block(
        [
            a,
            areg,
            aaxis,
            aturn,
            native_gate.stmts.R(aaxis.result, aturn.result, areg.result),
        ]
    )
    return test, expected


def _symbolic_axis_gets_a_runtime_mod():
    """The compiler cannot read this one, so the reduction is deferred.

    `py.Mod` is emitted rather than the axis being skipped: there is nothing
    to fold, so nothing is lost by deferring, and the program still gets a
    normalized axis -- computed by whoever runs it.
    """
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    lhs = py.Constant(1.5)
    rhs = py.Constant(0.25)
    axis = py.Add(lhs.result, rhs.result)
    turn = py.Constant(0.5)
    test = ir.Block(
        [
            q,
            reg,
            lhs,
            rhs,
            axis,
            turn,
            native_gate.stmts.R(axis.result, turn.result, reg.result),
        ]
    )

    a = squin_qubit.stmts.New()
    areg = ilist.New(values=(a.result,), elem_type=QUBIT)
    albs = py.Constant(1.5)
    arhs = py.Constant(0.25)
    aaxis = py.Add(albs.result, arhs.result)
    aturn = py.Constant(0.5)
    one = py.Constant(1.0)
    reduced = py.Mod(aaxis.result, one.result)
    expected = ir.Block(
        [
            a,
            areg,
            albs,
            arhs,
            aaxis,
            aturn,
            one,
            reduced,
            native_gate.stmts.R(reduced.result, aturn.result, areg.result),
        ]
    )
    return test, expected


CASES = {
    "axis_above_one_turn_is_reduced": _axis_above_one_turn_is_reduced,
    "negative_axis_wraps_into_range": _negative_axis_wraps_into_range,
    "axis_already_in_range_is_untouched": _axis_already_in_range_is_untouched,
    "rotation_angle_is_left_alone": _rotation_angle_is_left_alone,
    "symbolic_axis_gets_a_runtime_mod": _symbolic_axis_gets_a_runtime_mod,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_rewrite(case):
    test_block, expected_block = CASES[case]()
    rewrite.Walk(NormalizeGateAxisAngles()).rewrite(test_block)
    assert_nodes(test_block, expected_block)


def test_a_constant_shared_across_roles_is_not_mutated():
    """The reason a fresh constant is inserted rather than the old one edited.

    The real post-unroll IR contains a single ``py.Constant`` feeding both the
    axis angle and the rotation angle of the same statement, and
    ``HoistConstants`` runs just before this point, making that sharing more
    likely rather than less. Editing the constant in place would silently move
    the rotation angle too.
    """
    q = squin_qubit.stmts.New()
    reg = ilist.New(values=(q.result,), elem_type=QUBIT)
    shared = py.Constant(1.25)
    gate = native_gate.stmts.R(shared.result, shared.result, reg.result)
    block = ir.Block([q, reg, shared, gate])

    rewrite.Walk(NormalizeGateAxisAngles()).rewrite(block)

    assert shared.value.unwrap() == 1.25, "the shared constant must not be edited"
    assert gate.rotation_angle is shared.result, "the rotation angle must not move"
    assert gate.axis_angle is not shared.result
    assert gate.axis_angle.owner.value.unwrap() == 0.25  # type: ignore[union-attr]


def test_a_constant_axis_is_idempotent():
    """A reduced constant is already in range, so a second pass finds nothing."""
    test_block, _ = _axis_above_one_turn_is_reduced()
    rewrite.Walk(NormalizeGateAxisAngles()).rewrite(test_block)
    result = rewrite.Walk(NormalizeGateAxisAngles()).rewrite(test_block)
    assert not result.has_done_something


def test_a_symbolic_axis_is_not_idempotent():
    """Pins the contract stated on the class, rather than guarding against it.

    `constant_float` cannot see through a `py.Mod`, so a second pass wraps the axis
    again. The rule is run once by `NativeToPlaceBase.emit`; this test exists
    so that anyone reaching for `Fixpoint` -- which would nest one `py.Mod`
    per iteration until `max_iter` -- finds the property written down and
    failing loudly rather than discovering it in an emitted program.
    """
    test_block, _ = _symbolic_axis_gets_a_runtime_mod()

    rewrite.Walk(NormalizeGateAxisAngles()).rewrite(test_block)
    first = sum(1 for stmt in test_block.stmts if isinstance(stmt, py.Mod))

    rewrite.Walk(NormalizeGateAxisAngles()).rewrite(test_block)
    second = sum(1 for stmt in test_block.stmts if isinstance(stmt, py.Mod))

    assert (first, second) == (
        1,
        2,
    ), "documented behaviour changed; update the docstring"
