"""Tests for QubitRegisterValidation."""

from bloqade.native.dialects import gate as native_gate
from kirin import ir, types as kirin_types
from kirin.dialects import func, ilist, py
from kirin.validation import ValidationSuite

from bloqade import qubit as squin_qubit, squin, types as bloqade_types
from bloqade.lanes.validation.qubit_register import QubitRegisterValidation

QUBIT = bloqade_types.QubitType


def _method(block: ir.Block) -> ir.Method:
    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(block),
    )
    return out


def _error_count(block: ir.Block) -> int:
    result = ValidationSuite([QubitRegisterValidation]).validate(_method(block))
    return 0 if result.is_valid else result.error_count()


def test_a_register_of_distinct_qubits_is_valid():
    q0 = squin_qubit.stmts.New()
    q1 = squin_qubit.stmts.New()
    reg = ilist.New(values=(q0.result, q1.result), elem_type=QUBIT)
    angle = py.Constant(0.25)
    block = ir.Block(
        [
            q0,
            q1,
            reg,
            angle,
            native_gate.stmts.Rz(angle.result, reg.result),
            func.Return(),
        ]
    )

    assert _error_count(block) == 0


def test_a_repeated_qubit_is_rejected():
    """No per-qubit state is well defined for it: one atom, two pending phases."""
    q = squin_qubit.stmts.New()
    doubled = ilist.New(values=(q.result, q.result), elem_type=QUBIT)
    angle = py.Constant(0.25)
    block = ir.Block(
        [
            q,
            doubled,
            angle,
            native_gate.stmts.Rz(angle.result, doubled.result),
            func.Return(),
        ]
    )

    assert _error_count(block) == 1


def test_a_register_not_from_ilist_new_is_rejected():
    """Then the addressed qubits cannot be read off the IR at all."""
    opaque = ir.TestValue(type=ilist.IListType[QUBIT, kirin_types.Any])
    angle = py.Constant(0.25)
    block = ir.Block([angle, native_gate.stmts.Rz(angle.result, opaque), func.Return()])

    assert _error_count(block) == 1


def test_statements_without_a_qubits_operand_are_ignored():
    block = ir.Block([py.Constant(1), func.Return()])

    assert _error_count(block) == 0
