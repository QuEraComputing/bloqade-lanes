from typing import Literal

import pytest
from bloqade.types import Qubit, QubitType
from kirin import ir, types as kirin_types
from kirin.dialects import func, ilist, py, scf
from kirin.validation import ValidationSuite

from bloqade import squin
from bloqade.gemini import physical
from bloqade.gemini.common.dialects import qubit as gemini_qubit
from bloqade.gemini.logical.dialects.extensions import stmts as extensions
from bloqade.gemini.logical.dialects.operations import stmts as logical
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.transform.qmove_frontend import lower_to_native
from bloqade.lanes.validation.qmove_input import get_input_validation


def _messages(method: ir.Method, subroutine: bool = False) -> list[str]:
    result = ValidationSuite([get_input_validation(subroutine)]).validate(method)
    return [str(err.args[0]) for errs in result.errors.values() for err in errs]


def _native(kernel, subroutines=()):
    return lower_to_native(kernel, subroutines, get_arch_spec())


def _method(*blocks: ir.Block) -> ir.Method:
    @squin.kernel
    def stub():
        return None

    out = stub.similar()
    out.code = func.Function(
        sym_name="stub",
        signature=func.Signature(inputs=(), output=kirin_types.NoneType),
        body=ir.Region(list(blocks)),
    )
    return out


def test_supported_kernel_is_valid():
    @squin.kernel
    def k():
        qs = squin.qalloc(2)
        for q in qs:
            squin.h(q)
        if squin.is_one(squin.measure(qs[0])):
            squin.x(qs[1])
        return squin.broadcast.measure(qs)

    assert _messages(_native(k).entry) == []


def test_reset_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        squin.reset(qs[0])

    assert _messages(_native(k).entry) == ["qubit.reset is not supported"]


def test_noise_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        squin.depolarize(0.1, qs[0])

    (message,) = _messages(_native(k).entry)
    assert "is not supported by the qmove lowering" in message


def test_function_value_applying_gates_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(2)

        def f(q: Qubit):
            squin.h(q)

        ilist.for_each(f, qs)

    assert _messages(_native(k).entry) == [
        "for_each applies gates through a function value; use a for loop"
    ]


def test_allocation_only_function_value_is_fine():
    @physical.kernel(verify=False)
    def k():
        def at(i: int):
            return gemini_qubit.new_at(0, i, 0)

        qs = ilist.map(at, ilist.range(2))
        squin.h(qs[0])

    assert _messages(_native(k).entry) == []


def test_allocation_in_a_subroutine_is_rejected():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[1]]):
        extra = squin.qalloc(1)
        squin.cz(qs[0], extra[0])

    @squin.kernel
    def k():
        qs = squin.qalloc(1)
        sub(qs)

    clone = _native(k, [sub]).subroutines[sub]
    assert _messages(clone, subroutine=True) == [
        "qubits may only be allocated in the entry kernel"
    ]


def _logical_statements() -> list[ir.Statement]:
    qubits = ir.TestValue(ilist.IListType[QubitType, kirin_types.Literal(7)])
    angle = ir.TestValue(kirin_types.Float)
    return [
        logical.TerminalLogicalMeasurement(qubits),
        logical.Initialize(angle, angle, angle, qubits),
        extensions.StarRz(angle, qubits),
    ]


@pytest.mark.parametrize("index", range(3))
def test_logical_statements_are_rejected(index: int):
    stmt = _logical_statements()[index]
    (message,) = _messages(_method(ir.Block([stmt, func.Return()])))
    assert "logical-pipeline statement" in message


def test_call_through_a_function_value_is_rejected():
    call = func.Call(ir.TestValue(kirin_types.Any), (), kwargs=())
    assert _messages(_method(ir.Block([call, func.Return()]))) == [
        "calls through a function value are not supported"
    ]


def test_early_return_and_multiple_blocks_are_rejected():
    cond = ir.TestValue(kirin_types.Bool)
    then_block = ir.Block()
    then_block.args.append_from(kirin_types.Bool)
    then_block.stmts.append(func.Return(py.Constant(1).result))
    else_block = ir.Block()
    else_block.args.append_from(kirin_types.Bool)
    else_block.stmts.append(scf.Yield())
    branch = scf.IfElse(cond, then_block, else_block)
    messages = _messages(
        _method(ir.Block([branch, func.Return()]), ir.Block([func.Return()]))
    )
    assert "early return inside scf is not supported" in messages
    assert any("region has 2 blocks" in m for m in messages)


def test_bottom_typed_quantum_result_is_rejected():
    @squin.kernel
    def k():
        qs = squin.qalloc(3)
        # measure takes one Qubit, so its result is Bottom.
        m = squin.measure(qs)  # type: ignore[arg-type]
        squin.h(qs[0])
        return m[0]  # type: ignore[index]

    (message,) = _messages(_native(k).entry)
    assert message == (
        "measure result has no valid type (Bottom); check its argument types"
    )
