from typing import Literal, TypeVar

from bloqade.native.dialects.gate import stmts as gate
from bloqade.types import MeasurementResultType, Qubit, QubitType
from kirin import ir, rewrite, types
from kirin.dialects import func, ilist

from bloqade import qubit, squin
from bloqade.gemini.common.dialects.arrange import stmts as arrange
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.rewrite.native2qmove import RewriteNativeToQMove

QUBITS = ilist.IListType[QubitType, types.Literal(2)]
T = TypeVar("T", bound=ir.Statement)


def _lower(block: ir.Block, subroutines=frozenset()) -> list[ir.Statement]:
    rule = rewrite.Walk(RewriteNativeToQMove(subroutines))
    assert rule.rewrite(block).has_done_something
    return list(block.stmts)


def _assert_wrapped(stmts: list[ir.Statement], kind: type[T]) -> T:
    load, stateful, store = stmts
    assert isinstance(load, move.Load) and isinstance(store, move.Store)
    assert isinstance(stateful, kind)
    assert stateful.args[0] is load.result
    assert store.current_state is stateful.results[0]
    return stateful


def test_gates_are_wrapped_in_load_and_store():
    qs = ir.TestValue(QUBITS)
    angle = ir.TestValue(types.Float)
    cases: list[tuple[ir.Statement, type[ir.Statement]]] = [
        (gate.CZ(qs, qs), qmove.CZ),
        (gate.R(angle, angle, qs), qmove.R),
        (gate.Rz(angle, qs), qmove.Rz),
    ]
    for native, kind in cases:
        operands = tuple(native.args)
        stateful = _assert_wrapped(_lower(ir.Block([native])), kind)
        assert tuple(stateful.args[1:]) == operands


def test_move_to_and_permute_keep_their_attributes():
    qs = ir.TestValue(QUBITS)
    move_to = arrange.MoveTo(qs, ir.TestValue(types.Any), multi_move_warning=False)
    lowered = _assert_wrapped(_lower(ir.Block([move_to])), qmove.MoveTo)
    assert lowered.multi_move_warning is False

    permute = arrange.Permute(qs, ir.TestValue(types.Any), insert_moves=True)
    relabel = _assert_wrapped(_lower(ir.Block([permute])), qmove.Permute)
    assert relabel.insert_moves is True


def test_measure_result_is_forwarded():
    measure = qubit.stmts.Measure(ir.TestValue(QUBITS))
    measure.result.type = ilist.IListType[MeasurementResultType, types.Literal(2)]
    user = ilist.New(values=(measure.result,))
    lowered = _assert_wrapped(_lower(ir.Block([measure, user]))[:3], qmove.Measure)
    assert user.values[0] is lowered.measurements
    assert lowered.measurements.type == measure.result.type


def test_only_subroutine_invokes_are_lowered():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    @squin.kernel
    def helper(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    qs = ir.TestValue(QUBITS)
    call_sub = func.Invoke((qs,), callee=sub)
    call_helper = func.Invoke((qs,), callee=helper)
    stmts = _lower(ir.Block([call_sub, call_helper]), frozenset({sub}))
    lowered = _assert_wrapped(stmts[:3], qmove.Invoke)
    assert lowered.callee is sub and tuple(lowered.inputs) == (qs,)
    assert stmts[3] is call_helper
