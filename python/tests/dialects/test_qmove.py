from typing import Literal

from bloqade.types import MeasurementResultType, Qubit, QubitType
from kirin import ir, types
from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects import move, qmove
from bloqade.lanes.types import StateType

QUBITS2 = ilist.IListType[QubitType, types.Literal(2)]


def _state() -> ir.TestValue:
    return ir.TestValue(StateType)


def test_gate_statements_thread_the_state():
    s, qs = _state(), ir.TestValue(QUBITS2)
    angle = ir.TestValue(types.Float)
    for stmt in (
        qmove.CZ(s, qs, qs),
        qmove.R(s, angle, angle, qs),
        qmove.Rz(s, angle, qs),
    ):
        assert stmt.current_state is s
        assert stmt.result.type is StateType
        assert stmt.get_trait(move.ConsumesState) == move.ConsumesState(False)
        assert stmt.get_trait(move.EmitsState) == move.EmitsState(False)


def test_measure_returns_state_then_measurements():
    s, qs = _state(), ir.TestValue(QUBITS2)
    stmt = qmove.Measure(s, qs)
    assert stmt.results[0].type is StateType
    assert stmt.measurements.type.is_subseteq(
        ilist.IListType[MeasurementResultType, types.Any]
    )


def test_move_to_and_permute_keep_their_attributes():
    s, qs = _state(), ir.TestValue(QUBITS2)
    locs = ir.TestValue(types.Any)
    perm = ir.TestValue(ilist.IListType[types.Int, types.Literal(2)])
    assert (
        qmove.MoveTo(s, qs, locs, multi_move_warning=False).multi_move_warning is False
    )
    assert qmove.Permute(s, qs, perm, insert_moves=True).insert_moves is True


def test_enter_holds_an_optional_hashable_frame():
    frame = qmove.Frame(
        qmove.FrameShape(((0, 2),), scratch_slots=1),
        (LocationAddress(0, 0), LocationAddress(1, 0), LocationAddress(2, 0)),
        qmove.Effects(cz_zones=frozenset({ZoneAddress(0)})),
    )
    assert frame.shape.total_slots == 3
    assert qmove.Enter(frame=frame).frame == frame
    assert hash(qmove.Enter(frame=frame).attributes["frame"]) is not None
    assert qmove.Enter().frame is None
    assert qmove.Enter().get_trait(move.EmitsState) == move.EmitsState(True)
    assert qmove.Exit(_state()).get_trait(move.ConsumesState) == move.ConsumesState(
        False
    )


def test_effects_subset():
    z0, z1 = ZoneAddress(0), ZoneAddress(1)
    inner = qmove.Effects(cz_zones=frozenset({z0}))
    outer = qmove.Effects(cz_zones=frozenset({z0, z1}), global_pulses=True)
    assert inner.is_subset_of(outer)
    assert not outer.is_subset_of(inner)


def test_invoke_and_prepare_take_a_callee_and_inputs():
    @squin.kernel
    def sub(qs: ilist.IList[Qubit, Literal[2]]):
        squin.h(qs[0])

    s, qs = _state(), ir.TestValue(QUBITS2)
    invoke = qmove.Invoke(s, (qs,), callee=sub)
    assert invoke.callee is sub and tuple(invoke.inputs) == (qs,)
    assert len(invoke.results) == 2
    prepare = qmove.Prepare(s, (qs,), callee=sub)
    assert prepare.result.type is StateType


def test_dialect_contains_every_statement():
    assert set(qmove.dialect.stmts) == {
        qmove.CZ,
        qmove.R,
        qmove.Rz,
        qmove.MoveTo,
        qmove.Permute,
        qmove.Measure,
        qmove.Enter,
        qmove.Exit,
        qmove.Prepare,
        qmove.Invoke,
    }
