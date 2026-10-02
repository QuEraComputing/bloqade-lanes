from __future__ import annotations

from kirin import ir, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade import types as bloqade_types
from bloqade.lanes.dialects.arch import LocationAddressType
from bloqade.lanes.dialects.move import ConsumesState, EmitsState, StatefulStatement
from bloqade.lanes.types import StateType

from ._dialect import dialect
from .frame import Frame

N = types.TypeVar("N")
Len = types.TypeVar("Len")
QubitList = ilist.IListType[bloqade_types.QubitType, types.Any]


@statement(dialect=dialect)
class CZ(StatefulStatement):
    controls: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, N])
    targets: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, N])


@statement(dialect=dialect)
class R(StatefulStatement):
    axis_angle: ir.SSAValue = info.argument(types.Float)
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(QubitList)


@statement(dialect=dialect)
class Rz(StatefulStatement):
    rotation_angle: ir.SSAValue = info.argument(types.Float)
    qubits: ir.SSAValue = info.argument(QubitList)


@statement(dialect=dialect)
class MoveTo(StatefulStatement):
    """Leaves quantum information unchanged; afterwards the atoms carrying
    ``qubits`` are at ``locations``."""

    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, Len])
    locations: ir.SSAValue = info.argument(ilist.IListType[LocationAddressType, Len])
    multi_move_warning: bool = info.attribute(default=True)


@statement(dialect=dialect)
class Permute(StatefulStatement):
    """Afterwards ``qubits[i]`` holds what ``qubits[perm[i]]`` held.

    ``insert_moves`` constrains how synthesis realizes it: ``False`` by
    relabeling the reference-to-atom binding with no moves, ``True`` with moves
    that restore the previous binding.
    """

    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, Len])
    perm: ir.SSAValue = info.argument(ilist.IListType[types.Int, Len])
    insert_moves: bool = info.attribute(default=False)


@statement(dialect=dialect)
class Measure(StatefulStatement):
    """Non-terminal measurement: the state continues after it."""

    qubits: ir.SSAValue = info.argument(ilist.IListType[bloqade_types.QubitType, Len])
    measurements: ir.ResultValue = info.result(
        ilist.IListType[bloqade_types.MeasurementResultType, Len]
    )


@statement(dialect=dialect)
class Enter(ir.Statement):
    """Open a subroutine's chain; ``move.load`` plus the frame's precondition.

    ``frame=None`` is a hole that later synthesis fills.
    """

    traits = frozenset({EmitsState(True)})
    frame: Frame | None = info.attribute(default=None)
    result: ir.ResultValue = info.result(StateType)


@statement(dialect=dialect)
class Exit(ir.Statement):
    """Close a subroutine's chain; ``move.store`` plus the frame's postcondition."""

    traits = frozenset({ConsumesState(False)})
    current_state: ir.SSAValue = info.argument(StateType)


@statement(dialect=dialect)
class Prepare(StatefulStatement):
    """Establish ``callee``'s frame precondition for ``inputs``."""

    callee: ir.Method = info.attribute()
    inputs: tuple[ir.SSAValue, ...] = info.argument()


@statement(dialect=dialect)
class Invoke(StatefulStatement):
    """Call a subroutine on the caller's chain.

    ``value`` always exists (a statement's result count is fixed by its
    declaration) and has the callee's return type, ``NoneType`` included.
    """

    callee: ir.Method = info.attribute()
    inputs: tuple[ir.SSAValue, ...] = info.argument()
    value: ir.ResultValue = info.result()
