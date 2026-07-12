from bloqade.types import QubitType
from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import ilist

from bloqade.lanes.dialects.arch import LocationAddressType

from ._dialect import dialect

Len = types.TypeVar("Len")


@statement(dialect=dialect)
class MoveTo(ir.Statement):
    """User-facing move_to directive: move qubits to specified LocationAddress destinations.

    Lowered from a Python call by FromPythonCall. RewritePlaceOperations rewrites this
    to a StaticPlacement(place.MoveTo) after const-folding the locations ilist.
    """

    name = "move_to"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(type=ilist.IListType[QubitType, Len])
    locations: ir.SSAValue = info.argument(
        type=ilist.IListType[LocationAddressType, Len]
    )
    multi_move_warning: bool = info.attribute(default=True)


@statement(dialect=dialect)
class Permute(ir.Statement):
    """User-facing permute directive: move qubits into a permutation of their
    own current locations. qubits[i] moves to the current location of
    qubits[perm[i]].

    Lowered from a Python call by FromPythonCall. RewritePlaceOperations rewrites
    this to a StaticPlacement(place.Permute) after const-folding the perm ilist.

    ``insert_moves`` (compile-time constant) selects how the permutation is
    realized: ``False`` (default) relabels the qubit references only — no atoms
    move, the quantum information is permuted for free; ``True`` also commits the
    physical moves that realize the permutation in place. See ``place.Permute`` /
    ``MoveToPlacementStrategyABC.permute_placements`` for details.
    """

    name = "permute"
    traits = frozenset({lowering.FromPythonCall()})
    qubits: ir.SSAValue = info.argument(type=ilist.IListType[QubitType, Len])
    perm: ir.SSAValue = info.argument(type=ilist.IListType[types.Int, Len])
    insert_moves: bool = info.attribute(default=False)
