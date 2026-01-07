from collections import OrderedDict
from dataclasses import dataclass, field

from kirin import ir, types
from kirin.analysis.forward import ForwardFrame
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.arch import ArchSpec


@dataclass
class InsertQubits(rewrite_abc.RewriteRule):
    atom_state_map: ForwardFrame[atom.MoveExecution]
    physical_ssa_values: dict[int, ir.SSAValue] = field(
        default_factory=OrderedDict, init=False
    )

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        atom_state = self.atom_state_map.get(node.result)
        if not isinstance(atom_state, atom.AtomState):
            return rewrite_abc.RewriteResult()

        qubit_ids = tuple(
            atom_state.data.get_qubit(location_addr)
            for location_addr in node.location_addresses
        )

        if not types.is_tuple_of(qubit_ids, int):
            return rewrite_abc.RewriteResult()

        qubit_ids = sorted(qubit_ids)

        for qubit_id in qubit_ids:
            (new_qubit := qubit.stmts.New()).insert_before(node)
            self.physical_ssa_values[qubit_id] = new_qubit.result

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class AtomStateRewriter(rewrite_abc.RewriteRule):
    arch_spec: ArchSpec
    physical_ssa_values: dict[int, ir.SSAValue]

    def get_qubit_ssa(
        self, atom_state: atom.AtomState, location: LocationAddress
    ) -> ir.SSAValue | None:
        qubit_index = atom_state.data.get_qubit(location)
        if qubit_index is None:
            return None

        return self.physical_ssa_values.get(qubit_index)

    def get_qubit_ssa_from_locations(
        self,
        atom_state: atom.AtomState,
        location_addresses: tuple[LocationAddress, ...],
    ) -> tuple[ir.SSAValue | None, ...]:
        def get_qubit_ssa(location: LocationAddress) -> ir.SSAValue | None:
            return self.get_qubit_ssa(atom_state, location)

        return tuple(map(get_qubit_ssa, location_addresses))


@dataclass
class CleanUpMoveDialect(rewrite_abc.RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if type(node) not in move.dialect.stmts:
            return rewrite_abc.RewriteResult()

        node.delete(safe=False)

        return rewrite_abc.RewriteResult(has_done_something=True)
