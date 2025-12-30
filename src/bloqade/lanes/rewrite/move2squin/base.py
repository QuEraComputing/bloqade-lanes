from dataclasses import dataclass, field

from kirin import ir
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.arch import ArchSpec


@dataclass
class InsertQubits(rewrite_abc.RewriteRule):
    physical_ssa_values: list[ir.SSAValue] = field(default_factory=list, init=False)

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        for location_addr in node.location_addresses:
            (new_qubit := qubit.stmts.New()).insert_before(node)
            self.physical_ssa_values.append(new_qubit.result)

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class AtomStateRewriter(rewrite_abc.RewriteRule):
    arch_spec: ArchSpec
    physical_ssa_values: tuple[ir.SSAValue, ...]

    def get_qubit_ssa(
        self, atom_state: atom.AtomState, location: LocationAddress
    ) -> ir.SSAValue | None:
        qubit_index = atom_state.data.get_qubit(location)

        if qubit_index is None:
            return None

        return self.physical_ssa_values[qubit_index]

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
