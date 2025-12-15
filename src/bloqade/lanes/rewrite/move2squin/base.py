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
        qubit_index = atom_state.get_qubit(location)

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
    atom_state_map: dict[ir.Statement, atom.AtomStateType]

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not (
            isinstance(
                node,
                (
                    move.Fill,
                    move.LocalRz,
                    move.GlobalRz,
                    move.LocalR,
                    move.GlobalR,
                    move.GetMeasurementResult,
                    move.PhysicalInitialize,
                    move.Move,
                    move.EndMeasure,
                    move.CZ,
                ),
            )
            and isinstance(self.atom_state_map.get(node), atom.AtomState)
        ):
            return rewrite_abc.RewriteResult()

        if any(len(result.uses) > 0 for result in node.results):
            return rewrite_abc.RewriteResult()

        node.delete()

        return rewrite_abc.RewriteResult(has_done_something=True)
