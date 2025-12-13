from dataclasses import dataclass, field

from bloqade.squin.gate import stmts as gate_stmts
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LocationAddress

from . import utils


@dataclass
class InsertQubits(rewrite_abc.RewriteRule):
    physical_ssa_values: list[ir.SSAValue] = field(default_factory=list, init=False)
    initial_location_map: dict[LocationAddress, ir.SSAValue] = field(
        default_factory=dict, init=False
    )

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        for location_addr in self.initial_location_map:
            (new_qubit := qubit.stmts.New()).insert_before(node)
            self.physical_ssa_values.append(new_qubit.result)
            self.initial_location_map[location_addr] = new_qubit.result

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class RewriteMoveDialect(rewrite_abc.RewriteRule):
    physical_ssa_values: tuple[ir.SSAValue, ...]
    atom_state_map: dict[ir.Statement, atom.AtomState]

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not (
            isinstance(
                node,
                (
                    move.LocalRz,
                    move.GlobalRz,
                    move.LocalR,
                    move.GlobalR,
                    move.GetMeasurementResult,
                ),
            )
            and isinstance(atom_state := self.atom_state_map.get(node), atom.AtomState)
        ):
            return rewrite_abc.RewriteResult()

        rewriter = getattr(self, f"rewrite_{type(node).__name__}")
        return rewriter(atom_state, node)

    def get_qubit_ssa(self, qubit_index: int) -> ir.SSAValue | None:
        if 0 <= qubit_index < len(self.physical_ssa_values):
            return self.physical_ssa_values[qubit_index]
        return None

    def get_qubit_ssa_from_locations(
        self,
        atom_state: atom.AtomState,
        location_addresses: tuple[LocationAddress, ...],
    ) -> tuple[ir.SSAValue | None, ...]:
        qubit_ssa = tuple(
            map(
                self.get_qubit_ssa,
                filter(None, map(atom_state.get_qubit, location_addresses)),
            )
        )

        return qubit_ssa

    def rewrite_LocalRz(
        self, atom_state: atom.AtomState, node: move.LocalRz
    ) -> rewrite_abc.RewriteResult:

        qubit_ssa = self.get_qubit_ssa_from_locations(
            atom_state, node.location_addresses
        )

        if not utils.no_none_elements_tuple(qubit_ssa):
            return rewrite_abc.RewriteResult()

        (zero := py.Constant(0.0)).insert_before(node)
        (reg := ilist.New(qubit_ssa)).insert_before(node)
        node.replace_by(
            gate_stmts.U3(zero.result, node.rotation_angle, zero.result, reg.result)
        )
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_GlobalRz(
        self, atom_state: atom.AtomState, node: move.GlobalRz
    ) -> rewrite_abc.RewriteResult:
        (zero := py.Constant(0.0)).insert_before(node)
        (reg := ilist.New(self.physical_ssa_values)).insert_before(node)
        node.replace_by(
            gate_stmts.U3(zero.result, node.rotation_angle, zero.result, reg.result)
        )
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_LocalR(
        self, atom_state: atom.AtomState, node: move.LocalR
    ) -> rewrite_abc.RewriteResult:
        # R -> U3: https://algassert.com/quirk#circuit={%22cols%22:[[%22QFT3%22],[%22inputA3%22,1,1,%22+=A3%22],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22-pi%20t%22}],[],[1,1,1,1,1,{%22id%22:%22Rxft%22,%22arg%22:%22-pi%20t^3%22}],[],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22pi%20t%22}],[1,1,1,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22-pi%20t%20+%20pi/2%22}],[],[],[1,1,1,1,1,{%22id%22:%22Ryft%22,%22arg%22:%22pi%20t^3%22}],[],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22pi%20t%20-%20pi/2%22}]]}

        (quarter_turn := py.Constant(0.25)).insert_before(node)
        (phi := py.Sub(quarter_turn.result, node.axis_angle)).insert_before(node)
        (lam := py.Sub(node.axis_angle, quarter_turn.result)).insert_before(node)

        qubit_ssa = self.get_qubit_ssa_from_locations(
            atom_state, node.location_addresses
        )

        if not utils.no_none_elements_tuple(qubit_ssa):
            return rewrite_abc.RewriteResult()

        (reg := ilist.New(qubit_ssa)).insert_before(node)
        node.replace_by(
            gate_stmts.U3(phi.result, node.rotation_angle, lam.result, reg.result)
        )
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_GlobalR(
        self, atom_state: atom.AtomState, node: move.GlobalR
    ) -> rewrite_abc.RewriteResult:
        (quarter_turn := py.Constant(0.25)).insert_before(node)
        (phi := py.Sub(quarter_turn.result, node.axis_angle)).insert_before(node)
        (lam := py.Sub(node.axis_angle, quarter_turn.result)).insert_before(node)
        (reg := ilist.New(self.physical_ssa_values)).insert_before(node)
        node.replace_by(
            gate_stmts.U3(phi.result, node.rotation_angle, lam.result, reg.result)
        )
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_GetMeasurementResult(
        self, atom_state: atom.AtomState, node: move.GetMeasurementResult
    ) -> rewrite_abc.RewriteResult:
        (qubit_ssa,) = self.get_qubit_ssa_from_locations(
            atom_state, (node.location_address,)
        )
        if qubit_ssa is None:
            return rewrite_abc.RewriteResult()

        node.replace_by(func.Invoke((qubit_ssa,), callee=qubit.measure))

        return rewrite_abc.RewriteResult(has_done_something=True)
