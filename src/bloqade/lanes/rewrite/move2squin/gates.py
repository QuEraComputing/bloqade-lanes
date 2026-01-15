from dataclasses import dataclass, field
from functools import singledispatchmethod
from typing import Any

from bloqade.squin.gate import stmts as gate_stmts
from kirin import ir
from kirin.analysis.forward import ForwardFrame
from kirin.dialects import func, ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import LocationAddress, ZoneAddress

from .. import utils
from .base import AtomStateRewriter


@dataclass
class InsertGates(AtomStateRewriter):
    move_exec_analysis: ForwardFrame[atom.MoveExecution]
    initialize_kernel: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None
    ) = None
    measurement_index_map: dict[ZoneAddress, dict[LocationAddress, int]] = field(
        init=False, default_factory=dict
    )
    qubit_ids: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.qubit_ids = tuple(sorted(self.physical_ssa_values.keys()))

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        trait = node.get_trait(move.EmitsState)
        if trait is None:
            return rewrite_abc.RewriteResult()

        atom_state = self.move_exec_analysis.get(trait.get_state_result(node))

        if not isinstance(atom_state, atom.AtomState):
            return rewrite_abc.RewriteResult()

        return self.rewriter(node, atom_state)

    @singledispatchmethod
    def rewriter(
        self, node: ir.Statement, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        return rewrite_abc.RewriteResult()

    @rewriter.register(move.LocalRz)
    def _(
        self, node: move.LocalRz, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:

        qubit_ssa = self.get_qubit_ssa_from_locations(
            atom_state, node.location_addresses
        )

        if not utils.no_none_elements_tuple(qubit_ssa):
            return rewrite_abc.RewriteResult()

        (zero := py.Constant(0.0)).insert_before(node)
        (reg := ilist.New(qubit_ssa)).insert_before(node)
        (
            gate_stmts.U3(zero.result, node.rotation_angle, zero.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.GlobalRz)
    def _(
        self, node: move.GlobalRz, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        qubit_ssas = [self.physical_ssa_values[qubit_id] for qubit_id in self.qubit_ids]
        (zero := py.Constant(0.0)).insert_before(node)
        (reg := ilist.New(qubit_ssas)).insert_before(node)
        (
            gate_stmts.U3(zero.result, node.rotation_angle, zero.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.LocalR)
    def _(
        self, node: move.LocalR, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        # R -> U3: https://algassert.com/quirk#circuit={%22cols%22:[[%22QFT3%22],[%22inputA3%22,1,1,%22+=A3%22],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22-pi%20t%22}],[],[1,1,1,1,1,{%22id%22:%22Rxft%22,%22arg%22:%22-pi%20t^3%22}],[],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22pi%20t%22}],[1,1,1,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22-pi%20t%20+%20pi/2%22}],[],[],[1,1,1,1,1,{%22id%22:%22Ryft%22,%22arg%22:%22pi%20t^3%22}],[],[1,1,1,1,1,{%22id%22:%22Rzft%22,%22arg%22:%22pi%20t%20-%20pi/2%22}]]}

        qubit_ssa = self.get_qubit_ssa_from_locations(
            atom_state, node.location_addresses
        )

        if not utils.no_none_elements_tuple(qubit_ssa):
            return rewrite_abc.RewriteResult()

        (quarter_turn := py.Constant(0.25)).insert_before(node)
        (phi := py.Sub(quarter_turn.result, node.axis_angle)).insert_before(node)
        (lam := py.Sub(node.axis_angle, quarter_turn.result)).insert_before(node)
        (reg := ilist.New(qubit_ssa)).insert_before(node)
        (
            gate_stmts.U3(node.rotation_angle, phi.result, lam.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.GlobalR)
    def _(
        self, node: move.GlobalR, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        qubit_ssas = [self.physical_ssa_values[qubit_id] for qubit_id in self.qubit_ids]

        (quarter_turn := py.Constant(0.25)).insert_before(node)
        (phi := py.Sub(quarter_turn.result, node.axis_angle)).insert_before(node)
        (lam := py.Sub(node.axis_angle, quarter_turn.result)).insert_before(node)
        (reg := ilist.New(qubit_ssas)).insert_before(node)
        (
            gate_stmts.U3(node.rotation_angle, phi.result, lam.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.GetFutureResult)
    def _(
        self, node: move.GetFutureResult, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        zone_address = node.zone_address
        qubit_ssas: list[ir.SSAValue] = []

        for word_id in self.arch_spec.zones[zone_address.zone_id]:
            for site_id, _ in enumerate(self.arch_spec.words[word_id].sites):
                location_address = LocationAddress(word_id, site_id)
                qubit_ssa = self.get_qubit_ssa(atom_state, location_address)
                if qubit_ssa is None:
                    continue
                location_mapping = self.measurement_index_map.setdefault(
                    zone_address, {}
                )
                location_mapping[location_address] = len(qubit_ssas)
                qubit_ssas.append(qubit_ssa)

        if len(qubit_ssas) == 0:
            return rewrite_abc.RewriteResult()

        (reg := ilist.New(tuple(qubit_ssas))).insert_before(node)
        node.replace_by(func.Invoke((reg.result,), callee=qubit.broadcast.measure))

        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.LogicalInitialize)
    def _(
        self, node: move.LogicalInitialize, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        stmts_to_insert: list[ir.Statement] = []
        for theta, phi, lam, location in zip(
            node.thetas, node.phis, node.lams, node.location_addresses
        ):
            qubit_ssa = self.get_qubit_ssa(atom_state, location)

            if qubit_ssa is None:
                return rewrite_abc.RewriteResult()

            stmts_to_insert.append(reg := ilist.New((qubit_ssa,)))
            stmts_to_insert.append(gate_stmts.U3(theta, phi, lam, reg.result))

        for stmt in stmts_to_insert:
            stmt.insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=len(stmts_to_insert) > 0)

    @rewriter.register(move.PhysicalInitialize)
    def _(
        self, node: move.PhysicalInitialize, atom_state: atom.AtomState
    ) -> rewrite_abc.RewriteResult:
        if self.initialize_kernel is None:
            return rewrite_abc.RewriteResult()

        nodes_to_insert: list[ir.Statement] = []
        for theta, phi, lam, locations in zip(
            node.thetas, node.phis, node.lams, node.location_addresses
        ):
            qubit_ssa = self.get_qubit_ssa_from_locations(atom_state, locations)
            if not utils.no_none_elements_tuple(qubit_ssa):
                return rewrite_abc.RewriteResult()

            nodes_to_insert.append(reg_stmt := ilist.New(qubit_ssa))
            inputs = (theta, phi, lam, reg_stmt.result)
            nodes_to_insert.append(func.Invoke(inputs, callee=self.initialize_kernel))

        for n in nodes_to_insert:
            n.insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=True)

    @rewriter.register(move.CZ)
    def _(self, node: move.CZ, atom_state: atom.AtomState) -> rewrite_abc.RewriteResult:
        controls, targets, _ = atom_state.data.get_qubit_pairing(
            node.zone_address, self.arch_spec
        )
        controls_ssa: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in controls
        )
        targets_ssa: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in targets
        )

        (control_reg := ilist.New(controls_ssa)).insert_before(node)
        (target_reg := ilist.New(targets_ssa)).insert_before(node)
        gate_stmts.CZ(control_reg.result, target_reg.result).insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class InsertMeasurements(rewrite_abc.RewriteRule):
    physical_ssa_values: dict[int, ir.SSAValue]
    move_exec_analysis: ForwardFrame[atom.MoveExecution]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, move.GetFutureResult):
            return rewrite_abc.RewriteResult()

        result = self.move_exec_analysis.get(node.result)
        if not isinstance(result, atom.MeasureResult):
            return rewrite_abc.RewriteResult()

        node.replace_by(
            func.Invoke(
                (self.physical_ssa_values[result.qubit_id],), callee=qubit.measure
            )
        )
        return rewrite_abc.RewriteResult(has_done_something=True)
