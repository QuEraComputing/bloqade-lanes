from dataclasses import dataclass, field
from typing import Any

from bloqade.squin.gate import stmts as gate_stmts
from kirin import ir
from kirin.dialects import func, ilist, py
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import LaneAddress, MoveType

from . import utils


@dataclass
class InsertQubits(rewrite_abc.RewriteRule):
    physical_ssa_values: list[ir.SSAValue] = field(default_factory=list, init=False)

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        for location_addr in node.location_addresses:
            (new_qubit := qubit.stmts.New()).insert_before(node)
            self.physical_ssa_values.append(new_qubit.result)

        node.delete()

        return rewrite_abc.RewriteResult(has_done_something=True)


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
class InsertGates(AtomStateRewriter):
    atom_state_map: dict[ir.Statement, atom.AtomStateType]
    initialize_kernel: ir.Method[
        [float, float, float, ilist.IList[qubit.Qubit, Any]], None
    ]

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
                    move.PhysicalInitialize,
                    move.CZ,
                ),
            )
            and isinstance(atom_state := self.atom_state_map.get(node), atom.AtomState)
        ):
            return rewrite_abc.RewriteResult()

        rewriter = getattr(self, f"rewrite_{type(node).__name__}")
        return rewriter(atom_state, node)

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
        (
            gate_stmts.U3(zero.result, node.rotation_angle, zero.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_GlobalRz(
        self, atom_state: atom.AtomState, node: move.GlobalRz
    ) -> rewrite_abc.RewriteResult:
        (zero := py.Constant(0.0)).insert_before(node)
        (reg := ilist.New(self.physical_ssa_values)).insert_before(node)
        (
            gate_stmts.U3(zero.result, node.rotation_angle, zero.result, reg.result)
        ).insert_before(node)
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
        (
            gate_stmts.U3(node.rotation_angle, phi.result, lam.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_GlobalR(
        self, atom_state: atom.AtomState, node: move.GlobalR
    ) -> rewrite_abc.RewriteResult:
        (quarter_turn := py.Constant(0.25)).insert_before(node)
        (phi := py.Sub(quarter_turn.result, node.axis_angle)).insert_before(node)
        (lam := py.Sub(node.axis_angle, quarter_turn.result)).insert_before(node)
        (reg := ilist.New(self.physical_ssa_values)).insert_before(node)
        (
            gate_stmts.U3(node.rotation_angle, phi.result, lam.result, reg.result)
        ).insert_before(node)
        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_GetMeasurementResult(
        self, atom_state: atom.AtomState, node: move.GetMeasurementResult
    ) -> rewrite_abc.RewriteResult:
        qubit_ssa = self.get_qubit_ssa(atom_state, node.location_address)
        if qubit_ssa is None:
            return rewrite_abc.RewriteResult()

        node.replace_by(func.Invoke((qubit_ssa,), callee=qubit.measure))

        return rewrite_abc.RewriteResult(has_done_something=True)

    def rewrite_PhysicalInitialize(
        self, atom_state: atom.AtomState, node: move.PhysicalInitialize
    ) -> rewrite_abc.RewriteResult:
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

    def rewrite_CZ(
        self, atom_state: atom.AtomState, node: move.CZ
    ) -> rewrite_abc.RewriteResult:
        controls, targets, _ = atom_state.get_qubit_pairing(
            node.zone_address, self.arch_spec
        )
        assert len(atom_state.locations) == len(
            self.physical_ssa_values
        ), "Mismatch between atom state and physical SSA values"
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
class InsertNoise(AtomStateRewriter):
    atom_state_map: dict[ir.Statement, atom.AtomStateType]
    lane_noise: dict[LaneAddress, ir.Method[[qubit.Qubit], None]] = field(
        default_factory=dict
    )
    bus_idle_noise: dict[
        tuple[MoveType, int], ir.Method[[ilist.IList[qubit.Qubit, Any]], None]
    ] = field(default_factory=dict)
    cz_unpaired_noise: ir.Method[[ilist.IList[qubit.Qubit, Any]], None] | None = field(
        default=None
    )

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not (
            isinstance(
                node,
                (
                    move.Move,
                    move.CZ,
                ),
            )
            and isinstance(atom_state := self.atom_state_map.get(node), atom.AtomState)
        ):
            return rewrite_abc.RewriteResult()

        rewriter = getattr(self, f"rewrite_{type(node).__name__}")
        return rewriter(atom_state, node)

    def rewrite_Move(
        self, atom_state: atom.AtomState, node: move.Move
    ) -> rewrite_abc.RewriteResult:
        if len(node.lanes) == 0:
            return rewrite_abc.RewriteResult()

        first_lane = node.lanes[0]

        stmts_to_insert: list[ir.Statement] = []
        stationary_qubits: set[ir.SSAValue] = set(self.physical_ssa_values)
        for lane in node.lanes:
            src, _ = self.arch_spec.get_endpoints(lane)

            qubit_ssa = self.get_qubit_ssa(atom_state, src)
            if qubit_ssa is None:
                return rewrite_abc.RewriteResult()

            stationary_qubits.discard(qubit_ssa)
            noise_method = self.lane_noise.get(lane)
            if noise_method is None:
                return rewrite_abc.RewriteResult()

            stmts_to_insert.append(func.Invoke((qubit_ssa,), callee=noise_method))

        bus_idle_method = self.bus_idle_noise.get(
            (first_lane.move_type, first_lane.bus_id)
        )
        if bus_idle_method is None:
            return rewrite_abc.RewriteResult()

        idle_qubits = tuple(stationary_qubits)
        stmts_to_insert.append(idle_reg := ilist.New(idle_qubits))
        stmts_to_insert.append(func.Invoke((idle_reg.result,), callee=bus_idle_method))

        for stmt in stmts_to_insert:
            stmt.insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=len(stmts_to_insert) > 0)

    def rewrite_CZ(
        self, atom_state: atom.AtomState, node: move.CZ
    ) -> rewrite_abc.RewriteResult:
        if self.cz_unpaired_noise is None:
            return rewrite_abc.RewriteResult()

        assert len(atom_state.locations) == len(
            self.physical_ssa_values
        ), "Mismatch between atom state and physical SSA values"

        _, _, unpaired = atom_state.get_qubit_pairing(node.zone_address, self.arch_spec)

        unpaired_qubits: tuple[ir.SSAValue, ...] = tuple(
            self.physical_ssa_values[i] for i in unpaired
        )

        (unpaired_reg := ilist.New(unpaired_qubits)).insert_before(node)
        func.Invoke(
            (unpaired_reg.result,), callee=self.cz_unpaired_noise
        ).insert_before(node)

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class CleanUpMoveDialect(rewrite_abc.RewriteRule):
    atom_state_map: dict[ir.Statement, atom.AtomStateType]

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
