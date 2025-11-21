import abc
from dataclasses import dataclass, field

from kirin import ir
from kirin.dialects import func
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.analysis import placement
from bloqade.lanes.dialects import circuit, move
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress, ZoneAddress


@dataclass
class MoveSchedulerABC(abc.ABC):
    arch_spec: ArchSpec
    zone_address_map: dict[LocationAddress, tuple[ZoneAddress, int]] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self):
        for zone_id, zone in enumerate(self.arch_spec.zones):
            index = 0
            for word_id in zone:
                word = self.arch_spec.words[word_id]
                for site_id, _ in enumerate(word.sites):
                    loc_addr = LocationAddress(word_id=word_id, site_id=site_id)
                    zone_address = ZoneAddress(zone_id)
                    self.zone_address_map[loc_addr] = (zone_address, index)
                    index += 1

    def compute_zone_addresses(
        self,
        locations: tuple[LocationAddress, ...],
    ) -> list[ZoneAddress]:
        return sorted(
            set(self.zone_address_map[loc][0] for loc in locations),
            key=lambda za: za.zone_id,
        )

    @abc.abstractmethod
    def compute_moves(
        self,
        state_before: placement.AtomState,
        state_after: placement.AtomState,
    ) -> list[tuple[LaneAddress, ...]]:
        pass


@dataclass
class InsertMoves(RewriteRule):
    move_heuristic: MoveSchedulerABC
    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, circuit.QuantumStmt):
            return RewriteResult()

        moves = self.move_heuristic.compute_moves(
            self.placement_analysis[node.state_before],
            self.placement_analysis[node.state_after],
        )

        if len(moves) == 0:
            return RewriteResult()

        for move_lanes in moves:
            move.Move(lanes=move_lanes).insert_before(node)

        return RewriteResult(has_done_something=True)


class InsertPalindromeMoves(RewriteRule):
    """This rewrite goes through a static circuit and for every move statement,
    it inserts a reverse move statement at the end of the circuit to undo the move.

    The idea here you can cancel out some systematic move errors by playing moves backwards.

    """

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, circuit.StaticCircuit):
            return RewriteResult()

        yield_stmt = node.body.blocks[0].last_stmt
        assert isinstance(yield_stmt, circuit.Yield)

        for stmt in node.body.walk(reverse=True):
            if not isinstance(stmt, move.Move):
                continue

            reverse_moves = tuple(lane.reverse() for lane in stmt.lanes[::-1])
            move.Move(lanes=reverse_moves).insert_before(yield_stmt)

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteCZ(RewriteRule):
    """Rewrite CZ circuit statements to move CZ statements.

    Requires placement analysis to know where the qubits are located and a move heuristic
    to determine which zone addresses to use for the CZ moves.

    """

    placement_analysis: dict[ir.SSAValue, placement.AtomState]
    move_heuristic: MoveSchedulerABC

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, circuit.CZ):
            return RewriteResult()

        state_after = self.placement_analysis.get(node.state_after)

        if not isinstance(state_after, placement.ConcreteState):
            return RewriteResult()

        zone_addresses = self.move_heuristic.compute_zone_addresses(
            tuple(state_after.layout[i] for i in node.controls + node.targets)
        )

        for zone_address in zone_addresses:
            move.CZ(zone_address=zone_address).insert_after(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteR(RewriteRule):
    """Rewrite R circuit statements to move R statements."""

    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, circuit.R):
            return RewriteResult()

        state_after = self.placement_analysis.get(node.state_after)

        if not isinstance(state_after, placement.ConcreteState):
            return RewriteResult()

        is_global = len(
            state_after.occupied
        ) == 0 and len(  # static circuit includes all atoms
            state_after.layout
        ) == len(
            node.qubits
        )  # gate statement includes all atoms

        if is_global:
            move.GlobalR(
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)
        else:
            location_addresses = tuple(state_after.layout[i] for i in node.qubits)
            move.LocalR(
                location_addresses=location_addresses,
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteRz(RewriteRule):
    """Rewrite Rz circuit statements to move Rz statements.

    requires placement analysis to know where the qubits are located to do the rewrite.

    """

    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, circuit.Rz):
            return RewriteResult()

        state_after = self.placement_analysis.get(node.state_after)

        if not isinstance(state_after, placement.ConcreteState):
            # do not know the location of the qubits, cannot rewrite
            return RewriteResult()

        is_global = len(
            state_after.occupied
        ) == 0 and len(  # static circuit includes all atoms
            state_after.layout
        ) == len(
            node.qubits
        )  # gate statement includes all atoms

        if is_global:
            move.GlobalRz(
                rotation_angle=node.rotation_angle,
            ).insert_after(node)
        else:
            location_addresses = tuple(state_after.layout[i] for i in node.qubits)
            move.LocalRz(
                location_addresses=location_addresses,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class InsertMeasure(RewriteRule):

    placement_analysis: dict[ir.SSAValue, placement.AtomState]
    move_heuristic: MoveSchedulerABC

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, circuit.EndMeasure):
            return RewriteResult()

        if not isinstance(
            atom_state := self.placement_analysis.get(state_after := node.results[0]),
            placement.ConcreteState,
        ):
            return RewriteResult()

        zone_addresses = self.move_heuristic.compute_zone_addresses(atom_state.layout)

        futures = {}
        for zone_address in zone_addresses:
            measure_stmt = move.EndMeasure(zone_address=zone_address)
            measure_stmt.insert_before(node)
            futures[zone_address] = measure_stmt.result

        for qubit_index, result in zip(node.qubits, node.results[1:]):
            loc_addr = atom_state.layout[qubit_index]
            zone_address, index = self.move_heuristic.zone_address_map[loc_addr]
            get_result_stmt = move.GetMeasurementResult(
                measurement_future=futures[zone_address],
                location_address=loc_addr,
            )
            get_result_stmt.insert_before(node)
            result.replace_by(get_result_stmt.result)

        state_after.replace_by(node.state_before)
        node.delete()
        return RewriteResult(has_done_something=True)


class LiftMoveStatements(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        if not (
            type(node) in move.dialect.stmts
            and (parent_stmt := node.parent_stmt) is not None
        ):
            return RewriteResult()

        node.detach()
        node.insert_before(parent_stmt)

        return RewriteResult(has_done_something=True)


class RemoveNoOpCircuits(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        if not (
            isinstance(node, circuit.StaticCircuit)
            and isinstance(yield_stmt := node.body.blocks[0].first_stmt, circuit.Yield)
        ):
            return RewriteResult()

        for yield_result, node_result in zip(
            yield_stmt.classical_results, node.results
        ):
            node_result.replace_by(yield_result)

        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class InsertInitialize(RewriteRule):
    initial_layout: tuple[LocationAddress, ...]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, func.Function):
            return RewriteResult()

        first_stmt = node.body.blocks[0].first_stmt

        if first_stmt is None or isinstance(first_stmt, move.Initialize):
            return RewriteResult()

        move.Initialize(location_addresses=self.initial_layout).insert_before(
            first_stmt
        )
        return RewriteResult(has_done_something=True)
