import abc
from dataclasses import dataclass
from functools import singledispatchmethod

from bloqade.analysis import address
from kirin import ir
from kirin.dialects import func
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.analysis import placement
from bloqade.lanes.dialects import move, place
from bloqade.lanes.layout import ArchSpec, LaneAddress, LocationAddress


@dataclass
class MoveSchedulerABC(abc.ABC):
    arch_spec: ArchSpec

    @abc.abstractmethod
    def compute_moves(
        self,
        state_before: placement.AtomState,
        state_after: placement.AtomState,
    ) -> list[tuple[LaneAddress, ...]]: ...


@dataclass
class InsertMoves(RewriteRule):
    move_heuristic: MoveSchedulerABC
    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, place.QuantumStmt):
            return RewriteResult()

        moves = self.move_heuristic.compute_moves(
            self.placement_analysis.get(node.state_before, placement.AtomState.top()),
            self.placement_analysis.get(node.state_after, placement.AtomState.top()),
        )

        if len(moves) == 0:
            return RewriteResult()

        (current_state := move.Load()).insert_before(node)
        for move_lanes in moves:
            (
                current_state := move.Move(current_state.result, lanes=move_lanes)
            ).insert_before(node)
        (move.Store(current_state.result)).insert_before(node)

        return RewriteResult(has_done_something=True)


class InsertPalindromeMoves(RewriteRule):
    """This rewrite goes through a static circuit and for every move statement,
    it inserts a reverse move statement at the end of the circuit to undo the move.

    The idea here you can cancel out some systematic move errors by playing moves backwards.

    """

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, place.StaticPlacement):
            return RewriteResult()

        yield_stmt = node.body.blocks[0].last_stmt
        assert isinstance(yield_stmt, place.Yield)

        (current_state := move.Load()).insert_before(yield_stmt)
        for stmt in node.body.walk(reverse=True):
            if not isinstance(stmt, move.Move):
                continue
            reverse_moves = tuple(lane.reverse() for lane in stmt.lanes[::-1])
            (
                current_state := move.Move(current_state.result, lanes=reverse_moves)
            ).insert_before(yield_stmt)

        (move.Store(current_state.result)).insert_before(yield_stmt)

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteGates(RewriteRule):
    """Rewrite R circuit statements to move R statements."""

    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    @singledispatchmethod
    def stmts_to_insert(
        self, node: ir.Statement, state_after: placement.AtomState
    ) -> list[ir.Statement] | None:
        return None

    @stmts_to_insert.register(place.CZ)
    def _(self, node: place.CZ, state_after: placement.AtomState):
        if not isinstance(state_after, placement.ExecuteCZ):
            return None

        stmts = []
        stmts.append(current_state := move.Load())

        for cz_zone_address in state_after.active_cz_zones:
            stmts.append(
                current_state := move.CZ(
                    current_state.result,
                    zone_address=cz_zone_address,
                )
            )

        stmts.append(move.Store(current_state.result))

        return stmts

    def is_global(
        self, node: place.R | place.Rz, state_after: placement.ConcreteState
    ) -> bool:
        return len(
            state_after.occupied
        ) == 0 and len(  # static circuit includes all atoms
            state_after.layout
        ) == len(
            node.qubits
        )  # gate statement includes all atoms

    @stmts_to_insert.register(place.R)
    def _(self, node: place.R, state_after: placement.AtomState):
        if not isinstance(state_after, placement.ConcreteState):
            return None

        is_global = self.is_global(node, state_after)

        current_state = move.Load()
        if is_global:
            return [
                current_state,
                (
                    mid_state := move.GlobalR(
                        current_state.result,
                        axis_angle=node.axis_angle,
                        rotation_angle=node.rotation_angle,
                    )
                ),
                move.Store(mid_state.result),
            ]
        else:
            location_addresses = tuple(state_after.layout[i] for i in node.qubits)
            return [
                current_state,
                (
                    mid_state := move.LocalR(
                        current_state.result,
                        location_addresses=location_addresses,
                        axis_angle=node.axis_angle,
                        rotation_angle=node.rotation_angle,
                    )
                ),
                move.Store(mid_state.result),
            ]

    @stmts_to_insert.register(place.Rz)
    def _(self, node: place.Rz, state_after: placement.AtomState):
        if not isinstance(state_after, placement.ConcreteState):
            return None

        is_global = len(
            state_after.occupied
        ) == 0 and len(  # static circuit includes all atoms
            state_after.layout
        ) == len(
            node.qubits
        )  # gate statement includes all atoms

        current_state = move.Load()
        if is_global:
            return [
                current_state,
                mid_state := move.GlobalRz(
                    current_state.result,
                    rotation_angle=node.rotation_angle,
                ),
                move.Store(mid_state.result),
            ]
        else:
            location_addresses = tuple(state_after.layout[i] for i in node.qubits)
            return [
                current_state,
                mid_state := move.LocalRz(
                    current_state.result,
                    location_addresses=location_addresses,
                    rotation_angle=node.rotation_angle,
                ),
                move.Store(mid_state.result),
            ]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, place.QuantumStmt):
            return RewriteResult()

        stmts = self.stmts_to_insert(
            node,
            self.placement_analysis.get(node.state_after),
        )

        if stmts is None:
            return RewriteResult()

        for stmt in reversed(stmts):
            stmt.insert_after(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class InsertMeasure(RewriteRule):

    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, place.EndMeasure):
            return RewriteResult()

        if not isinstance(
            atom_state := self.placement_analysis.get(node.results[0]),
            placement.ExecuteMeasure,
        ):
            return RewriteResult()

        (current_state := move.Load()).insert_before(node)
        zone_addresses = tuple(sorted(set(atom_state.zone_maps)))
        (
            future_stmt := move.EndMeasure(
                current_state.result, zone_addresses=zone_addresses
            )
        ).insert_before(node)

        for result, zone_address, loc_addr in zip(
            node.results[1:], atom_state.zone_maps, atom_state.layout, strict=True
        ):
            (
                get_item_stmt := move.GetFutureResult(
                    future_stmt.result,
                    zone_address=zone_address,
                    location_address=loc_addr,
                )
            ).insert_before(node)

            result.replace_by(get_item_stmt.result)

        node.state_after.replace_by(node.state_before)
        node.delete()
        return RewriteResult(has_done_something=True)


class LiftMoveStatements(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        if not (
            type(node) not in place.dialect.stmts
            and isinstance((parent_stmt := node.parent_stmt), place.StaticPlacement)
        ):
            return RewriteResult()

        node.detach()
        node.insert_before(parent_stmt)

        return RewriteResult(has_done_something=True)


class RemoveNoOpStaticPlacements(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement):
        if not (
            isinstance(node, place.StaticPlacement)
            and isinstance(yield_stmt := node.body.blocks[0].first_stmt, place.Yield)
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
    address_entries: dict[ir.SSAValue, address.Address]
    initial_layout: tuple[LocationAddress, ...]

    def rewrite_Block(self, node: ir.Block) -> RewriteResult:
        stmt = node.first_stmt
        thetas: list[ir.SSAValue] = []
        phis: list[ir.SSAValue] = []
        lams: list[ir.SSAValue] = []
        location_addresses: list[LocationAddress] = []

        while stmt is not None:
            if not isinstance(stmt, place.NewLogicalQubit):
                stmt = stmt.next_stmt
                continue

            if not isinstance(
                qubit_addr := self.address_entries.get(stmt.result),
                address.AddressQubit,
            ):
                return RewriteResult()

            if qubit_addr.data >= len(self.initial_layout):
                return RewriteResult()

            location_addresses.append(self.initial_layout[qubit_addr.data])
            thetas.append(stmt.theta)
            phis.append(stmt.phi)
            lams.append(stmt.lam)
            stmt = stmt.next_stmt
            if len(location_addresses) == len(self.initial_layout):
                break

        if stmt is None:
            return RewriteResult()

        (current_state := move.Load()).insert_before(stmt)
        (
            current_state := move.LogicalInitialize(
                current_state.result,
                tuple(thetas),
                tuple(phis),
                tuple(lams),
                location_addresses=tuple(location_addresses),
            )
        ).insert_before(stmt)
        (move.Store(current_state.result)).insert_before(stmt)

        return RewriteResult(has_done_something=True)


@dataclass
class InsertFill(RewriteRule):
    initial_layout: tuple[LocationAddress, ...]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, func.Function):
            return RewriteResult()

        first_stmt = node.body.blocks[0].first_stmt

        if first_stmt is None or isinstance(first_stmt, move.Fill):
            return RewriteResult()

        (current_state := move.Load()).insert_before(first_stmt)
        (
            current_state := move.Fill(
                current_state.result, location_addresses=self.initial_layout
            )
        ).insert_before(first_stmt)
        move.Store(current_state.result).insert_before(first_stmt)
        return RewriteResult(has_done_something=True)


class DeleteQubitNew(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not (isinstance(node, place.NewLogicalQubit) and len(node.result.uses) == 0):
            return RewriteResult()

        node.delete()

        return RewriteResult(has_done_something=True)


class DeleteInitialize(RewriteRule):
    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, place.Initialize):
            return RewriteResult()

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)
