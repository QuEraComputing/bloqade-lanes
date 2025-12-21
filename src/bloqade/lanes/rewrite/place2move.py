import abc
from dataclasses import dataclass, field
from functools import singledispatchmethod

from bloqade.analysis import address
from kirin import ir
from kirin.dialects import cf, func, py
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.analysis import placement
from bloqade.lanes.dialects import move, place
from bloqade.lanes.layout import ArchSpec, LaneAddress, LocationAddress, ZoneAddress
from bloqade.lanes.types import StateType


@dataclass
class MoveSchedulerABC(abc.ABC):
    arch_spec: ArchSpec

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
        if not isinstance(node, place.QuantumStmt):
            return RewriteResult()

        moves = self.move_heuristic.compute_moves(
            self.placement_analysis.get(node.state_before, placement.AtomState.top()),
            self.placement_analysis.get(node.state_after, placement.AtomState.top()),
        )

        if len(moves) == 0:
            return RewriteResult()

        for move_lanes in moves:
            (current_state := move.Load()).insert_before(node)
            (move.Move(current_state.result, lanes=move_lanes)).insert_before(node)

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

        for stmt in node.body.walk(reverse=True):
            if not isinstance(stmt, move.Move):
                continue
            (current_state := move.Load()).insert_before(yield_stmt)
            reverse_moves = tuple(lane.reverse() for lane in stmt.lanes[::-1])
            (move.Move(current_state.result, lanes=reverse_moves)).insert_before(
                yield_stmt
            )

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteCZ(RewriteRule):
    """Rewrite CZ circuit statements to move CZ statements.

    Requires placement analysis to know where the qubits are located and a move heuristic
    to determine which zone addresses to use for the CZ moves.

    """

    move_heuristic: MoveSchedulerABC
    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, place.CZ):
            return RewriteResult()

        state_after = self.placement_analysis.get(node.state_after)

        if not isinstance(state_after, placement.ExecuteCZ):
            return RewriteResult()

        stmts_to_insert: list[move.CZ | move.Load] = []
        for cz_zone_address in state_after.active_cz_zones:
            stmts_to_insert.append(current_state := move.Load())
            stmts_to_insert.append(
                move.CZ(
                    current_state.result,
                    zone_address=cz_zone_address,
                )
            )

        for stmt in reversed(stmts_to_insert):
            stmt.insert_after(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteR(RewriteRule):
    """Rewrite R circuit statements to move R statements."""

    move_heuristic: MoveSchedulerABC
    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, place.R):
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

        current_state = move.Load()
        if is_global:
            move.GlobalR(
                current_state.result,
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)
        else:
            location_addresses = tuple(state_after.layout[i] for i in node.qubits)
            move.LocalR(
                current_state.result,
                location_addresses=location_addresses,
                axis_angle=node.axis_angle,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)

        current_state.insert_before(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class RewriteRz(RewriteRule):
    """Rewrite Rz circuit statements to move Rz statements.

    requires placement analysis to know where the qubits are located to do the rewrite.

    """

    move_heuristic: MoveSchedulerABC
    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, place.Rz):
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

        current_state = move.Load()
        if is_global:
            move.GlobalRz(
                current_state.result,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)
        else:
            location_addresses = tuple(state_after.layout[i] for i in node.qubits)
            move.LocalRz(
                current_state.result,
                location_addresses=location_addresses,
                rotation_angle=node.rotation_angle,
            ).insert_after(node)
        current_state.insert_before(node)

        node.state_after.replace_by(node.state_before)
        node.delete()

        return RewriteResult(has_done_something=True)


@dataclass
class InsertMeasure(RewriteRule):

    move_heuristic: MoveSchedulerABC
    placement_analysis: dict[ir.SSAValue, placement.AtomState]

    def rewrite_Statement(self, node: ir.Statement):
        if not isinstance(node, place.EndMeasure):
            return RewriteResult()

        if not isinstance(
            atom_state := self.placement_analysis.get(state_after := node.results[0]),
            placement.ExecuteMeasure,
        ):
            return RewriteResult()

        (current_state := move.Load()).insert_before(node)
        zone_addresses = tuple(set(atom_state.zone_maps))
        (
            future_stmt := move.EndMeasure(
                current_state.result, zone_addresses=zone_addresses
            )
        ).insert_before(node)

        future_results: dict[ZoneAddress, ir.SSAValue] = {}
        for zone_address in zone_addresses:
            (
                future_result := move.GetFutureResult(
                    future_stmt.result, zone_address=zone_address
                )
            ).insert_before(node)
            future_results[zone_address] = future_result.result

        for result, zone_address, loc_addr in zip(
            node.results[1:], atom_state.zone_maps, atom_state.layout
        ):
            future_result = future_results[zone_address]
            (
                index_stmt := move.GetZoneIndex(
                    zone_address=zone_address, location_address=loc_addr
                )
            ).insert_before(node)

            (
                get_item_stmt := py.GetItem(future_result, index_stmt.result)
            ).insert_before(node)

            result.replace_by(get_item_stmt.result)

        state_after.replace_by(node.state_before)
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
        move.LogicalInitialize(
            current_state.result,
            tuple(thetas),
            tuple(phis),
            tuple(lams),
            location_addresses=tuple(location_addresses),
        ).insert_before(stmt)

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
        move.Fill(
            current_state.result, location_addresses=self.initial_layout
        ).insert_before(first_stmt)

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


@dataclass
class FixUpStateFlow(RewriteRule):
    current_states: ir.SSAValue | None = field(default=None, init=False)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        return self.rewrite_node(node)

    @singledispatchmethod
    def rewrite_node(self, node: ir.Statement) -> RewriteResult:
        return RewriteResult()

    @rewrite_node.register(move.Load)
    def rewrite_Load(self, node: move.Load):
        if len(uses := node.result.uses) != 1:
            # Something is wrong, we cannot fix up
            return RewriteResult()

        (use,) = uses

        if not isinstance(stmt := use.stmt, move.StatefulStatement):
            return RewriteResult()

        if self.current_states is None:
            self.current_states = stmt.result
            return RewriteResult()
        else:
            stmt.current_state.replace_by(self.current_states)
            self.current_states = stmt.result
            return RewriteResult(has_done_something=True)

    @rewrite_node.register(move.Store)
    def rewrite_Store(self, node: move.Store):
        if self.current_states is None:
            # Something is wrong, we cannot fix up
            return RewriteResult()

        node.current_state.replace_by(self.current_states)
        self.current_state = None
        return RewriteResult(has_done_something=True)

    @rewrite_node.register(cf.Branch)
    def rewrite_Branch(self, node: cf.Branch):
        if self.current_states is None:
            return RewriteResult()

        node.replace_by(
            cf.Branch(
                successor=node.successor,
                arguments=(self.current_states,) + node.arguments,
            )
        )
        self.current_states = None
        return RewriteResult(has_done_something=True)

    @rewrite_node.register(cf.ConditionalBranch)
    def rewrite_ConditionalBranch(self, node: cf.ConditionalBranch):
        if self.current_states is None:
            return RewriteResult()

        node.replace_by(
            cf.ConditionalBranch(
                cond=node.cond,
                then_successor=node.then_successor,
                then_arguments=(self.current_states,) + node.then_arguments,
                else_successor=node.else_successor,
                else_arguments=(self.current_states,) + node.else_arguments,
            )
        )

        self.current_states = None
        return RewriteResult(has_done_something=True)

    def is_entry_block(self, node: ir.Block) -> bool:
        if (parent_stmt := node.parent_stmt) is None:
            return False

        callable_stmt_trait = parent_stmt.get_trait(ir.CallableStmtInterface)

        if callable_stmt_trait is None:
            return False

        parent_region = callable_stmt_trait.get_callable_region(parent_stmt)
        return parent_region._block_idx[node] == 0

    def rewrite_Block(self, node: ir.Block):
        if self.is_entry_block(node):
            return RewriteResult()

        if self.current_states is not None:
            # something has gone wrong, we cannot fix up
            return RewriteResult()

        self.current_states = node.args.insert_from(0, StateType, "current_state")
        return RewriteResult(has_done_something=True)
