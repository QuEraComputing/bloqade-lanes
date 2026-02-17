from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import starmap

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    SingleZonePlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.lattice import ExecuteMeasure
from bloqade.lanes.analysis.placement.strategy import PlacementStrategyABC
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers, move_to_left


@dataclass(frozen=True)
class MoveOp:
    """Data class to store a move operation along with its source and destination addresses."""

    arch_spec: layout.ArchSpec
    src: layout.LocationAddress
    dst: layout.LocationAddress

    @cached_property
    def src_position(self) -> tuple[float, float]:
        return self.arch_spec.get_position(self.src)

    @cached_property
    def dst_position(self) -> tuple[float, float]:
        return self.arch_spec.get_position(self.dst)


def check_conflict(m0: MoveOp, m1: MoveOp):
    def check_coord_conflict(
        src0: float, dst0: float, src1: float, dst1: float
    ) -> bool:
        dir_src = (src1 - src0) // abs(src1 - src0) if src1 != src0 else 0
        dir_dst = (dst1 - dst0) // abs(dst1 - dst0) if dst1 != dst0 else 0
        return dir_src != dir_dst

    return any(
        starmap(
            check_coord_conflict,
            zip(m0.src_position, m1.src_position, m0.dst_position, m1.dst_position),
        )
    )


@dataclass
class LogicalPlacementMethods:
    arch_spec: layout.ArchSpec

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        for addr in initial_layout:
            if addr.word_id >= 2:
                raise ValueError(
                    "Initial layout contains invalid word id for logical arch"
                )
            if addr.site_id >= 5:
                raise ValueError(
                    "Initial layout should only site ids < 5 for logical arch"
                )

    def _word_balance(
        self, state: ConcreteState, controls: tuple[int, ...], targets: tuple[int, ...]
    ) -> int:
        word_move_counts = {0: 0, 1: 0}
        for c, t in zip(controls, targets):
            c_addr = state.layout[c]
            t_addr = state.layout[t]
            if c_addr.word_id != t_addr.word_id:
                word_move_counts[c_addr.word_id] += state.move_count[c]
                word_move_counts[t_addr.word_id] += state.move_count[t]

        return 0 if word_move_counts[0] <= word_move_counts[1] else 1

    def _pick_move_by_conflict(
        self,
        moves: list[MoveOp],
        move1: MoveOp,
        move2: MoveOp,
    ) -> MoveOp:
        def count_conflicts(proposed_move: MoveOp) -> int:
            return sum(
                check_conflict(
                    proposed_move,
                    existing_move,
                )
                for existing_move in moves
            )

        return move1 if count_conflicts(move1) <= count_conflicts(move2) else move2

    def _pick_move(
        self,
        state: ConcreteState,
        moves: list[MoveOp],
        start_word_id: int,
        control: int,
        target: int,
    ) -> MoveOp:
        c_addr = state.layout[control]
        t_addr = state.layout[target]

        c_addr_dst = layout.LocationAddress(t_addr.word_id, t_addr.site_id + 5)
        t_addr_dst = layout.LocationAddress(c_addr.word_id, c_addr.site_id + 5)
        c_move_count = state.move_count[control]
        t_move_count = state.move_count[target]

        move_t_to_c = MoveOp(self.arch_spec, t_addr, t_addr_dst)
        move_c_to_t = MoveOp(self.arch_spec, c_addr, c_addr_dst)

        if c_addr.word_id == t_addr.word_id:
            if c_move_count < t_move_count:
                return move_c_to_t
            if c_move_count > t_move_count:
                return move_t_to_c
            return self._pick_move_by_conflict(moves, move_c_to_t, move_t_to_c)
        if t_addr.word_id == start_word_id:
            return move_t_to_c
        return move_c_to_t

    def _update_positions(
        self,
        state: ConcreteState,
        moves: list[MoveOp],
    ) -> ConcreteState:
        new_positions: dict[int, layout.LocationAddress] = {}
        for move in moves:
            src_qubit = state.get_qubit_id(move.src)
            assert src_qubit is not None, "Source qubit must exist in state"
            new_positions[src_qubit] = move.dst

        new_layout = tuple(
            new_positions.get(i, loc) for i, loc in enumerate(state.layout)
        )
        new_move_count = list(state.move_count)
        for qid in new_positions:
            new_move_count[qid] += 1

        return replace(state, layout=new_layout, move_count=tuple(new_move_count))

    def _sorted_cz_pairs_by_move_count(
        self, state: ConcreteState, controls: tuple[int, ...], targets: tuple[int, ...]
    ) -> list[tuple[int, int]]:
        return sorted(
            zip(controls, targets),
            key=lambda x: state.move_count[x[0]] + state.move_count[x[1]],
            reverse=True,
        )


@dataclass
class LogicalPlacementStrategy(LogicalPlacementMethods, SingleZonePlacementStrategyABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)

    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState:
        start_word_id = self._word_balance(state, controls, targets)
        moves: list[MoveOp] = []
        for c, t in self._sorted_cz_pairs_by_move_count(state, controls, targets):
            moves.append(self._pick_move(state, moves, start_word_id, c, t))
        return self._update_positions(state, moves)

    def compute_moves(
        self, state_before: ConcreteState, state_after: ConcreteState
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        return compute_move_layers(self.arch_spec, state_before, state_after)


@dataclass
class LogicalPlacementStrategyNoHome(LogicalPlacementMethods, PlacementStrategyABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)

    def compute_moves(
        self, state_before: ConcreteState, state_after: ConcreteState
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        return compute_move_layers(self.arch_spec, state_before, state_after)

    def choose_return_layout(
        self,
        state_before: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
        del controls, targets
        left_sites = {
            layout.LocationAddress(word_id, site_id)
            for word_id in range(2)
            for site_id in range(5)
        }

        used_left_sites = {addr for addr in state_before.layout if addr.site_id < 5}
        used_left_sites |= {addr for addr in state_before.occupied if addr.site_id < 5}
        available_left_sites = left_sites - used_left_sites
        return_layout = list(state_before.layout)

        def distance_key(
            right_addr: layout.LocationAddress, left_addr: layout.LocationAddress
        ) -> tuple[int, int, int, int]:
            right_row = right_addr.site_id - 5
            word_distance = 0 if left_addr.word_id == right_addr.word_id else 1
            site_distance = abs(left_addr.site_id - right_row)
            return (
                word_distance,
                site_distance,
                left_addr.word_id,
                left_addr.site_id,
            )

        for qid, addr in enumerate(state_before.layout):
            if addr.site_id < 5:
                continue
            if not available_left_sites:
                raise ValueError(
                    "No empty left-column site available for right-column return move"
                )
            best_left_site = min(
                available_left_sites,
                key=lambda left_site: distance_key(addr, left_site),
            )
            return_layout[qid] = best_left_site
            available_left_sites.remove(best_left_site)

        mid_state = ConcreteState(
            occupied=state_before.occupied,
            layout=tuple(return_layout),
            move_count=tuple(
                mc + int(src != dst)
                for mc, src, dst in zip(
                    state_before.move_count,
                    state_before.layout,
                    return_layout,
                )
            ),
        )
        _, left_move_layers = move_to_left(self.arch_spec, state_before, mid_state)
        return mid_state, left_move_layers

    def cz_placements(
        self, state: AtomState, controls: tuple[int, ...], targets: tuple[int, ...]
    ) -> AtomState:
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()

        if not isinstance(state, ConcreteState):
            return AtomState.top()

        mid_state, left_move_layers = self.choose_return_layout(
            state, controls, targets
        )
        state_after = self.desired_cz_layout(mid_state, controls, targets)
        final_move_layers = self.compute_moves(mid_state, state_after)

        return ExecuteCZ(
            occupied=state_after.occupied,
            layout=state_after.layout,
            move_count=state_after.move_count,
            active_cz_zones=frozenset([layout.ZoneAddress(0)]),
            move_layers=(left_move_layers + final_move_layers),
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state

        if len(qubits) != len(state.layout):
            return AtomState.bottom()

        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(layout.ZoneAddress(0) for _ in qubits),
        )
