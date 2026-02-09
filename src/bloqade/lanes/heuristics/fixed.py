from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import chain, combinations, starmap

from kirin import interp

from bloqade.lanes import layout
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import (
    ConcreteState,
    SingleZonePlacementStrategyABC,
)
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers


@dataclass(frozen=True)
class MoveOp:
    """Data class to store a move operation along with its source and destination addresses."""

    arch_spec: layout.ArchSpec
    """Architecture specification for move position lookups."""
    src: layout.LocationAddress
    """Source location address of the move."""
    dst: layout.LocationAddress
    """Destination location address of the move."""

    @cached_property
    def src_position(self) -> tuple[float, float]:
        return self.arch_spec.get_position(self.src)

    @cached_property
    def dst_position(self) -> tuple[float, float]:
        return self.arch_spec.get_position(self.dst)


def check_conflict(m0: MoveOp, m1: MoveOp):
    """Check if two move operations conflict based on their source and destination positions.

    A conflict occurs if the layout.direction of movement for any dimension differs between the two moves.
    Args:
        m0 (MoveOp): The first move operation.
        m1 (MoveOp): The second move operation.
    Returns:
        bool: True if there is a conflict, False otherwise.

    """

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
class LogicalPlacementStrategy(SingleZonePlacementStrategyABC):
    """A placement strategy that assumes a logical architecture.

    The logical architecture assumes 2 word buses (word_id 0 and 1) and a single word bus.
    This is equivalent to the generic architecture but with a hypercube dimension of 1,

    The idea is to keep the initial locations of the qubits are all on even site ids. Then when
    two qubits need to be entangled via a cz gate, one qubit (the control or target) is moved to the
    odd site id next to the other qubit. This ensures that no two qubits ever occupy the same
    location address and that there is always a clear path for qubits to traverse the architecture.

    The placement heuristic prioritizes balancing the number of moves each qubit has made, instead
    of prioritizing parallelism of moves.


    The hope is that this should balance out the number of moves across all qubits in the circuit.
    """

    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)

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

        # prioritize word move that reduces the max move count
        if word_move_counts[0] <= word_move_counts[1]:
            return 0
        else:
            return 1

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

        if count_conflicts(move1) <= count_conflicts(move2):
            return move1
        else:
            return move2

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
            if c_move_count < t_move_count:  # move control to target
                return move_c_to_t
            elif c_move_count > t_move_count:  # move target to control
                return move_t_to_c
            else:
                return self._pick_move_by_conflict(moves, move_c_to_t, move_t_to_c)
        elif t_addr.word_id == start_word_id:
            return move_t_to_c
        else:
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
        for qid in new_positions.keys():
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

    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState:
        # since cz gates are symmetric swap controls and targets based on
        # word_id and site_id the idea being to minimize the layout.directions
        # needed to rearrange qubits.
        start_word_id = self._word_balance(state, controls, targets)
        moves: list[MoveOp] = []

        # sort cz pairs by total move count descending order to prioritize
        # balancing moves across qubits and letting pairs with no moves go last
        # to pick the moves that cause least conflicts
        for c, t in self._sorted_cz_pairs_by_move_count(state, controls, targets):
            moves.append(self._pick_move(state, moves, start_word_id, c, t))

        return self._update_positions(state, moves)

    def compute_moves(
        self, state_before: ConcreteState, state_after: ConcreteState
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        return compute_move_layers(self.arch_spec, state_before, state_after)


@dataclass
class LogicalLayoutHeuristic(LayoutHeuristicABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)

    def score_parallelism(
        self,
        edges: dict[tuple[int, int], int],
        qubit_map: dict[int, layout.LocationAddress],
    ) -> int:
        move_weights = {}
        for n, m in combinations(qubit_map.keys(), 2):
            n, m = (min(n, m), max(n, m))
            edge_weight = edges.get((n, m))
            if edge_weight is None:
                continue

            addr_n = qubit_map[n]
            addr_m = qubit_map[m]
            site_diff = (addr_n.site_id - addr_m.site_id) // 2
            word_diff = addr_n.word_id - addr_m.word_id
            if word_diff != 0:
                edge_weight *= 2

            move_weights[(word_diff, site_diff)] = (
                move_weights.get((word_diff, site_diff), 0) + edge_weight
            )

        all_moves = list(move_weights.keys())
        score = 0
        for i, move_i in enumerate(all_moves):
            for move_j in all_moves[i + 1 :]:
                score += move_weights[move_i] + move_weights[move_j]

        return score

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[tuple[tuple[int, int], ...]],
    ) -> tuple[layout.LocationAddress, ...]:

        if len(all_qubits) > self.arch_spec.max_qubits:
            raise interp.InterpreterError(
                f"Number of qubits in circuit ({len(all_qubits)}) exceeds maximum supported by logical architecture ({self.arch_spec.max_qubits})"
            )

        edges = {}

        for control, target in chain.from_iterable(stages):
            n, m = min(control, target), max(control, target)
            edge_weight = edges.get((n, m), 0)
            edges[(n, m)] = edge_weight + 1

        available_addresses = set(
            [
                layout.LocationAddress(word_id, site_id)
                for word_id in range(len(self.arch_spec.words))
                for site_id in range(5)
            ]
        )

        qubit_map: dict[int, layout.LocationAddress] = {}
        layout_map: dict[layout.LocationAddress, int] = {}
        for qubit in sorted(all_qubits):

            scores: dict[layout.LocationAddress, int] = {}
            for addr in available_addresses:
                qubit_map = qubit_map.copy()
                qubit_map[qubit] = addr
                scores[addr] = self.score_parallelism(edges, qubit_map)

            best_addr = min(
                scores.keys(), key=lambda x: (scores[x], x.word_id, x.site_id)
            )
            available_addresses.remove(best_addr)
            qubit_map[qubit] = best_addr
            layout_map[best_addr] = qubit

        # invert layout
        final_layout = list(layout_map.keys())
        final_layout.sort(key=lambda x: layout_map[x])
        return tuple(final_layout)
