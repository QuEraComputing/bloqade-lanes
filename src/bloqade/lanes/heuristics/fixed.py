from dataclasses import dataclass, field, replace
from itertools import chain, combinations

from kirin import interp

from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.analysis.placement.lattice import AtomState, ConcreteState
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)
from bloqade.lanes.rewrite.circuit2move import MoveSchedulerABC


@dataclass
class LogicalPlacementStrategy(PlacementStrategyABC):
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

    def validate_initial_layout(
        self,
        initial_layout: tuple[LocationAddress, ...],
    ) -> None:
        for addr in initial_layout:
            if addr.word_id >= 2:
                raise ValueError(
                    "Initial layout contains invalid word id for logical arch"
                )
            if addr.site_id % 2 != 0:
                raise ValueError(
                    "Initial layout should only contain even site ids for fixed home location strategy"
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

    def _pick_mover_and_location(
        self,
        state: ConcreteState,
        start_word_id: int,
        control: int,
        target: int,
    ):
        c_addr = state.layout[control]
        t_addr = state.layout[target]
        if c_addr.word_id == t_addr.word_id:
            if (
                state.move_count[control] <= state.move_count[target]
            ):  # move control to target
                return control, t_addr
            else:  # move target to control
                return target, c_addr
        elif t_addr.word_id == start_word_id:
            return target, c_addr
        else:
            return control, t_addr

    def _update_positions(
        self,
        state: ConcreteState,
        new_positions: dict[int, LocationAddress],
    ) -> ConcreteState:
        new_layout = tuple(
            new_positions.get(i, loc) for i, loc in enumerate(state.layout)
        )
        new_move_count = list(state.move_count)
        for qid in new_positions.keys():
            new_move_count[qid] += 1

        return replace(state, layout=new_layout, move_count=tuple(new_move_count))

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state

        # invalid cz statement
        if len(controls) != len(targets):
            return AtomState.top()

        # since cz gates are symmetric swap controls and targets based on
        # word_id and site_id the idea being to minimize the directions
        # needed to rearrange qubits.
        new_positions: dict[int, LocationAddress] = {}
        start_word_id = self._word_balance(state, controls, targets)
        for c, t in zip(controls, targets):
            mover, dst_addr = self._pick_mover_and_location(state, start_word_id, c, t)
            new_positions[mover] = LocationAddress(
                word_id=dst_addr.word_id,
                site_id=dst_addr.site_id + 1,
            )

        return self._update_positions(state, new_positions)

    def sq_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        return state  # No movement for single-qubit gates


@dataclass()
class LogicalMoveScheduler(MoveSchedulerABC):
    arch_spec: ArchSpec = field(default_factory=get_arch_spec, init=False)

    def assert_valid_word_bus_move(
        self,
        direction: Direction,
        src_word: int,
        src_site: int,
        bus_id: int,
    ) -> WordLaneAddress:
        assert bus_id < len(
            self.arch_spec.word_buses
        ), f"Invalid bus id {bus_id} for word bus move"
        assert (
            src_word in self.arch_spec.word_buses[bus_id].src
        ), f"Invalid source word {src_word} for word bus move"
        assert (
            src_site in self.arch_spec.has_word_buses
        ), f"Invalid source site {src_site} for word bus move"

        return WordLaneAddress(
            direction,
            src_word,
            src_site,
            bus_id,
        )

    def assert_valid_site_bus_move(
        self,
        direction: Direction,
        src_word: int,
        src_site: int,
        bus_id: int,
    ) -> SiteLaneAddress:
        assert bus_id < len(
            self.arch_spec.site_buses
        ), f"Invalid bus id {bus_id} for site bus move"
        assert (
            src_site in self.arch_spec.site_buses[bus_id].src
        ), f"Invalid source site {src_site} for site bus move {bus_id}"
        assert (
            src_word in self.arch_spec.has_site_buses
        ), f"Invalid source word {src_word} for site bus move {bus_id}"

        return SiteLaneAddress(
            direction,
            src_word,
            src_site,
            bus_id,
        )

    def site_moves(
        self, diffs: list[tuple[LocationAddress, LocationAddress]], word_id: int
    ) -> list[tuple[LaneAddress, ...]]:
        start_site_ids = [before.site_id for before, _ in diffs]
        assert len(set(start_site_ids)) == len(
            start_site_ids
        ), "Start site ids must be unique"

        bus_moves = {}
        for before, end in diffs:
            bus_id = end.site_id // 2 - before.site_id // 2
            if bus_id < 0:
                bus_id += len(self.arch_spec.site_buses)

            bus_moves.setdefault(bus_id, []).append(
                self.assert_valid_site_bus_move(
                    Direction.FORWARD,
                    word_id,
                    before.site_id,
                    bus_id,
                )
            )

        return list(map(tuple, bus_moves.values()))

    def compute_moves(
        self, state_before: AtomState, state_after: AtomState
    ) -> list[tuple[LaneAddress, ...]]:
        if not (
            isinstance(state_before, ConcreteState)
            and isinstance(state_after, ConcreteState)
        ):
            return []

        diffs = [
            ele
            for ele in zip(state_before.layout, state_after.layout)
            if ele[0] != ele[1]
        ]

        groups: dict[tuple[int, int], list[tuple[LocationAddress, LocationAddress]]] = (
            {}
        )
        for src, dst in diffs:
            groups.setdefault((src.word_id, dst.word_id), []).append((src, dst))

        match (groups.get((1, 0), []), groups.get((0, 1), [])):
            case ([] as word_moves, []):
                word_start = 0
            case (list() as word_moves, []):
                word_start = 1
            case ([], list() as word_moves):
                word_start = 0
            case _:
                raise AssertionError(
                    "Cannot have both (0,1) and (1,0) moves in logical arch"
                )

        moves: list[tuple[LaneAddress, ...]] = self.site_moves(word_moves, word_start)
        if len(moves) > 0:
            moves.append(
                tuple(
                    self.assert_valid_word_bus_move(
                        Direction.FORWARD if word_start == 0 else Direction.BACKWARD,
                        0,
                        end.site_id,
                        0,
                    )
                    for _, end in word_moves
                )
            )

        moves.extend(self.site_moves(groups.get((0, 0), []), 0))
        moves.extend(self.site_moves(groups.get((1, 1), []), 1))

        return moves


@dataclass
class LogicalLayoutHeuristic(LayoutHeuristicABC):
    arch_spec: ArchSpec = field(default_factory=get_arch_spec, init=False)

    def score_parallelism(
        self, edges: dict[tuple[int, int], int], qubit_map: dict[int, LocationAddress]
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
    ) -> tuple[LocationAddress, ...]:

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
                LocationAddress(word_id, site_id)
                for word_id, word in enumerate(self.arch_spec.words)
                for site_id in range(len(word.sites))
                if site_id % 2 == 0
            ]
        )

        qubit_map: dict[int, LocationAddress] = {}
        layout_map: dict[LocationAddress, int] = {}
        for qubit in sorted(all_qubits):

            scores: dict[LocationAddress, int] = {}
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
