from dataclasses import dataclass, field, replace
from itertools import chain

from kirin import interp

from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.analysis.placement.lattice import AtomState, ConcreteState
from bloqade.lanes.gemini import generate_arch
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)
from bloqade.lanes.layout.path import PathFinder
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


@dataclass(init=False)
class LogicalMoveScheduler(MoveSchedulerABC):
    path_finder: PathFinder

    def __init__(self):
        super().__init__(generate_arch(1))
        self.path_finder = PathFinder(self.arch_spec)

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
            case (word_moves, []) if len(word_moves) >= 0:
                word_start = 1
            case ([], word_moves) if len(word_moves) > 0:
                word_start = 0
            case _:
                raise AssertionError(
                    "Cannot have both (0,1) and (1,0) moves in logical arch"
                )

        moves: list[tuple[LaneAddress, ...]] = self.site_moves(word_moves, word_start)
        if len(moves) > 0:
            direction = Direction.FORWARD if word_start == 0 else Direction.BACKWARD
            moves.append(
                tuple(
                    self.assert_valid_word_bus_move(
                        direction,
                        0,
                        end.site_id,
                        0,
                    )
                    for _, end in diffs
                )
            )

        moves.extend(self.site_moves(groups.get((0, 0), []), 0))
        moves.extend(self.site_moves(groups.get((1, 1), []), 1))

        return moves


@dataclass
class LogicalLayoutHeuristic(LayoutHeuristicABC):
    arch_spec: ArchSpec = field(default=generate_arch(1))

    def layout_from_weights(
        self,
        edges: dict[tuple[int, int], int],
        weighted_degrees: dict[int, int],
    ) -> tuple[LocationAddress, ...]:
        left_atoms = []
        right_atoms = []

        unassigned = sorted(
            weighted_degrees.keys(),
            key=lambda x: (weighted_degrees[x], x),
        )

        def get_weight(addr: int, curr_atoms: list[int], other_atoms: list[int]):

            weight = 0
            for same in curr_atoms:
                edge = (min(addr, same), max(addr, same))
                weight += 2 * edges.get(edge, 0)

            for other in other_atoms:
                edge = (min(addr, other), max(addr, other))
                weight += edges.get(edge, 0)

            return weight

        while unassigned:
            addr = unassigned.pop()

            left_weight = get_weight(addr, left_atoms, right_atoms)
            right_weight = get_weight(addr, right_atoms, left_atoms)

            if right_weight < left_weight:
                best_list, not_best_list = left_atoms, right_atoms
            else:
                best_list, not_best_list = right_atoms, left_atoms

            if len(best_list) < 5:
                best_list.append(addr)
            else:
                not_best_list.append(addr)

        layout_map = {}
        for i, addr in enumerate(left_atoms):
            layout_map[
                LocationAddress(
                    word_id=0,
                    site_id=i * 2,
                )
            ] = addr

        for i, addr in enumerate(right_atoms):
            layout_map[
                LocationAddress(
                    word_id=1,
                    site_id=i * 2,
                )
            ] = addr

        # invert layout
        final_layout = list(layout_map.keys())
        final_layout.sort(key=lambda x: layout_map[x])

        return tuple(final_layout)

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

        weighted_degrees = {i: 0 for i in all_qubits}

        for (n, m), weight in edges.items():
            weighted_degrees[n] += weight
            weighted_degrees[m] += weight

        return self.layout_from_weights(edges, weighted_degrees)
