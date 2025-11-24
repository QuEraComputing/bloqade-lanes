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
    """A placement strategy that assumes a logical architecture."""

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

    def _pick_mover_and_location(
        self,
        state: ConcreteState,
        control: int,
        target: int,
    ):
        c_addr = state.layout[control]
        t_addr = state.layout[target]
        if c_addr.site_id <= t_addr.site_id:
            return control, LocationAddress(
                word_id=t_addr.word_id,
                site_id=t_addr.site_id + 1,
            )
        else:
            return target, LocationAddress(
                word_id=c_addr.word_id,
                site_id=c_addr.site_id + 1,
            )

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
        for c, t in zip(controls, targets):
            mover, dst_addr = self._pick_mover_and_location(state, c, t)
            new_positions[mover] = dst_addr

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
            assert bus_id >= 0, "Bus id should be non-negative"

            bus_moves.setdefault(bus_id, []).append(
                self.assert_valid_site_bus_move(
                    Direction.FORWARD,
                    word_id,
                    before.site_id,
                    bus_id,
                )
            )

        return list(map(tuple, bus_moves.values()))

    def moves_00(
        self, diffs: list[tuple[LocationAddress, LocationAddress]]
    ) -> list[tuple[LaneAddress, ...]]:
        if len(diffs) == 0:
            return []
        return self.site_moves(diffs, word_id=0)

    def moves_11(
        self, diffs: list[tuple[LocationAddress, LocationAddress]]
    ) -> list[tuple[LaneAddress, ...]]:
        if len(diffs) == 0:
            return []
        return self.site_moves(diffs, word_id=1)

    def moves_01(
        self, diffs: list[tuple[LocationAddress, LocationAddress]]
    ) -> list[tuple[LaneAddress, ...]]:
        if len(diffs) == 0:
            return []

        first_moves = self.site_moves(diffs, word_id=0)
        second_moves = tuple(
            self.assert_valid_word_bus_move(
                Direction.FORWARD,
                0,
                end.site_id,
                0,
            )
            for _, end in diffs
        )

        return first_moves + [second_moves]

    def moves_10(
        self, diffs: list[tuple[LocationAddress, LocationAddress]]
    ) -> list[tuple[LaneAddress, ...]]:
        if len(diffs) == 0:
            return []
        last_moves = self.site_moves(diffs, word_id=0)

        first_moves = [
            tuple(
                self.assert_valid_site_bus_move(
                    Direction.FORWARD,
                    1,
                    before.site_id,
                    0,
                )
                for before, _ in diffs
            ),
            tuple(
                self.assert_valid_word_bus_move(
                    Direction.BACKWARD,
                    0,
                    after.site_id,
                    0,
                )
                for _, after in diffs
            ),
            tuple(
                self.assert_valid_site_bus_move(
                    Direction.BACKWARD,
                    0,
                    before.site_id,
                    0,
                )
                for before, _ in diffs
            ),
        ]

        return first_moves + last_moves

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

        raise NotImplementedError


@dataclass
class LogicalLayoutHeuristic(LayoutHeuristicABC):
    arch_spec: ArchSpec = field(default=generate_arch(1))

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

        # sort locations from hightest site id and word id to lowest
        sites = range(0, 10, 2)[::-1]
        words = [1, 0]
        all_locations = [
            LocationAddress(word_id=w, site_id=s) for w in words for s in sites
        ]

        # sorted qubits from highest weighted degree to lowest
        sorted_qubits = sorted(
            all_qubits, key=lambda q: weighted_degrees[q], reverse=True
        )
        # assign highest weighted degree qubit to highest location
        location_map = {q: loc for q, loc in zip(sorted_qubits, all_locations)}
        return tuple(location_map[q] for q in all_qubits)
