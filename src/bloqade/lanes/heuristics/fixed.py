from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Any, Generator

import rustworkx as rx

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

    def __init__(self):
        super().__init__(generate_arch(1))

    def get_direction(self, diff: int) -> Direction:
        if diff > 0:
            return Direction.FORWARD
        else:
            return Direction.BACKWARD

    def get_site_y(self, location: LocationAddress) -> int:
        return location.site_id // 2

    def get_site_bus_id(self, src: LocationAddress, dst: LocationAddress):
        diff_y = self.get_site_y(dst) - self.get_site_y(src)
        if diff_y >= 0:
            return diff_y, Direction.FORWARD
        else:
            return -diff_y, Direction.BACKWARD

    def _yield_site_moves(
        self,
        site_dict: dict[
            tuple[int, Direction], list[tuple[LocationAddress, LocationAddress]]
        ],
    ) -> Generator[tuple[LaneAddress, ...], Any, None]:
        for (site_bus_id, site_bus_direction), sites in site_dict.items():
            yield tuple(
                SiteLaneAddress(
                    site_bus_direction,
                    src.word_id,
                    src.site_id,
                    site_bus_id,
                )
                for src, _ in sites
            )

    def compute_moves(self, state_before: AtomState, state_after: AtomState):
        if not (
            isinstance(state_before, ConcreteState)
            and isinstance(state_after, ConcreteState)
        ):
            return []

        diffs = (
            ele
            for ele in zip(state_before.layout, state_after.layout)
            if ele[0] != ele[1]
        )

        diff_moves_dict: dict[
            tuple[int, int],
            dict[tuple[int, Direction], list[tuple[LocationAddress, LocationAddress]]],
        ] = {}

        same_moves_dict = dict(diff_moves_dict)
        for src, dst in diffs:
            site_bus_id, site_bus_direction = self.get_site_bus_id(src, dst)
            if src.word_id != dst.word_id:
                diff_moves_dict.setdefault((src.word_id, dst.word_id), {}).setdefault(
                    (site_bus_id, site_bus_direction), []
                ).append((src, dst))
            else:
                same_moves_dict.setdefault((src.word_id, src.word_id), {}).setdefault(
                    (site_bus_id, site_bus_direction), []
                ).append((src, dst))

        assert (
            len(diff_moves_dict) <= 1
        ), "Multiple word bus moves detected, which should not happen in logical architecture"

        moves: list[tuple[LaneAddress, ...]] = []
        if len(diff_moves_dict) > 0:
            ((src_word_id, dst_word_id),) = diff_moves_dict.keys()
            diff_sites_dict = diff_moves_dict[(src_word_id, dst_word_id)]

            moves.extend(self._yield_site_moves(diff_sites_dict))
            direction = self.get_direction(dst_word_id - src_word_id)
            moves.append(
                tuple(
                    WordLaneAddress(direction, src_word_id, dst.site_id + 1, 0)
                    for sites in diff_sites_dict.values()
                    for _, dst in sites
                )
            )

        for sites_dict in same_moves_dict.values():
            moves.extend(self._yield_site_moves(sites_dict))
            # no word bus move needed here

        return moves


@dataclass
class LogicalLayoutHeuristic(LayoutHeuristicABC):
    arch_spec: ArchSpec = field(default=generate_arch(1))

    def compute_layout(
        self,
        stages: list[tuple[tuple[int, int], ...]],
    ) -> tuple[LocationAddress, ...]:
        graph = rx.PyGraph()

        all_global_addresses = sorted(
            set(node for edge in chain.from_iterable(stages) for node in edge)
        )
        assert all_global_addresses == list(
            range(len(all_global_addresses))
        ), f"{all_global_addresses} does not form a contiguous set starting from 0"

        for global_addr in all_global_addresses:
            graph.add_node(global_addr)

        edges = {}

        for control, target in chain.from_iterable(stages):
            edge_weight = edges.get((control, target), 0)
            edges[(control, target)] = edge_weight + 1
            edges[(target, control)] = edge_weight + 1

        for (src, dst), weight in edges.items():
            graph.add_edge(src, dst, weight)

        left_sides = list()
        right_sides = list()
        current_side, other_side = left_sides, right_sides
        stack = [0]
        visited = set()
        while stack:
            addr = stack.pop()
            current_side.append(addr)
            visited.add(addr)

            def get_weight(n):
                data: int = graph.get_edge_data(addr, n)
                if data is None:
                    return 0
                return data

            neighbors = sorted(graph.neighbors(addr), key=get_weight, reverse=False)

            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
                    break

            # swap sides
            current_side, other_side = other_side, current_side

        layout_map = {}

        for i, addr in enumerate(left_sides):
            layout_map[LocationAddress(word_id=0, site_id=i * 2)] = addr

        for i, addr in enumerate(right_sides):
            layout_map[LocationAddress(word_id=1, site_id=i * 2)] = addr

        # invert layout
        final_layout = list(layout_map.keys())
        final_layout.sort(key=lambda x: layout_map[x])

        return tuple(final_layout)
