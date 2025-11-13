from dataclasses import dataclass, field, replace

from bloqade.lanes.analysis.placement import PlacementStrategyABC
from bloqade.lanes.analysis.placement.lattice import AtomState, ConcreteState
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes.rewrite.circuit2move import MoveSchedulerABC


@dataclass
class LogicalPlacementStrategy(PlacementStrategyABC):

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

        new_controls = []
        new_targets = []

        for control, target in zip(controls, targets):
            control_addr = state.layout[control]
            target_addr = state.layout[target]

            if control_addr.word_id > target_addr.word_id or (
                control_addr.word_id == target_addr.word_id
                and control_addr.site_id > target_addr.site_id
            ):
                new_controls.append(target)
                new_targets.append(control)
            else:
                new_controls.append(control)
                new_targets.append(target)

        total_control_moves = sum(state.move_count[control] for control in new_controls)

        total_target_moves = sum(state.move_count[target] for target in new_targets)

        move_controls = total_control_moves <= total_target_moves

        updates = {}
        for control, target in zip(new_controls, new_targets):
            control_addr = state.layout[control]
            target_addr = state.layout[target]

            # moving the non-static qubit to the location of the static qubit but
            # the next odd site over to avoid collisions

            if move_controls:
                new_addr = LocationAddress(
                    word_id=target_addr.word_id,
                    site_id=target_addr.site_id + 1,
                )
                assert (
                    new_addr not in state.occupied
                ), "Attempting to move control qubit to an occupied location"
                updates[control] = new_addr
            else:
                new_addr = LocationAddress(
                    word_id=control_addr.word_id,
                    site_id=control_addr.site_id - 1,
                )
                assert (
                    new_addr not in state.occupied
                ), "Attempting to move target qubit to an occupied location"
                updates[target] = new_addr

        new_layout = tuple(updates.get(i, addr) for i, addr in enumerate(state.layout))
        new_move_count = tuple(
            state.move_count[i] + (1 if i in updates else 0)
            for i in range(len(state.layout))
        )

        next_state = replace(state, layout=new_layout, move_count=new_move_count)
        return next_state

    def sq_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        return state  # No movement for single-qubit gates


@dataclass
class LogicalMoveScheduler(MoveSchedulerABC):
    path_finder: PathFinder = field(init=False)

    def __post_init__(self):
        self.path_finder = PathFinder(self.arch_spec)

    def get_direction(self, diff: int) -> Direction:
        if diff > 0:
            return Direction.FORWARD
        else:
            return Direction.BACKWARD

    def compute_moves(self, state_before: AtomState, state_after: AtomState):
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

        moves_dict = {}

        for src, dst in diffs:
            assert dst.word_id < 2, "Destination word id out of range for logical arch"
            assert src.word_id < 2, "Source word id out of range for logical arch"
            assert (
                src.site_id + dst.site_id
            ) % 2 != 1, "Invalid move between incompatible sites"

            word_diff = dst.word_id - src.word_id
            site_diff = dst.site_id - src.site_id
            site_bus_id = abs(site_diff)
            site_bus_direction = self.get_direction(site_diff)

            moves_dict.setdefault((site_bus_id, site_bus_direction), {}).setdefault(
                word_diff, []
            ).append(src)

        moves = []
        for (site_bus_id, site_bus_direction), word_dict in moves_dict.items():
            for word_diff, src_sites in word_dict.items():
                site_lanes = tuple(
                    SiteLaneAddress(
                        site_bus_direction,
                        src.word_id,
                        src.site_id,
                        site_bus_id,
                    )
                    for src in src_sites
                )
                moves.append(site_lanes)
                if word_diff != 0:
                    # Add word bus move
                    word_bus_direction = self.get_direction(word_diff)
                    word_lanes = tuple(
                        WordLaneAddress(
                            word_bus_direction,
                            src.word_id,
                            src.word_id + word_diff,
                            src.site_id,
                        )
                        for src in src_sites
                    )
                    moves.append(word_lanes)

        return moves
