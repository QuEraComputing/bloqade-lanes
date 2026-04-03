"""Greedy shortest-path move generator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)
from bloqade.lanes.search.generators.aod_grouping import BusContext

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass(frozen=True)
class GreedyMoveGenerator:
    """Greedy shortest-path move generator.

    For each unresolved qubit, computes the shortest path to its target
    (without assuming other atoms have moved) and takes the first lane.
    Those first-step lanes are grouped by (move_type, bus_id, direction)
    and partitioned into AOD-compatible rectangular grids via BusContext.
    """

    target: dict[int, LocationAddress]
    """Mapping of qubit ID to target location."""

    def generate(
        self,
        node: ConfigurationNode,
        tree: ConfigurationTree,
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield candidate move sets.

        Steps:
        1. For each qubit, find shortest path to target (ignoring other atoms).
        2. Take the first lane from each path.
        3. Group by (move_type, bus_id, direction).
        4. Build AOD-compatible rectangular grids per group.
        """
        occupied = node.occupied_locations | tree.blocked_locations

        first_lanes: dict[int, LaneAddress] = {}
        for qid, current in node.configuration.items():
            target_loc = self.target.get(qid)
            if target_loc is None or current == target_loc:
                continue

            result = tree.path_finder.find_path(
                current, target_loc, occupied=occupied - {current}
            )
            if result is None:
                continue

            lanes, _ = result
            if lanes:
                first_lanes[qid] = lanes[0]

        if not first_lanes:
            return

        groups: dict[
            tuple[MoveType, int, Direction],
            dict[int, LaneAddress],
        ] = {}
        for qid, lane in first_lanes.items():
            key = (lane.move_type, lane.bus_id, lane.direction)
            groups.setdefault(key, {})[qid] = lane

        for (mt, bid, d), entries in groups.items():
            ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
            yield from ctx.build_aod_grids(entries)
