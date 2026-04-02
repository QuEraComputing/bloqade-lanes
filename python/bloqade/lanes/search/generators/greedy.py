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
    and extracts the first lane (next hop). Groups compatible next-hop
    lanes by bus triplet and builds AOD-compatible rectangular grids
    via cluster growth.

    This provides a fast, deterministic baseline that doesn't require
    search tree exploration — useful both as a standalone compilation
    strategy and as a comparison point for benchmarking.
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
        1. Compute path lengths to determine processing order.
        2. Compute next-hop lanes longest-path-first with evolving occupied.
        3. Group next-hop lanes by (move_type, bus_id, direction).
        4. Build AOD-compatible rectangular grids per group via cluster growth.
        """
        occupied = node.occupied_locations | tree.blocked_locations

        unresolved, path_lengths = self._compute_path_lengths(node, occupied, tree)
        if not unresolved:
            return

        next_hops = self._compute_next_hops(unresolved, path_lengths, occupied, tree)
        if not next_hops:
            return

        groups = self._group_by_triplet(next_hops)

        for (mt, bid, d), entries in groups.items():
            ctx = BusContext.from_tree(tree, occupied, mt, bid, d)
            yield from ctx.build_aod_grids(entries)

    def _compute_path_lengths(
        self,
        node: ConfigurationNode,
        occupied: frozenset[LocationAddress],
        tree: ConfigurationTree,
    ) -> tuple[dict[int, LocationAddress], dict[int, int]]:
        """Identify unresolved qubits and compute their path lengths.

        Returns:
            unresolved: Mapping of qubit ID to current location for qubits
                that are not yet at their target and have a valid path.
            path_lengths: Mapping of qubit ID to shortest path length (in hops).
        """
        unresolved: dict[int, LocationAddress] = {}
        path_lengths: dict[int, int] = {}

        for qid, current in node.configuration.items():
            if qid not in self.target or current == self.target[qid]:
                continue
            path_occupied = occupied - {current}
            result = tree.path_finder.find_path(
                current,
                self.target[qid],
                occupied=path_occupied,
            )
            if result is None:
                continue
            path_lanes, _ = result
            if not path_lanes:
                continue
            unresolved[qid] = current
            path_lengths[qid] = len(path_lanes)

        return unresolved, path_lengths

    def _compute_next_hops(
        self,
        unresolved: dict[int, LocationAddress],
        path_lengths: dict[int, int],
        occupied: frozenset[LocationAddress],
        tree: ConfigurationTree,
    ) -> dict[int, LaneAddress]:
        """Compute the next-hop lane for each unresolved qubit.

        Processes qubits in longest-path-first order and updates an
        evolving occupied set after each, so later atoms route around
        earlier commitments.

        Returns:
            Mapping of qubit ID to its next-hop LaneAddress.
        """
        sorted_qids = sorted(unresolved, key=lambda q: path_lengths[q], reverse=True)
        evolving_occupied = set(occupied)
        next_hops: dict[int, LaneAddress] = {}

        for qid in sorted_qids:
            current = unresolved[qid]
            path_occupied = frozenset(evolving_occupied - {current})

            result = tree.path_finder.find_path(
                current,
                self.target[qid],
                occupied=path_occupied,
            )
            if result is None:
                continue

            path_lanes, _ = result
            if not path_lanes:
                continue

            lane = path_lanes[0]
            _, dst = tree.arch_spec.get_endpoints(lane)
            next_hops[qid] = lane

            evolving_occupied.discard(current)
            evolving_occupied.add(dst)

        return next_hops

    @staticmethod
    def _group_by_triplet(
        next_hops: dict[int, LaneAddress],
    ) -> dict[tuple[MoveType, int, Direction], dict[int, LaneAddress]]:
        """Group next-hop lanes by (move_type, bus_id, direction).

        Returns:
            Mapping of bus triplet to dict of qubit_id -> lane.
        """
        groups: dict[
            tuple[MoveType, int, Direction],
            dict[int, LaneAddress],
        ] = {}
        for qid, lane in next_hops.items():
            key = (lane.move_type, lane.bus_id, lane.direction)
            groups.setdefault(key, {})[qid] = lane
        return groups
