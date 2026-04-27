"""Exhaustive AOD grid enumeration move generator."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import TYPE_CHECKING, Iterator

from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)

if TYPE_CHECKING:
    from bloqade.lanes.search.configuration import ConfigurationNode
    from bloqade.lanes.search.tree import ConfigurationTree


@dataclass(frozen=True)
class ExhaustiveMoveGenerator:
    """Enumerates all valid AOD grids from the configuration.

    For each (move_type, bus_id, direction) group, finds all source
    positions, enumerates grid subsets within AOD capacity,
    and yields the full grid of lane addresses.

    Pre-filters grids where an occupied source has an occupied
    destination (collision) as an optimization.

    NOTE for Rust port: replace itertools.combinations with Gosper's
    hack for bitmask enumeration of exactly-k-set-bits subsets. This
    avoids iterating all 2^n masks when capacity is small. See #298.
    """

    max_x_capacity: int | None = None
    """Maximum number of unique X positions the AOD can address."""

    max_y_capacity: int | None = None
    """Maximum number of unique Y positions the AOD can address."""

    def generate(
        self,
        node: ConfigurationNode,
        tree: ConfigurationTree,
    ) -> Iterator[frozenset[LaneAddress]]:
        occupied = node.occupied_locations | tree.blocked_locations

        # Enumerate site buses per zone
        for zone_id, zone in enumerate(tree.arch_spec.zones):
            for bus_id, bus in enumerate(zone.site_buses):
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    bus_word_ids = zone.words_with_site_buses
                    src_locs = [
                        LocationAddress(w, s, zone_id)
                        for w in bus_word_ids
                        for s in bus.src
                    ]
                    yield from self._rectangles_to_move_sets(
                        src_locs,
                        occupied,
                        MoveType.SITE,
                        bus_id,
                        direction,
                        tree,
                        zone_id,
                    )

            # Enumerate word buses per zone
            for bus_id, bus in enumerate(zone.word_buses):
                for direction in (Direction.FORWARD, Direction.BACKWARD):
                    src_locs = [
                        LocationAddress(w, s, zone_id)
                        for w in bus.src
                        for s in zone.sites_with_word_buses
                    ]
                    yield from self._rectangles_to_move_sets(
                        src_locs,
                        occupied,
                        MoveType.WORD,
                        bus_id,
                        direction,
                        tree,
                        zone_id,
                    )

    def _rectangles_to_move_sets(
        self,
        src_locs: list[LocationAddress],
        occupied: frozenset[LocationAddress],
        move_type: MoveType,
        bus_id: int,
        direction: Direction,
        tree: ConfigurationTree,
        zone_id: int = 0,
    ) -> Iterator[frozenset[LaneAddress]]:
        if not src_locs:
            return

        # Build position lookups
        pos_to_loc: dict[tuple[float, float], LocationAddress] = {}
        unique_xs: set[float] = set()
        unique_ys: set[float] = set()
        for loc in src_locs:
            pos = tree.arch_spec.get_position(loc)
            assert (
                pos is not None
            ), f"location {loc!r} has no position in this architecture"
            x, y = pos
            pos_to_loc[(x, y)] = loc
            unique_xs.add(x)
            unique_ys.add(y)

        sorted_xs = sorted(unique_xs)
        sorted_ys = sorted(unique_ys)

        # Pre-build lane addresses and cache invalid sources (destination occupied).
        # This pre-filter is safe because bus src and dst are disjoint sets —
        # a destination cannot also be a source within the same bus, so "follow"
        # moves (atom A moves into location vacated by atom B) cannot occur.
        loc_to_lane: dict[LocationAddress, LaneAddress] = {}
        invalid_locs: set[LocationAddress] = set()
        for loc in src_locs:
            lane = LaneAddress(
                move_type, loc.word_id, loc.site_id, bus_id, direction, zone_id
            )
            loc_to_lane[loc] = lane

            if loc in occupied:
                endpoints = tree.arch_spec.get_endpoints(lane)
                assert (
                    endpoints is not None
                ), f"lane {lane!r} has no endpoints in this architecture"
                _, dst = endpoints
                if dst in occupied:
                    invalid_locs.add(loc)

        # Enumerate all X × Y subset combinations within capacity
        max_nx = (
            self.max_x_capacity if self.max_x_capacity is not None else len(sorted_xs)
        )
        max_ny = (
            self.max_y_capacity if self.max_y_capacity is not None else len(sorted_ys)
        )

        for nx, ny in product(range(1, max_nx + 1), range(1, max_ny + 1)):
            yield from self._enumerate_xy_combinations(
                sorted_xs,
                nx,
                sorted_ys,
                ny,
                pos_to_loc,
                loc_to_lane,
                invalid_locs,
                occupied,
            )

    def _enumerate_xy_combinations(
        self,
        sorted_xs: list[float],
        max_xs: int,
        sorted_ys: list[float],
        max_ys: int,
        pos_to_loc: dict[tuple[float, float], LocationAddress],
        loc_to_lane: dict[LocationAddress, LaneAddress],
        invalid_locs: set[LocationAddress],
        occupied: frozenset[LocationAddress],
    ) -> Iterator[frozenset[LaneAddress]]:
        """Yield valid move sets for all nx × ny grids.

        NOTE for Rust port: replace itertools.combinations with Gosper's
        hack for bitmask enumeration of exactly-k-set-bits subsets. This
        avoids iterating all 2^n masks when capacity is small. See #298.
        """
        # Materialize only y combinations to avoid product() caching both
        # potentially large combination iterators.
        for x_subset in combinations(sorted_xs, max_xs):
            for y_subset in combinations(sorted_ys, max_ys):
                lanes: list[LaneAddress] = []
                valid = True
                has_atom = False

                for loc in map(pos_to_loc.get, product(x_subset, y_subset)):
                    if loc is not None and loc not in invalid_locs:
                        lanes.append(loc_to_lane[loc])
                        has_atom = has_atom or loc in occupied
                    else:
                        valid = False

                if valid and has_atom:
                    yield frozenset(lanes)
