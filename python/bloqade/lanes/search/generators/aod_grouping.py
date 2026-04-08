"""AOD-compatible rectangular grid construction.

Provides :class:`BusContext`, which encapsulates the bus-level state
and the divide-and-conquer algorithm for grouping lanes into valid
AOD rectangular grids.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from bloqade.lanes.layout import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)

if TYPE_CHECKING:
    from bloqade.lanes.bytecode._native import Zone as _RustZone
    from bloqade.lanes.layout.arch import ArchSpec
    from bloqade.lanes.search.tree import ConfigurationTree

# Type alias for a cluster: a pair of X and Y coordinate sets
# representing a rectangular region in physical space.
Cluster = tuple[set[float], set[float]]


@dataclass(frozen=True)
class BusContext:
    """Pre-computed context for AOD grid construction on a single bus.

    Encapsulates the position map, collision set, bus identity, and the
    divide-and-conquer algorithm for building AOD-compatible rectangular
    grids. Create one per (move_type, bus_id, direction) group via
    :meth:`from_tree`.

    The grid construction algorithm has two phases:

    1. **Greedy init** — O(n) sequential pass that forms initial clusters
       by greedily expanding rectangles one atom at a time.
    2. **Merge** — iterative passes that try to merge clusters. Clusters
       that don't participate in any merge are promoted to *solved* and
       removed from the active set, since merged clusters only grow and
       a cluster that can't merge now can never merge later.
    """

    move_type: MoveType
    bus_id: int
    direction: Direction
    arch_spec: ArchSpec
    pos_to_loc: dict[tuple[float, float], LocationAddress] = field(repr=False)
    collision_srcs: frozenset[LocationAddress] = field(repr=False)

    @classmethod
    def from_tree(
        cls,
        tree: ConfigurationTree,
        occupied: frozenset[LocationAddress],
        move_type: MoveType,
        bus_id: int,
        direction: Direction,
        zone_id: int | None = None,
    ) -> BusContext:
        """Build a BusContext from a ConfigurationTree and occupied set.

        Args:
            tree: The configuration tree.
            occupied: Set of occupied locations.
            move_type: Type of move (SITE or WORD).
            bus_id: Bus index.
            direction: Move direction.
            zone_id: If provided, restrict sources to this zone only.
        """
        arch_spec = tree.arch_spec

        # Aggregate sources across zones (or a single zone if specified)
        src_locs: list[LocationAddress] = []
        zone_iter: list[tuple[int, _RustZone]] = []
        if zone_id is not None:
            zone_iter = [(zone_id, arch_spec.zones[zone_id])]
        else:
            zone_iter = list(enumerate(arch_spec.zones))

        for zid, zone in zone_iter:
            if move_type == MoveType.SITE:
                if bus_id < len(zone.site_buses):
                    bus = zone.site_buses[bus_id]
                    src_locs.extend(
                        LocationAddress(w, s, zid)
                        for w in zone.words_with_site_buses
                        for s in bus.src
                    )
            else:
                if bus_id < len(zone.word_buses):
                    bus = zone.word_buses[bus_id]
                    src_locs.extend(
                        LocationAddress(w, s, zid)
                        for w in bus.src
                        for s in zone.sites_with_word_buses
                    )

        pos_to_loc: dict[tuple[float, float], LocationAddress] = {}
        for loc in src_locs:
            pos = arch_spec.get_position(loc)
            pos_to_loc[pos] = loc

        # Bus src and dst are disjoint, so follow-moves cannot occur.
        collision: set[LocationAddress] = set()
        for loc in src_locs:
            if loc in occupied:
                lane = LaneAddress(
                    move_type, loc.word_id, loc.site_id, bus_id, direction, loc.zone_id
                )
                _, dst = arch_spec.get_endpoints(lane)
                if dst in occupied:
                    collision.add(loc)

        return cls(
            move_type=move_type,
            bus_id=bus_id,
            direction=direction,
            arch_spec=arch_spec,
            pos_to_loc=pos_to_loc,
            collision_srcs=frozenset(collision),
        )

    # --- Primitives ---

    def is_valid_rect(self, xs: set[float], ys: set[float]) -> bool:
        """Check if every position in the X * Y rectangle is a valid bus
        source with no collision."""
        for x in xs:
            for y in ys:
                loc = self.pos_to_loc.get((x, y))
                if loc is None or loc in self.collision_srcs:
                    return False
        return True

    def rect_to_lanes(self, xs: set[float], ys: set[float]) -> frozenset[LaneAddress]:
        """Convert an X * Y rectangle into the corresponding lane set."""
        lanes: list[LaneAddress] = []
        for x in xs:
            for y in ys:
                loc = self.pos_to_loc.get((x, y))
                if loc is not None:
                    lanes.append(
                        LaneAddress(
                            self.move_type,
                            loc.word_id,
                            loc.site_id,
                            self.bus_id,
                            self.direction,
                            loc.zone_id,
                        )
                    )
        return frozenset(lanes)

    def lane_position(self, lane: LaneAddress) -> tuple[float, float]:
        """Get the physical (x, y) position of a lane's source."""
        src, _ = self.arch_spec.get_endpoints(lane)
        return self.arch_spec.get_position(src)

    # --- Grid construction ---

    def build_aod_grids(
        self,
        entries: dict[int, LaneAddress],
    ) -> list[frozenset[LaneAddress]]:
        """Build AOD-compatible rectangular grids from desired next-hop lanes.

        Two phases:
        1. Greedy sequential pass to form initial clusters (O(n)).
        2. Merge pass to combine clusters that the greedy ordering missed.
           Clusters that cannot merge are marked solved and removed from
           the active set so they don't slow down later rounds.
        """
        clusters = self.greedy_init(entries)
        solved = self.merge_clusters(clusters)

        return [moveset for xs, ys in solved if (moveset := self.rect_to_lanes(xs, ys))]

    def greedy_init(
        self,
        entries: dict[int, LaneAddress],
    ) -> list[Cluster]:
        """Form initial clusters via greedy sequential expansion.

        Processes atoms in order and greedily expands a rectangle. Atoms
        that don't fit start a new cluster in the next pass. Repeats
        until all atoms are assigned or individually invalid.
        """
        clusters: list[Cluster] = []
        remaining = dict(entries)

        while remaining:
            xs: set[float] = set()
            ys: set[float] = set()
            leftover: dict[int, LaneAddress] = {}

            for qid, lane in remaining.items():
                x, y = self.lane_position(lane)

                if x in xs and y in ys:
                    continue

                new_xs = xs | {x}
                new_ys = ys | {y}

                if self.is_valid_rect(new_xs, new_ys):
                    xs = new_xs
                    ys = new_ys
                else:
                    leftover[qid] = lane

            if xs and ys:
                clusters.append((xs, ys))
            else:
                break

            remaining = leftover

        return clusters

    def merge_clusters(
        self,
        clusters: list[Cluster],
    ) -> list[Cluster]:
        """Merge clusters until no more merges are possible.

        Each pass lets every active cluster try to absorb compatible
        neighbours. Clusters that don't participate in any merge are
        promoted to solved and removed from the active set — merged
        clusters only grow, so a cluster that can't merge now will
        never be able to merge later.
        """
        solved: list[Cluster] = []

        while len(clusters) > 1:
            n = len(clusters)
            consumed: set[int] = set()
            merged_flags = [False] * n

            for i in range(n):
                if i in consumed:
                    continue
                for j in range(i + 1, n):
                    if j in consumed:
                        continue
                    merged_xs = clusters[i][0] | clusters[j][0]
                    merged_ys = clusters[i][1] | clusters[j][1]
                    if self.is_valid_rect(merged_xs, merged_ys):
                        clusters[i] = (merged_xs, merged_ys)
                        consumed.add(j)
                        merged_flags[i] = True
                        merged_flags[j] = True

            if not any(merged_flags):
                break

            active: list[Cluster] = []
            for i in range(n):
                if i in consumed:
                    continue
                if merged_flags[i]:
                    active.append(clusters[i])
                else:
                    solved.append(clusters[i])
            clusters = active

        solved.extend(clusters)
        return solved
