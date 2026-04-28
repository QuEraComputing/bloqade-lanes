from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import rustworkx as nx

from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
    SiteLaneAddress,
    WordLaneAddress,
)
from bloqade.lanes.layout.move_metric import MoveMetricCalculator


@dataclass(frozen=True)
class PathFinder:
    spec: ArchSpec
    metrics: MoveMetricCalculator = field(init=False)
    site_graph: nx.PyDiGraph = field(init=False, default_factory=nx.PyDiGraph)
    """Graph representing all sites and edges as lanes."""
    physical_addresses: list[LocationAddress] = field(init=False, default_factory=list)
    """Map from graph node index to (zone_id, word_id, site_id) tuple."""
    physical_address_map: dict[LocationAddress, int] = field(
        init=False, default_factory=dict
    )
    """Map from (zone_id, word_id, site_id) tuple to graph node index."""
    end_points_cache: dict[LaneAddress, tuple[LocationAddress, LocationAddress]] = (
        field(init=False, default_factory=dict)
    )

    def __post_init__(self):
        object.__setattr__(self, "metrics", MoveMetricCalculator(arch_spec=self.spec))

        # Build physical addresses for all zone/word/site combos.
        # Use zone 0 as default for constructing the graph.
        for zone_id, zone in enumerate(self.spec._inner.zones):
            word_ids = range(len(self.spec.words))
            site_ids = range(self.spec.sites_per_word)
            for word_id, site_id in product(word_ids, site_ids):
                addr = LocationAddress(word_id, site_id, zone_id)
                if addr not in self.physical_address_map:
                    idx = len(self.physical_addresses)
                    self.physical_addresses.append(addr)
                    self.physical_address_map[addr] = idx

        self.site_graph.add_nodes_from(range(len(self.physical_addresses)))

        for zone_id, zone in enumerate(self.spec._inner.zones):
            for bus_id, bus in enumerate(zone.site_buses):
                bus_word_ids = zone.words_with_site_buses
                for word_id in bus_word_ids:
                    for src, dst in zip(bus.src, bus.dst):
                        src_site = LocationAddress(word_id, src, zone_id)
                        dst_site = LocationAddress(word_id, dst, zone_id)
                        lane_addr = SiteLaneAddress(
                            word_id,
                            src,
                            bus_id,
                            Direction.FORWARD,
                            zone_id,
                        )
                        src_idx = self.physical_address_map[src_site]
                        dst_idx = self.physical_address_map[dst_site]
                        self.site_graph.add_edge(src_idx, dst_idx, lane_addr)
                        self.site_graph.add_edge(
                            dst_idx,
                            src_idx,
                            rev_lane_addr := lane_addr.reverse(),
                        )
                        self.end_points_cache[lane_addr] = (src_site, dst_site)
                        self.end_points_cache[rev_lane_addr] = (dst_site, src_site)

            for bus_id, bus in enumerate(zone.word_buses):
                for src_word, dst_word in zip(bus.src, bus.dst):
                    for site in zone.sites_with_word_buses:
                        src_site = LocationAddress(src_word, site, zone_id)
                        dst_site = LocationAddress(dst_word, site, zone_id)
                        lane_addr = WordLaneAddress(
                            src_word,
                            site,
                            bus_id,
                            Direction.FORWARD,
                            zone_id,
                        )
                        src_idx = self.physical_address_map[src_site]
                        dst_idx = self.physical_address_map[dst_site]
                        self.site_graph.add_edge(src_idx, dst_idx, lane_addr)
                        self.site_graph.add_edge(
                            dst_idx,
                            src_idx,
                            rev_lane_addr := lane_addr.reverse(),
                        )
                        self.end_points_cache[lane_addr] = (src_site, dst_site)
                        self.end_points_cache[rev_lane_addr] = (dst_site, src_site)

        # Zone buses: inter-zone word movement.
        for bus_id, zb in enumerate(self.spec.zone_buses):
            for (src_zone, src_word), (dst_zone, dst_word) in zip(zb.src, zb.dst):
                for site in range(self.spec.sites_per_word):
                    src_site = LocationAddress(src_word, site, src_zone)
                    dst_site = LocationAddress(dst_word, site, dst_zone)
                    if (
                        src_site not in self.physical_address_map
                        or dst_site not in self.physical_address_map
                    ):
                        continue
                    lane_addr = LaneAddress(
                        MoveType.ZONE,
                        src_word,
                        site,
                        bus_id,
                        Direction.FORWARD,
                        src_zone,
                    )
                    src_idx = self.physical_address_map[src_site]
                    dst_idx = self.physical_address_map[dst_site]
                    self.site_graph.add_edge(src_idx, dst_idx, lane_addr)
                    self.site_graph.add_edge(
                        dst_idx,
                        src_idx,
                        rev_lane_addr := lane_addr.reverse(),
                    )
                    self.end_points_cache[lane_addr] = (src_site, dst_site)
                    self.end_points_cache[rev_lane_addr] = (dst_site, src_site)

    def extract_lanes_from_path(self, path: list[int]) -> tuple[LaneAddress, ...]:
        """Given a path as node indices, extract the lane addresses."""
        if len(path) < 2:
            raise ValueError("Path must have at least two nodes to extract lanes.")
        lanes = []
        for start_node, end_node in zip(path, path[1:]):
            lane = self.site_graph.get_edge_data(start_node, end_node)
            if lane is None:
                raise ValueError(
                    f"No lane exists between nodes {start_node} and {end_node}."
                )
            lanes.append(lane)
        return tuple(lanes)

    def extract_locations_from_path(
        self, path: list[int]
    ) -> tuple[LocationAddress, ...]:
        """Given a path as node indices, extract the location addresses."""
        return tuple(self.physical_addresses[ele] for ele in path)

    def get_lane(
        self, start: LocationAddress, end: LocationAddress
    ) -> LaneAddress | None:
        """Get the LaneAddress connecting two LocationAddress sites."""
        start_node = self.physical_address_map.get(start)
        end_node = self.physical_address_map.get(end)
        if start_node is None or end_node is None:
            return None
        edge_data = self.site_graph.get_edge_data(start_node, end_node)
        if edge_data is None:
            return None
        return edge_data

    def get_endpoints(self, lane: LaneAddress):
        """Get the start and end LocationAddress for a given LaneAddress."""
        if lane in self.end_points_cache:
            return self.end_points_cache[lane]
        return None, None

    def find_path(
        self,
        start: LocationAddress,
        end: LocationAddress,
        occupied: frozenset[LocationAddress] = frozenset(),
        path_heuristic: Callable[
            [tuple[LaneAddress, ...], tuple[LocationAddress, ...]], float
        ] = lambda _, __: 0.0,
        edge_weight: Callable[[LaneAddress], float] | None = None,
    ) -> tuple[tuple[LaneAddress, ...], tuple[LocationAddress, ...]] | None:
        """Find a weighted shortest path from start to end.

        Args:
            start: The starting location.
            end: The ending location.
            occupied: Locations to exclude when searching for a path. If this excludes
                `start` or `end`, no path is returned.
            path_heuristic: A tie-breaker over candidate shortest paths, evaluated on
                the candidate location sequence.
            edge_weight: Optional edge weight function used for shortest-path costs.
                Defaults to `Metrics.get_lane_duration_us` when not provided.

        Returns:
            A tuple containing:
                - The selected path as `LaneAddress` values.
                - The same path as `LocationAddress` values (including start and end).
            Returns `None` when no valid path exists.
        """
        start_node = self.physical_address_map.get(start)
        end_node = self.physical_address_map.get(end)
        if start_node is None or end_node is None:
            return None
        if start == end:
            return (), (start,)

        available_nodes = [
            node
            for node, address in enumerate(self.physical_addresses)
            if address not in occupied
        ]
        subgraph, node_map = self.site_graph.subgraph_with_nodemap(available_nodes)
        original_to_subgraph = {original: sub for sub, original in node_map.items()}

        if (
            start_node not in original_to_subgraph
            or end_node not in original_to_subgraph
        ):
            return None

        if edge_weight is None:
            resolved_edge_weight = self.metrics.get_lane_duration_us
        else:
            resolved_edge_weight = edge_weight

        try:
            path_nodes = nx.all_shortest_paths(
                subgraph,
                original_to_subgraph[start_node],
                original_to_subgraph[end_node],
                weight_fn=resolved_edge_weight,
            )
        except nx.NoPathFound:
            return None
        original_paths = ([node_map[node] for node in path] for path in path_nodes)
        paths = [
            (
                self.extract_lanes_from_path(original_path),
                self.extract_locations_from_path(original_path),
            )
            for original_path in original_paths
            if len(original_path) >= 2
        ]
        if len(paths) == 0:
            return None
        return min(paths, key=lambda p: path_heuristic(*p))
