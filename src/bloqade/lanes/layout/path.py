from dataclasses import dataclass, field
from itertools import product, starmap
from typing import Callable

import rustworkx as nx

from .arch import ArchSpec
from .encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
)


@dataclass(frozen=True)
class PathFinder:
    spec: ArchSpec
    site_graph: nx.PyDiGraph = field(init=False, default_factory=nx.PyDiGraph)
    """Graph representing all sites and edges as lanes."""
    physical_addresses: list[LocationAddress] = field(init=False, default_factory=list)
    """Map from graph node index to (word_id, site_id) tuple."""
    physical_address_map: dict[LocationAddress, int] = field(
        init=False, default_factory=dict
    )
    """Map from (word_id, site_id) tuple to graph node index."""
    end_points_cache: dict[LaneAddress, tuple[LocationAddress, LocationAddress]] = (
        field(init=False, default_factory=dict)
    )

    def __post_init__(self):
        word_ids = range(len(self.spec.words))
        site_ids = range(len(self.spec.words[0].sites))
        self.physical_addresses.extend(
            starmap(LocationAddress, product(word_ids, site_ids))
        )
        self.physical_address_map.update(
            {site: i for i, site in enumerate(self.physical_addresses)}
        )
        self.site_graph.add_nodes_from(range(len(self.physical_addresses)))

        for bus_id, bus in enumerate(self.spec.site_buses):
            for word_id in self.spec.has_site_buses:
                for src, dst in zip(bus.src, bus.dst):
                    src_site = LocationAddress(word_id, src)
                    dst_site = LocationAddress(word_id, dst)
                    lane_addr = SiteLaneAddress(Direction.FORWARD, word_id, src, bus_id)
                    self.site_graph.add_edge(
                        self.physical_address_map[src_site],
                        self.physical_address_map[dst_site],
                        lane_addr,
                    )
                    self.site_graph.add_edge(
                        self.physical_address_map[dst_site],
                        self.physical_address_map[src_site],
                        rev_lane_addr := lane_addr.reverse(),
                    )
                    self.end_points_cache[lane_addr] = (src_site, dst_site)
                    self.end_points_cache[rev_lane_addr] = (dst_site, src_site)

        for bus_id, bus in enumerate(self.spec.word_buses):
            for src_word, dst_word in zip(bus.src, bus.dst):
                print(src_word, dst_word, bus_id)
                for site in self.spec.has_word_buses:
                    src_site = LocationAddress(src_word, site)
                    dst_site = LocationAddress(dst_word, site)
                    lane_addr = WordLaneAddress(
                        Direction.FORWARD, src_word, site, bus_id
                    )
                    self.site_graph.add_edge(
                        self.physical_address_map[src_site],
                        self.physical_address_map[dst_site],
                        lane_addr,
                    )
                    self.site_graph.add_edge(
                        self.physical_address_map[dst_site],
                        self.physical_address_map[src_site],
                        rev_lane_addr := lane_addr.reverse(),
                    )
                    self.end_points_cache[lane_addr] = (src_site, dst_site)
                    self.end_points_cache[rev_lane_addr] = (dst_site, src_site)

    def extract_lanes_from_path(self, path: list[int]):
        """Given a path as a list of node indices, extract the lane addresses."""
        if len(path) < 2:
            raise ValueError("Path must have at least two nodes to extract lanes.")
        lanes: list[LaneAddress] = []
        for src, dst in zip(path[:-1], path[1:]):
            if not self.site_graph.has_edge(src, dst):
                return None

            lane: LaneAddress = self.site_graph.get_edge_data(src, dst)
            lanes.append(lane)

        print(f"Extracted lanes: {lanes}")
        return tuple(lanes)

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
        path_heuristic: Callable[[tuple[LaneAddress, ...]], float] = lambda _: 0.0,
    ) -> tuple[tuple[LaneAddress, ...] | None, frozenset[LocationAddress]]:
        """Find a path from start to end avoiding occupied sites.

        Args:
            start: The starting site as a PhysicalAddress.
            end: The ending site as a PhysicalAddress.
            occupied: A frozenset of sites PhysicalAddress that are occupied.
            path_heuristic: A heuristic function that takes a list of sites and returns a float
                cost for the path. Used to select among multiple shortest paths.

        Returns:
            A tuple containing:
                - A list of lane addresses representing the path, or None if no path found.
                - An updated frozenset of occupied sites including those used in the path.
        Raises:
            ValueError: If start or end sites are already occupied.
        """
        start_node = self.physical_address_map[start]
        end_node = self.physical_address_map[end]

        path_nodes = nx.all_simple_paths(self.site_graph, start_node, end_node)

        def filter_occupied(path: list[int] | None):
            if path is None:
                return False
            return all(
                self.physical_addresses[node] not in occupied for node in path[1:]
            )

        valid_paths = list(filter(filter_occupied, path_nodes))
        paths = list(filter(None, map(self.extract_lanes_from_path, valid_paths)))

        if len(paths) == 0:
            return None, occupied

        lanes = min(paths, key=lambda p: len(p) + path_heuristic(p))
        new_occupied = set(occupied)
        for lane in lanes:
            src, dst = self.get_endpoints(lane)
            if src is None or dst is None:
                raise RuntimeError(f"Endpoints not found for lane {lane}")

            new_occupied.add(src)
            new_occupied.add(dst)

        return lanes, frozenset(new_occupied)
