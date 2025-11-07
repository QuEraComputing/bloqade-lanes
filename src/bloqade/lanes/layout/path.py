from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import rustworkx as nx

from .arch import ArchSpec
from .encoding import Direction, InterMove, IntraMove, MoveType


@dataclass(frozen=True)
class PathFinder:
    spec: ArchSpec
    site_graph: nx.PyGraph = field(init=False, default_factory=nx.PyGraph)
    """Graph representing all sites and edges as lanes."""
    physical_addresses: list[tuple[int, int]] = field(init=False, default_factory=list)
    """Map from graph node index to (word_id, site_id) tuple."""
    physical_address_map: dict[tuple[int, int], int] = field(
        init=False, default_factory=dict
    )
    """Map from (word_id, site_id) tuple to graph node index."""

    def __post_init__(self):
        self.physical_address_map.update(
            {site: i for i, site in enumerate(self.physical_addresses)}
        )

        word_ids = range(len(self.spec.words))
        site_ids = range(len(self.spec.words[0].sites))
        self.physical_addresses.extend(product(word_ids, site_ids))
        self.physical_address_map.update(
            {site: i for i, site in enumerate(self.physical_addresses)}
        )

        for bus_id, bus in enumerate(self.spec.site_buses):
            for word_id in self.spec.has_site_buses:
                for src, dst in zip(bus.src, bus.dst):
                    src_site = (word_id, src)
                    dst_site = (word_id, dst)
                    lane_addr = IntraMove(Direction.FORWARD, word_id, src, bus_id)
                    self.site_graph.add_edge(
                        self.physical_address_map[src_site],
                        self.physical_address_map[dst_site],
                        lane_addr,
                    )

        for bus_id, bus in enumerate(self.spec.word_buses):
            for src_word, dst_word in zip(bus.src, bus.dst):
                for site in self.spec.has_word_buses:
                    src_site = (src_word, site)
                    dst_site = (dst_word, site)
                    lane_addr = InterMove(Direction.FORWARD, src_word, dst_word, bus_id)
                    self.site_graph.add_edge(
                        self.physical_address_map[src_site],
                        self.physical_address_map[dst_site],
                        lane_addr,
                    )

    def extract_lanes_from_path(self, path: list[int]):
        """Given a path as a list of node indices, extract the lane addresses."""
        if len(path) < 2:
            raise ValueError("Path must have at least two nodes to extract lanes.")
        lanes: list[int] = []
        for src, dst in zip(path[:-1], path[1:]):
            if not self.site_graph.has_edge(src, dst):
                raise ValueError(f"No lane between nodes {src} and {dst}")

            lane: MoveType = self.site_graph.get_edge_data(src, dst)

            lanes.append(lane.get_address(self.spec.encoding))
        return lanes

    def find_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        occupied: frozenset[tuple[int, int]] = frozenset(),
        path_heuristic: Callable[[list[tuple[int, int]]], float] = lambda _: 0.0,
    ):
        """Find a path from start to end avoiding occupied sites.

        Args:
            start: The starting site as a (word_id, site_id) tuple.
            end: The ending site as a (word_id, site_id) tuple.
            occupied: A frozenset of sites (word_id, site_id) that are occupied.
            encoding: The encoding type for the lane addresses.
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

        if start_node in occupied or end_node in occupied:
            raise ValueError("Start or end site is already occupied.")

        subgraph = self.site_graph.subgraph(
            [n for n, ele in enumerate(self.physical_addresses) if ele not in occupied]
        )

        path_nodes = nx.all_shortest_paths(subgraph, start_node, end_node)
        if len(path_nodes) == 0:
            # no path found
            return None, occupied

        path = min(
            path_nodes,
            key=lambda p: path_heuristic([self.physical_addresses[n] for n in p]),
        )
        lanes = self.extract_lanes_from_path(path)
        return lanes, occupied.union(
            frozenset(self.physical_addresses[n] for n in path)
        )
