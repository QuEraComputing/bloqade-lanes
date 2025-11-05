from dataclasses import dataclass, field
from itertools import product

import rustworkx as nx

from .arch import ArchSpec
from .encoding import Direction, EncodingType, InterMove, IntraMove, MoveType


@dataclass(frozen=True)
class PathFinder:
    spec: ArchSpec
    site_graph: nx.PyGraph = field(init=False, default_factory=nx.PyGraph)
    all_sites: list[tuple[int, int]] = field(init=False, default_factory=list)
    site_map: dict[tuple[int, int], int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.site_map.update({site: i for i, site in enumerate(self.all_sites)})

        block_ids = range(len(self.spec.blocks))
        site_ids = range(len(self.spec.blocks[0].sites))
        self.all_sites.extend(product(block_ids, site_ids))
        self.site_map.update({site: i for i, site in enumerate(self.all_sites)})

        for lane_id, lane in enumerate(self.spec.intra_lanes):
            for block_id in self.spec.has_intra_lanes:
                for src, dst in zip(lane.src, lane.dst):
                    src_site = (block_id, src)
                    dst_site = (block_id, dst)
                    lane_addr = IntraMove(Direction.FORWARD, block_id, src, lane_id)
                    self.site_graph.add_edge(
                        self.site_map[src_site], self.site_map[dst_site], lane_addr
                    )

        for lane_id, lane in enumerate(self.spec.inter_lanes):
            for src_block, dst_block in zip(lane.src, lane.dst):
                for site in self.spec.has_inter_lanes:
                    src_site = (src_block, site)
                    dst_site = (dst_block, site)
                    lane_addr = InterMove(
                        Direction.FORWARD, src_block, dst_block, lane_id
                    )
                    self.site_graph.add_edge(
                        self.site_map[src_site], self.site_map[dst_site], lane_addr
                    )

    def extract_lanes_from_path(self, path: list[int], encoding: EncodingType):
        """Given a path as a list of node indices, extract the lane addresses."""
        if len(path) < 2:
            raise ValueError("Path must have at least two nodes to extract lanes.")
        lanes: list[int] = []
        for src, dst in zip(path[:-1], path[1:]):
            if not self.site_graph.has_edge(src, dst):
                raise ValueError(f"No lane between nodes {src} and {dst}")

            lane: MoveType = self.site_graph.get_edge_data(src, dst)

            lanes.append(lane.get_address(encoding))
        return lanes

    def find_path(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        occupied: frozenset[tuple[int, int]] = frozenset(),
        encoding: EncodingType = EncodingType.BIT64,
    ):
        start_node = self.site_map[start]
        end_node = self.site_map[end]

        if start_node in occupied or end_node in occupied:
            return None, occupied

        subgraph = self.site_graph.subgraph(
            [n for n, ele in enumerate(self.all_sites) if ele not in occupied]
        )

        path_nodes = nx.all_shortest_paths(subgraph, start_node, end_node)
        if len(path_nodes) == 0:
            # no path found
            return None, occupied

        # TODO: Pick best path based on some heuristic
        path = path_nodes[0]
        lanes = self.extract_lanes_from_path(path, encoding)
        return lanes, occupied.union(frozenset(self.all_sites[n] for n in path))
