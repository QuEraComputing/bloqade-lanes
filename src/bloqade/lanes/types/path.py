from dataclasses import dataclass, field
from itertools import product

import rustworkx as nx

from .arch import ArchSpec
from .encoding import Direction, EncodingType, InterMove, IntraMove, MoveType


@dataclass(frozen=True)
class PathFinder:
    spec: ArchSpec
    site_graph: nx.PyGraph
    all_sites: list[tuple[int, int]]  # site-node (int) -> (block_id, site_index)
    site_map: dict[tuple[int, int], int] = field(init=False, default_factory=dict)

    def __post_init__(self):
        self.site_map.update({site: i for i, site in enumerate(self.all_sites)})

    @classmethod
    def new(cls, spec: ArchSpec):
        block_ids = range(len(spec.blocks))
        site_ids = range(len(spec.blocks[0].sites))
        all_sites = list(product(block_ids, site_ids))
        site_map = {site: i for i, site in enumerate(all_sites)}
        site_graph = nx.PyGraph()

        for lane_id, lane in enumerate(spec.intra_lanes):
            for block_id in spec.has_intra_lanes:
                for src, dst in zip(lane.src, lane.dst):
                    src_site = (block_id, src)
                    dst_site = (block_id, dst)
                    lane_addr = IntraMove(Direction.FORWARD, block_id, src, lane_id)
                    site_graph.add_edge(
                        site_map[src_site], site_map[dst_site], lane_addr
                    )

        for lane_id, lane in enumerate(spec.inter_lanes):
            for src_block, dst_block in zip(lane.src, lane.dst):
                for site in spec.has_inter_lanes:
                    src_site = (src_block, site)
                    dst_site = (dst_block, site)
                    lane_addr = InterMove(
                        Direction.FORWARD, src_block, dst_block, lane_id
                    )
                    site_graph.add_edge(
                        site_map[src_site], site_map[dst_site], lane_addr
                    )

        return cls(spec, site_graph, all_sites)

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
            return None, occupied

        path = path_nodes[0]
        lanes: list[int] = []
        new_occupied = set(occupied)

        for src, dst in zip(path[:-1], path[1:]):
            lane: MoveType = subgraph.get_edge_data(src, dst)
            new_occupied.update((src_site := self.all_sites[src], self.all_sites[dst]))
            if lane.src_site() != src_site:
                lane = lane.reverse()

            lanes.append(lane.get_address(encoding))

        return lanes, frozenset(new_occupied)
