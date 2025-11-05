from dataclasses import dataclass
from itertools import chain
from typing import Generic, Sequence

import numpy as np
import rustworkx as nx

from .block import Block, SiteType


@dataclass(frozen=True)
class Lane:
    """A group of inter-lanes that can be executed in parallel.

    For inter-lanes, src and dst are the block indices involved in the inter-lane.
    For intra-lanes, src are the source site indices and dst are the destination site indices.

    """

    src: tuple[int, ...]
    dst: tuple[int, ...]


@dataclass(frozen=True)
class ArchSpec(Generic[SiteType]):
    blocks: tuple[Block[SiteType], ...]
    """List of all blocks in the architecture."""
    has_intra_lanes: frozenset[int]
    """Set of blocks that have intra-lane moves."""
    has_inter_lanes: frozenset[int]  # set of sites in block that have inter-lanes moves
    """Set of sites (by index) that have inter-lane moves. These sites are the same across all blocks."""
    intra_lanes: tuple[Lane, ...]
    """List of all intra-lanes in the architecture by site index."""
    inter_lanes: tuple[Lane, ...]
    """List of all inter-lanes in the architecture by block index."""

    def get_graphs(self):

        block_intra_graph = nx.PyDiGraph()
        intra_move_compat = nx.PyGraph()

        all_moves = list(
            chain.from_iterable(
                (list(zip(lane.src, lane.dst)) for lane in self.intra_lanes)
            )
        )

        move_map = {move: i for i, move in enumerate(all_moves)}

        for lane in self.intra_lanes:
            moves = list(zip(lane.src, lane.dst))
            for src, dst in moves:
                block_intra_graph.add_edge(src, dst, None)

            for i, move_i in enumerate(moves):
                for j, move_j in enumerate(moves[i + 1 :], i + 1):
                    intra_move_compat.add_edge(move_map[move_i], move_map[move_j], None)

        inter_block_graph = nx.PyDiGraph()
        inter_move_compat = nx.PyGraph()

        for lane in self.inter_lanes:
            moves = list(zip(lane.src, lane.dst))
            for src, dst in moves:
                inter_block_graph.add_edge(src, dst, None)

            for i, move_i in enumerate(moves):
                for j, move_j in enumerate(moves[i + 1 :], i + 1):
                    inter_move_compat.add_edge(i, j, None)

    def plot(
        self,
        ax=None,
        show_blocks: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,
    ):
        import matplotlib.pyplot as plt  # type: ignore
        from scipy import interpolate as interp  # type: ignore

        if ax is None:
            ax = plt.gca()

        for block_id in show_blocks:
            block = self.blocks[block_id]
            block.plot(ax)

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        bow_y = (y_max - y_min) * 0.025
        bow_x = (x_max - x_min) * 0.025

        colors = {}
        for block_id in show_blocks:
            block = self.blocks[block_id]
            for lane_id in show_intra:
                lane = self.intra_lanes[lane_id]

                for start, end in zip(lane.src, lane.dst):
                    start = block[start]
                    end = block[end]

                    for (x_start, y_start), (x_end, y_end) in zip(
                        start.positions(), end.positions()
                    ):
                        mid_x = (x_start + x_end) / 2
                        mid_y = (y_start + y_end) / 2

                        if x_start == x_end:
                            mid_x += bow_y
                        elif y_start == y_end:
                            mid_y += bow_x

                        f = interp.interp1d(
                            [x_start, mid_x, x_end],
                            [y_start, mid_y, y_end],
                            kind="quadratic",
                        )
                        x_vals = np.linspace(x_start, x_end, num=10)
                        y_vals = f(x_vals)
                        (ln,) = ax.plot(
                            x_vals, y_vals, color=colors.get(lane), linestyle="--"
                        )
                        if lane not in colors:
                            colors[lane] = ln.get_color()

        for lane in show_inter:
            lane = self.inter_lanes[lane]
            for start_block_id, end_block_id in zip(lane.src, lane.dst):
                start_block = self.blocks[start_block_id]
                end_block = self.blocks[end_block_id]

                for site in self.has_inter_lanes:
                    start = start_block[site]
                    end = end_block[site]
                    for (x_start, y_start), (x_end, y_end) in zip(
                        start.positions(), end.positions()
                    ):
                        mid_x = (x_start + x_end) / 2
                        mid_y = (y_start + y_end) / 2

                        if x_start == x_end:
                            mid_x += bow_y
                        elif y_start == y_end:
                            mid_y += bow_x

                        f = interp.interp1d(
                            [x_start, mid_x, x_end],
                            [y_start, mid_y, y_end],
                            kind="quadratic",
                        )
                        x_vals = np.linspace(x_start, x_end, num=10)
                        y_vals = f(x_vals)
                        (ln,) = ax.plot(
                            x_vals, y_vals, color=colors.get(lane), linestyle="-"
                        )
                        if lane not in colors:
                            colors[lane] = ln.get_color()

        return ax

    def show(
        self,
        ax=None,
        show_blocks: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,
    ):
        import matplotlib.pyplot as plt  # type: ignore

        self.plot(
            ax,
            show_blocks=show_blocks,
            show_intra=show_intra,
            show_inter=show_inter,
            **scatter_kwargs,
        )
        plt.show()
