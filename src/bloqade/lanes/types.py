from dataclasses import dataclass, field
from functools import cached_property
from itertools import combinations, product

# from bloqade.geometry.dialects import grid
from typing import Any, Generic, Sequence, TypeVar

import networkx as nx
import numpy as np
import scipy

# from bloqade.shuttle.arch import Layout
from kirin.dialects.ilist import IList
from kirin.ir.attrs.types import is_tuple_of
from matplotlib import pyplot as plt

Nx = TypeVar("Nx")
Ny = TypeVar("Ny")


@dataclass(frozen=True)
class Grid(Generic[Nx, Ny]):
    x_positions: tuple[float, ...]
    y_positions: tuple[float, ...]

    @cached_property
    def shape(self) -> tuple[int, int]:
        return (len(self.x_positions), len(self.y_positions))

    @cached_property
    def positions(self) -> tuple[tuple[float, float], ...]:
        return tuple((x, y) for y, x in product(self.y_positions, self.x_positions))

    @staticmethod
    def slice_pos(
        positions: tuple[float, ...], slice_obj: slice | int | IList[int, Any]
    ):
        match slice_obj:
            case slice():
                return positions[slice_obj]
            case int():
                return positions[slice_obj : slice_obj + 1]
            case IList():
                return tuple(positions[i] for i in slice_obj.data)
            case _:
                raise TypeError(f"Invalid slice object: {slice_obj}")

    def __getitem__(self, key: tuple[Any, Any]):
        return Grid(
            self.slice_pos(self.x_positions, key[0]),
            self.slice_pos(self.y_positions, key[1]),
        )

    def plot(self, ax=None, **scatter_kwargs):
        if ax is None:
            ax = plt.gca()
        xx, yy = np.meshgrid(self.x_positions, self.y_positions)
        ax.scatter(xx.flatten(), yy.flatten(), **scatter_kwargs)
        return ax


SiteType = TypeVar("SiteType")


@dataclass(frozen=True)
class Block(Generic[SiteType]):
    # note that the `SiteType` is really just here for visualization purposes
    # you can simply ignore the site in general
    sites: tuple[SiteType, ...]

    def __getitem__(self, index: int):
        return BlockSite(block=self, site_index=index)

    def subblock_positions(self, site_index: int):
        site = self.sites[site_index]
        if isinstance(site, Grid):
            yield from product(site.x_positions, site.y_positions)
        elif is_tuple_of(site, float) or is_tuple_of(site, int) and len(site) == 2:
            yield site
        else:
            raise TypeError(f"Invalid site type: {type(site)}")

    def all_positions(self):
        for site_index in range(len(self.sites)):
            yield from self.subblock_positions(site_index)

    def plot(self, ax=None, **scatter_kwargs):
        if ax is None:
            ax = plt.gca()
        x_positions, y_positions = zip(*self.all_positions())
        ax.scatter(x_positions, y_positions, **scatter_kwargs)
        return ax


BlockType = TypeVar("BlockType", bound=Block[Any])


@dataclass(frozen=True)
class BlockSite(Generic[BlockType]):
    block: BlockType
    site_index: int

    def positions(self):
        yield from self.block.subblock_positions(self.site_index)


@dataclass(frozen=True)
class IntraLane:
    site_1: int
    site_2: int


@dataclass(frozen=True)
class InterLane(Generic[BlockType]):
    blocks: tuple[BlockType, ...]
    site: int


@dataclass(frozen=True)
class ArchSpec(Generic[SiteType]):
    blocks: tuple[Block[SiteType], ...]
    intra_lanes: tuple[IntraLane, ...]
    allowed_inter_lanes: tuple[InterLane[Block[SiteType]], ...]
    has_intra_lanes: frozenset[
        Block[SiteType]
    ]  # the block indices that have intra-lanes
    _block_id: dict[Block[SiteType], int] = field(
        init=False, default_factory=dict, repr=False, compare=False, hash=False
    )

    def __post_init__(self):
        assert len(self.blocks) == len(set(self.blocks)), "Blocks must be unique"
        self._block_id.update({block: i for i, block in enumerate(self.blocks)})

    def get_graph(self):
        g: nx.Graph[tuple[int, int, int]] = nx.Graph()
        raise NotImplementedError("Graph generation not implemented yet")

        return g

    def plot(
        self,
        ax=None,
        show_blocks: Sequence[int] = (),
        show_intra: Sequence[int] = (),
        show_inter: Sequence[int] = (),
        **scatter_kwargs,
    ):
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

                start = block[lane.site_1]
                end = block[lane.site_2]

                for (x_start, y_start), (x_end, y_end) in zip(
                    start.positions(), end.positions()
                ):
                    mid_x = (x_start + x_end) / 2
                    mid_y = (y_start + y_end) / 2

                    if x_start == x_end:
                        mid_x += bow_y
                    elif y_start == y_end:
                        mid_y += bow_x

                    interp = scipy.interpolate.interp1d(
                        [x_start, mid_x, x_end],
                        [y_start, mid_y, y_end],
                        kind="quadratic",
                    )
                    x_vals = np.linspace(x_start, x_end, num=10)
                    y_vals = interp(x_vals)
                    (ln,) = ax.plot(
                        x_vals, y_vals, color=colors.get(lane), linestyle="--"
                    )
                    if lane not in colors:
                        colors[lane] = ln.get_color()

        for lane in show_inter:
            lane = self.allowed_inter_lanes[lane]
            for block_1, block_2 in combinations(lane.blocks, 2):
                start = block_1[lane.site]
                end = block_2[lane.site]

                for (x_start, y_start), (x_end, y_end) in zip(
                    start.positions(), end.positions()
                ):
                    mid_x = (x_start + x_end) / 2
                    mid_y = (y_start + y_end) / 2

                    if x_start == x_end:
                        mid_x += bow_y
                    elif y_start == y_end:
                        mid_y += bow_x

                    interp = scipy.interpolate.interp1d(
                        [x_start, mid_x, x_end],
                        [y_start, mid_y, y_end],
                        kind="quadratic",
                    )
                    x_vals = np.linspace(x_start, x_end, num=10)
                    y_vals = interp(x_vals)
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
        self.plot(
            ax,
            show_blocks=show_blocks,
            show_intra=show_intra,
            show_inter=show_inter,
            **scatter_kwargs,
        )
        plt.show()
