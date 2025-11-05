from dataclasses import dataclass
from itertools import product
from typing import Any, Generic, TypeVar

from .grid import Grid

SiteType = TypeVar("SiteType", bound=Grid | tuple[float, float] | tuple[int, int])


@dataclass(frozen=True)
class Block(Generic[SiteType]):
    # note that the `SiteType` is really just here for visualization purposes
    # you can simply ignore the site in general
    sites: tuple[SiteType, ...]

    def __getitem__(self, index: int):
        return BlockSite(block=self, site_index=index)

    def subblock_positions(self, site_index: int):
        site = self.sites[site_index]
        match site:
            case Grid(x_positions, y_positions):
                yield from ((x, y) for y, x in product(y_positions, x_positions))
            case (float(), float()):
                yield site
            case (int() as x, int() as y):
                yield (float(x), float(y))
            case _:
                raise TypeError(f"Unsupported site type: {type(site)}")

    def all_positions(self):
        for site_index in range(len(self.sites)):
            yield from self.subblock_positions(site_index)

    def plot(self, ax=None, **scatter_kwargs):
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

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
