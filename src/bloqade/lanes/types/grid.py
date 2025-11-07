from dataclasses import dataclass
from functools import cached_property
from itertools import product
from typing import Any, Generic, TypeVar

import numpy as np
from kirin.dialects.ilist import IList

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
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

        if ax is None:
            ax = plt.gca()  # type: ignore
        xx, yy = np.meshgrid(self.x_positions, self.y_positions)
        ax.scatter(xx.flatten(), yy.flatten(), **scatter_kwargs)
        return ax

    def shift(self, dx: float = 0.0, dy: float = 0.0):
        return Grid(
            tuple(x + dx for x in self.x_positions),
            tuple(y + dy for y in self.y_positions),
        )
