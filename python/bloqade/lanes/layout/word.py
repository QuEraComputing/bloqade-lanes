from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.geometry.dialects import grid

from bloqade.lanes.bytecode._native import Word as _RustWord
from bloqade.lanes.bytecode.arch import grid_to_rust

from .encoding import LocationAddress

if TYPE_CHECKING:
    from collections.abc import Iterator


class Word:
    """A group of atom sites that share a coordinate grid."""

    _inner: _RustWord
    positions: grid.Grid

    def __init__(
        self,
        positions: grid.Grid,
        site_indices: tuple[tuple[int, int], ...],
        has_cz: tuple[LocationAddress, ...] | None = None,
    ):
        self.positions = positions

        if len(self.positions.positions) != len(site_indices):
            raise ValueError("Number of positions must match number of site indices")
        if has_cz is not None and len(has_cz) != len(site_indices):
            raise ValueError("Length of has_cz must match number of site indices")

        rust_cz = (
            [(addr.word_id, addr.site_id) for addr in has_cz]
            if has_cz is not None
            else None
        )

        self._inner = _RustWord(
            positions=grid_to_rust(positions),
            site_indices=list(site_indices),
            has_cz=rust_cz,
        )

    @property
    def site_indices(self) -> tuple[tuple[int, int], ...]:
        return tuple(self._inner.site_indices)

    @property
    def has_cz(self) -> tuple[LocationAddress, ...] | None:
        cz = self._inner.has_cz
        if cz is None:
            return None
        return tuple(LocationAddress(word_id, site_id) for word_id, site_id in cz)

    @property
    def n_rows(self) -> int:
        """Number of rows (sites per column) in this word."""
        return int(self.positions.shape[1])

    def __getitem__(self, index: int) -> WordSite:
        return WordSite(
            word=self,
            site_index=index,
            cz_pair=self.has_cz[index] if self.has_cz is not None else None,
        )

    def site_position(self, site_index: int) -> tuple[float, float]:
        return self.positions.get(self.site_indices[site_index])

    def all_positions(self) -> Iterator[tuple[float, float]]:
        yield from map(self.site_position, range(len(self.site_indices)))

    def plot(self, ax=None, **scatter_kwargs):  # type: ignore[no-untyped-def]
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

        if ax is None:
            ax = plt.gca()
        x_positions, y_positions = zip(*self.all_positions())
        ax.scatter(x_positions, y_positions, **scatter_kwargs)
        return ax

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            return NotImplemented
        return (
            self.site_indices == other.site_indices
            and self.has_cz == other.has_cz
            and self._inner.positions == other._inner.positions
        )

    def __hash__(self) -> int:
        return hash((self._inner.positions, self.site_indices, self.has_cz))

    def __repr__(self) -> str:
        return f"Word(n_sites={len(self.site_indices)})"


@dataclass(frozen=True)
class WordSite:
    word: Word
    site_index: int
    cz_pair: LocationAddress | None = None

    def position(self) -> tuple[float, float]:
        return self.word.site_position(self.site_index)
