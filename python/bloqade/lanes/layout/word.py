from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.lanes.bytecode._native import Word as _RustWord

if TYPE_CHECKING:
    pass


class Word:
    """A group of atom sites positioned via grid index pairs."""

    _inner: _RustWord

    def __init__(
        self,
        sites: tuple[tuple[int, int], ...],
    ):
        self._inner = _RustWord(
            sites=list(sites),
        )

    @property
    def sites(self) -> tuple[tuple[int, int], ...]:
        return tuple((s[0], s[1]) for s in self._inner.sites)

    @property
    def site_indices(self) -> tuple[tuple[int, int], ...]:
        """Alias for sites, for backward compatibility."""
        return self.sites

    @property
    def n_sites(self) -> int:
        """Number of sites in this word."""
        return len(self._inner.sites)

    def __getitem__(self, index: int) -> WordSite:
        return WordSite(word=self, site_index=index)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            return NotImplemented
        return self.sites == other.sites

    def __hash__(self) -> int:
        return hash(self.sites)

    def __repr__(self) -> str:
        return f"Word(n_sites={self.n_sites})"


@dataclass(frozen=True)
class WordSite:
    word: Word
    site_index: int
