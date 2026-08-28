from __future__ import annotations

from bloqade.lanes.bytecode._native import Word as _RustWord
from bloqade.lanes.bytecode._wrapper import RustWrapper


class Word(RustWrapper[_RustWord]):
    """A group of atom sites positioned via grid index pairs."""

    def __init__(
        self,
        sites: tuple[tuple[int, int], ...],
    ):
        self._inner = _RustWord(sites=list(sites))

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

    # ``__eq__`` / ``__hash__`` come from ``RustWrapper``, which delegates to
    # ``_native.Word``'s value-based dunders (#476).

    def __repr__(self) -> str:
        return f"Word(n_sites={self.n_sites})"
