from __future__ import annotations

from bloqade.lanes._wrapper import RustWrapper
from bloqade.lanes.bytecode._native import Word as _RustWord


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

    # NOTE: the underlying Rust ``_native.Word`` does not yet implement
    # value-based ``__eq__``/``__hash__`` (it falls back to identity), so we
    # cannot rely on ``RustWrapper``'s delegation here. Override with a
    # ``sites``-based comparison to preserve the legacy semantic that
    # ``ArchSpec.__eq__`` relies on (it does ``self.words == other.words``).
    # Tracked in #476 — drop these overrides once Word/Mode/Zone get
    # value-based dunders on the Rust side.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Word):
            return NotImplemented
        return self.sites == other.sites

    def __hash__(self) -> int:
        return hash(self.sites)

    def __repr__(self) -> str:
        return f"Word(n_sites={self.n_sites})"
