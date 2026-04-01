"""Topology protocols and implementations for zone-based architectures.

Topologies define connectivity patterns (buses) for words and sites.
All topologies are 2D-grid-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bloqade.lanes.layout.arch import Bus

if TYPE_CHECKING:
    from .word_factory import WordGrid


# ── Protocols ──


@runtime_checkable
class SiteTopology(Protocol):
    """Generates site buses for movement within a single word (row of sites)."""

    def generate_site_buses(self, num_sites: int) -> tuple[Bus, ...]: ...


@runtime_checkable
class WordTopology(Protocol):
    """Generates word buses for movement between words in a 2D grid."""

    def generate_word_buses(self, grid: WordGrid) -> tuple[Bus, ...]: ...


@runtime_checkable
class InterZoneTopology(Protocol):
    """Generates word buses connecting words across two zones."""

    def generate_word_buses(
        self, grid_a: WordGrid, grid_b: WordGrid
    ) -> tuple[Bus, ...]: ...


# ── Helpers ──


def _check_power_of_two(n: int, name: str) -> int:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"{name} must be a power of 2, got {n}")
    return int(log2(n))


# ── Site topologies ──


@dataclass(frozen=True)
class HypercubeSiteTopology:
    """Hypercube site connectivity within a single word.

    For N = 2^k sites, produces k buses. Bus for dimension d connects
    sites that differ in bit d: src = [sites with bit d=0],
    dst = [sites with bit d=1]. Each bus has N/2 parallel moves.
    """

    def generate_site_buses(self, num_sites: int) -> tuple[Bus, ...]:
        dims = _check_power_of_two(num_sites, "num_sites")
        buses: list[Bus] = []
        for d in range(dims):
            mask = 1 << d
            src = [i for i in range(num_sites) if i & mask == 0]
            dst = [i | mask for i in src]
            buses.append(Bus(src=src, dst=dst))
        return tuple(buses)


@dataclass(frozen=True)
class AllToAllSiteTopology:
    """All-to-all site connectivity: one bus per (src, dst) pair.

    For N sites, produces N*(N-1)/2 single-element buses allowing
    any site to reach any other site directly.
    """

    def generate_site_buses(self, num_sites: int) -> tuple[Bus, ...]:
        return tuple(
            Bus(src=[i], dst=[j])
            for i in range(num_sites)
            for j in range(i + 1, num_sites)
        )


# ── Word topologies ──


@dataclass(frozen=True)
class HypercubeWordTopology:
    """Hypercube word connectivity along both row and column dimensions.

    For a grid of (R x C) words where R = 2^r and C = 2^c, produces
    r + c buses. Row dimension d connects word(row, col) to
    word(row ^ 2^d, col). Column dimension d connects word(row, col)
    to word(row, col ^ 2^d).
    """

    def generate_word_buses(self, grid: WordGrid) -> tuple[Bus, ...]:
        row_dims = _check_power_of_two(grid.num_rows, "num_rows")
        col_dims = _check_power_of_two(grid.num_cols, "num_cols")
        buses: list[Bus] = []

        for d in range(row_dims):
            mask = 1 << d
            src: list[int] = []
            dst: list[int] = []
            for r in range(grid.num_rows):
                if r & mask != 0:
                    continue
                for c in range(grid.num_cols):
                    src.append(grid.word_id_at(r, c))
                    dst.append(grid.word_id_at(r | mask, c))
            buses.append(Bus(src=src, dst=dst))

        for d in range(col_dims):
            mask = 1 << d
            src = []
            dst = []
            for r in range(grid.num_rows):
                for c in range(grid.num_cols):
                    if c & mask != 0:
                        continue
                    src.append(grid.word_id_at(r, c))
                    dst.append(grid.word_id_at(r, c | mask))
            buses.append(Bus(src=src, dst=dst))

        return tuple(buses)


@dataclass(frozen=True)
class DiagonalWordTopology:
    """Diagonal word connectivity replicating old Gemini site-bus pattern.

    For a grid of (N rows × 2 cols), produces 2*N - 1 buses.
    Each bus connects words in column 0 to words in column 1 at a
    diagonal offset:

    - Group 1 (shift 0..N-1): col_0[r] → col_1[r + shift]
      for r in 0..N-1-shift
    - Group 2 (shift 1..N-1): col_0[r + shift] → col_1[r]
      for r in 0..N-1-shift (reverse diagonal)

    This gives full connectivity between all (col_0, col_1) word pairs,
    organized by diagonal. Requires exactly 2 columns.
    """

    def generate_word_buses(self, grid: WordGrid) -> tuple[Bus, ...]:
        if grid.num_cols != 2:
            raise ValueError(
                f"DiagonalWordTopology requires exactly 2 columns, got {grid.num_cols}"
            )
        n = grid.num_rows
        buses: list[Bus] = []

        # Group 1: col_0[r] → col_1[r + shift]
        for shift in range(n):
            src = [grid.word_id_at(r, 0) for r in range(n - shift)]
            dst = [grid.word_id_at(r + shift, 1) for r in range(n - shift)]
            buses.append(Bus(src=src, dst=dst))

        # Group 2: col_0[r + shift] → col_1[r] (reverse diagonals)
        for diff in range(1, n):
            shift = n - diff
            src = [grid.word_id_at(r + shift, 0) for r in range(n - shift)]
            dst = [grid.word_id_at(r, 1) for r in range(n - shift)]
            buses.append(Bus(src=src, dst=dst))

        return tuple(buses)


# ── Inter-zone topologies ──


@dataclass(frozen=True)
class MatchingTopology:
    """1:1 matching of words by grid position across two zones.

    Produces a single bus pairing grid_a(r, c) with grid_b(r, c)
    for all (r, c). Requires both grids to have the same dimensions.
    """

    def generate_word_buses(
        self, grid_a: WordGrid, grid_b: WordGrid
    ) -> tuple[Bus, ...]:
        if (grid_a.num_rows, grid_a.num_cols) != (grid_b.num_rows, grid_b.num_cols):
            raise ValueError(
                f"Grid dimensions must match: "
                f"({grid_a.num_rows}, {grid_a.num_cols}) vs "
                f"({grid_b.num_rows}, {grid_b.num_cols})"
            )
        src: list[int] = []
        dst: list[int] = []
        for r in range(grid_a.num_rows):
            for c in range(grid_a.num_cols):
                src.append(grid_a.word_id_at(r, c))
                dst.append(grid_b.word_id_at(r, c))
        return (Bus(src=src, dst=dst),)
