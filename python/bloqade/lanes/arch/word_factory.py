"""Word creation helpers for zone-based architectures.

Creates row-words arranged in a 2D grid, with interleaved CZ pairing
between horizontally adjacent words in entangling zones.

Physical layout (2 words × 5 sites, interleaved):

  w0.s0  w1.s0  w0.s1  w1.s1  w0.s2  w1.s2  w0.s3  w1.s3  w0.s4  w1.s4
    o      o      o      o      o      o      o      o      o      o

CZ partners (w0.si and w1.si) are separated by site_spacing (blockade radius).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bloqade.geometry.dialects.grid import Grid

from bloqade.lanes.layout.encoding import LocationAddress
from bloqade.lanes.layout.word import Word

from .zone import DeviceLayout, ZoneSpec

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class WordGrid:
    """2D grid of words within a zone, preserving row/col structure."""

    words: tuple[Word, ...]
    num_rows: int
    num_cols: int
    word_id_offset: int

    def word_at(self, row: int, col: int) -> Word:
        """Get the word at a given grid position."""
        return self.words[row * self.num_cols + col]

    def word_id_at(self, row: int, col: int) -> int:
        """Get the global word ID at a given grid position."""
        return self.word_id_offset + row * self.num_cols + col

    def cz_pairs(self) -> Iterator[tuple[int, int]]:
        """Yield (word_id_a, word_id_b) for all CZ entangling pairs."""
        for row in range(self.num_rows):
            for col in range(0, self.num_cols, 2):
                yield (self.word_id_at(row, col), self.word_id_at(row, col + 1))


def create_zone_words(
    zone_spec: ZoneSpec,
    layout: DeviceLayout,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    word_id_offset: int = 0,
) -> WordGrid:
    """Create all words for a zone in a 2D grid layout.

    Words are horizontal rows with sites along the x-axis. Within a CZ pair,
    the two words' sites are interleaved: even word at x = 0, 2s, 4s, ...
    and odd word at x = s, 3s, 5s, ... (where s = site_spacing).

    Words are assigned IDs in row-major order starting from word_id_offset.
    In entangling zones, horizontally adjacent pairs (col 0-1, 2-3, ...)
    get CZ pairing. Non-entangling zones have has_cz=None on all words.
    """
    n = layout.sites_per_word
    s = layout.site_spacing
    site_indices = tuple((i, 0) for i in range(n))

    # Even word sites at x = 0, 2s, 4s, ...
    even_x = [2.0 * s * i for i in range(n)]
    even_base = Grid.from_positions(even_x, [0.0])

    # Odd word sites at x = s, 3s, 5s, ...
    odd_x = [s + 2.0 * s * i for i in range(n)]
    odd_base = Grid.from_positions(odd_x, [0.0])

    # Width of one interleaved pair
    pair_width = (2 * n - 1) * s

    words: list[Word] = []

    for row in range(zone_spec.num_rows):
        row_y = y_offset + row * layout.row_spacing
        for col in range(zone_spec.num_cols):
            pair_idx = col // 2
            is_odd = col % 2
            pair_x = x_offset + pair_idx * (pair_width + layout.pair_spacing)

            base = odd_base if is_odd else even_base
            grid = base.shift(pair_x, row_y)

            partner_id: int | None = None
            if zone_spec.entangling:
                word_id = word_id_offset + row * zone_spec.num_cols + col
                partner_id = word_id + 1 if not is_odd else word_id - 1

            has_cz: tuple[LocationAddress, ...] | None = None
            if partner_id is not None:
                has_cz = tuple(
                    LocationAddress(partner_id, i) for i in range(n)
                )

            words.append(Word(grid, site_indices, has_cz))

    return WordGrid(
        words=tuple(words),
        num_rows=zone_spec.num_rows,
        num_cols=zone_spec.num_cols,
        word_id_offset=word_id_offset,
    )
