"""Word creation helpers for zone-based architectures.

Creates single-column words arranged in a 2D grid, with CZ pairing
between horizontally adjacent words in entangling zones.
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

    Words are assigned IDs in row-major order starting from word_id_offset.
    In entangling zones, horizontally adjacent pairs (col 0-1, 2-3, ...)
    get CZ pairing. Non-entangling zones have has_cz=None on all words.
    """
    base_grid = Grid.from_positions(
        [0.0], [layout.site_spacing * i for i in range(layout.sites_per_word)]
    )
    site_indices = tuple((0, i) for i in range(layout.sites_per_word))

    words: list[Word] = []
    pair_width = layout.word_spacing + layout.pair_spacing
    word_height = (layout.sites_per_word - 1) * layout.site_spacing
    row_step = word_height + layout.row_spacing

    for row in range(zone_spec.num_rows):
        row_y = y_offset + row * row_step
        for col in range(zone_spec.num_cols):
            word_x = x_offset + (col // 2) * pair_width + (col % 2) * layout.word_spacing
            grid = base_grid.shift(word_x, row_y)

            partner_id: int | None = None
            if zone_spec.entangling:
                word_id = word_id_offset + row * zone_spec.num_cols + col
                partner_id = word_id + 1 if col % 2 == 0 else word_id - 1

            has_cz: tuple[LocationAddress, ...] | None = None
            if partner_id is not None:
                has_cz = tuple(
                    LocationAddress(partner_id, i)
                    for i in range(layout.sites_per_word)
                )

            words.append(Word(grid, site_indices, has_cz))

    return WordGrid(
        words=tuple(words),
        num_rows=zone_spec.num_rows,
        num_cols=zone_spec.num_cols,
        word_id_offset=word_id_offset,
    )
