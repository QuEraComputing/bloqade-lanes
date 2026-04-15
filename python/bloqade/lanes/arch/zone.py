"""Zone definition data model for multi-zone architectures.

Provides ZoneSpec for defining individual zone properties, DeviceLayout for
physical placement parameters, and ArchBlueprint for composing zones into
a complete architecture definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .topology import SiteTopology, WordTopology


@dataclass(frozen=True)
class ZoneSpec:
    """Specification for a single zone in a multi-zone architecture.

    Words are arranged in a 2D grid (num_rows x num_cols). Horizontally
    adjacent word pairs form CZ entangling pairs in entangling zones.

    Args:
        num_rows: Number of rows in the word grid. Must be >= 1.
        num_cols: Number of columns in the word grid. Must be >= 2 and even.
        entangling: Whether this zone supports CZ entangling gates.
        measurement: Whether this zone supports measurement.
    """

    num_rows: int
    num_cols: int
    entangling: bool = False
    measurement: bool = True
    word_topology: WordTopology | None = None
    site_topology: SiteTopology | None = None

    def __post_init__(self) -> None:
        if self.num_rows < 1:
            raise ValueError(f"num_rows must be >= 1, got {self.num_rows}")
        if self.num_cols < 2:
            raise ValueError(f"num_cols must be >= 2, got {self.num_cols}")
        if self.num_cols % 2 != 0:
            raise ValueError(f"num_cols must be even, got {self.num_cols}")

    @property
    def num_words(self) -> int:
        """Total number of words in this zone."""
        return self.num_rows * self.num_cols


@dataclass(frozen=True)
class DeviceLayout:
    """Physical layout parameters for word and site placement.

    Words are horizontal rows with interleaved CZ pairs. Within a pair,
    sites alternate: even word at x=0, 2s, 4s, ... and odd word at
    x=s, 3s, 5s, ... where s = site_spacing (blockade radius).

    Args:
        sites_per_word: Number of sites per word.
        site_spacing: Distance between adjacent atoms (micrometers).
            Also determines the CZ blockade distance between paired sites.
        pair_spacing: Horizontal gap between adjacent CZ pairs (micrometers).
        row_spacing: Vertical distance between word grid rows (micrometers).
        zone_gap: Additional vertical gap between zones (micrometers).
        x_clearance: Minimum x-axis clearance (µm) between AOD path
            waypoints and grid lines.
        y_clearance: Minimum y-axis clearance (µm) between AOD path
            waypoints and grid lines.  Separate x/y values are useful
            when row and column spacings differ substantially.
    """

    sites_per_word: int = 5
    site_spacing: float = 10.0
    pair_spacing: float = 10.0
    row_spacing: float = 20.0
    zone_gap: float = 20.0
    x_clearance: float = 3.0
    y_clearance: float = 3.0

    def __post_init__(self) -> None:
        if self.sites_per_word < 1:
            raise ValueError(f"sites_per_word must be >= 1, got {self.sites_per_word}")
        if self.x_clearance <= 0:
            raise ValueError(f"x_clearance must be positive, got {self.x_clearance}")
        if self.y_clearance <= 0:
            raise ValueError(f"y_clearance must be positive, got {self.y_clearance}")
        for name in ("site_spacing", "pair_spacing", "row_spacing", "zone_gap"):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be non-negative, got {getattr(self, name)}"
                )


@dataclass(frozen=True)
class ArchBlueprint:
    """High-level architecture definition composed of named zones and layout.

    Zones are ordered by insertion order of the ``zones`` dict, which
    determines word ID assignment (contiguous per zone) and vertical
    layout (top to bottom).

    Args:
        zones: Named zones with their specifications.
        layout: Physical layout parameters for word/site placement.
    """

    zones: dict[str, ZoneSpec]
    layout: DeviceLayout = field(default_factory=DeviceLayout)

    def __post_init__(self) -> None:
        if not self.zones:
            raise ValueError("At least one zone must be specified")
        dims = {(s.num_rows, s.num_cols) for s in self.zones.values()}
        if len(dims) > 1:
            raise ValueError(
                f"All zones must have the same grid dimensions, got {dims}"
            )

    @property
    def words_per_zone(self) -> int:
        """Number of words per zone (all zones have equal grid dimensions)."""
        return next(iter(self.zones.values())).num_words

    @property
    def total_words(self) -> int:
        """Total number of words across all zones."""
        return sum(spec.num_words for spec in self.zones.values())

    @property
    def zone_names(self) -> tuple[str, ...]:
        """Zone names in definition order."""
        return tuple(self.zones.keys())
