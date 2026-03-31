"""Zone definition data model for multi-zone architectures.

Provides ZoneSpec for defining individual zone properties, DeviceLayout for
physical placement parameters, and ArchBlueprint for composing zones into
a complete architecture definition.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ZoneSpec:
    """Specification for a single zone in a multi-zone architecture.

    Args:
        num_words: Number of words in this zone. Must be >= 1.
        entangling: Whether this zone supports CZ entangling gates.
        measurement: Whether this zone supports measurement.
    """

    num_words: int
    entangling: bool = False
    measurement: bool = True

    def __post_init__(self) -> None:
        if self.num_words < 1:
            raise ValueError(f"num_words must be >= 1, got {self.num_words}")


@dataclass(frozen=True)
class DeviceLayout:
    """Physical layout parameters for word and site placement.

    Args:
        word_size_y: Number of sites per column in each word.
        site_spacing: Distance between adjacent sites within a word (micrometers).
        word_spacing: Distance between adjacent words within a zone (micrometers).
        zone_gap: Vertical gap between zones (micrometers). Added on top of
            the word height (word_size_y * site_spacing).
    """

    word_size_y: int = 5
    site_spacing: float = 10.0
    word_spacing: float = 10.0
    zone_gap: float = 20.0


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
        sizes = {spec.num_words for spec in self.zones.values()}
        if len(sizes) > 1:
            raise ValueError(
                f"All zones must have the same num_words for now, got {sizes}"
            )

    @property
    def words_per_zone(self) -> int:
        """Number of words per zone (all zones are equal size)."""
        return next(iter(self.zones.values())).num_words

    @property
    def total_words(self) -> int:
        """Total number of words across all zones."""
        return sum(spec.num_words for spec in self.zones.values())

    @property
    def zone_names(self) -> tuple[str, ...]:
        """Zone names in definition order."""
        return tuple(self.zones.keys())
