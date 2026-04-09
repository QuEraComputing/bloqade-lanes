"""Zone-aware architecture builder.

Provides ``ZoneBuilder`` (single-zone construction with validation) and
``ArchBuilder`` (multi-zone composition) as the low-level building blocks.
The high-level ``build_arch()`` function uses these internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bloqade.lanes.bytecode._native import (
    Grid as _RustGrid,
    LocationAddress as _RustLocAddr,
    Mode as _RustMode,
    SiteBus as _RustSiteBus,
    WordBus as _RustWordBus,
    Zone as _RustZone,
    ZoneBus as _RustZoneBus,
)
from bloqade.lanes.layout.arch import ArchSpec
from bloqade.lanes.layout.word import Word

if TYPE_CHECKING:
    pass


# ── Helpers ──


def _normalize_index(idx: slice | int | list[int], size: int) -> list[int]:
    """Convert a slice, int, or list to a sorted list of indices."""
    if isinstance(idx, slice):
        return list(range(*idx.indices(size)))
    if isinstance(idx, int):
        return [idx]
    return list(idx)


def _validate_aod_rectangle(
    positions: list[tuple[int, int]],
    label: str,
) -> None:
    """Validate that positions form a Cartesian product on the grid.

    The set of (x, y) positions must equal {x_values} × {y_values}
    for some sets x_values and y_values. This ensures the bus defines
    a complete rectangular AOD grid.
    """
    if not positions:
        return
    pos_set = set(positions)
    xs = sorted({p[0] for p in positions})
    ys = sorted({p[1] for p in positions})
    expected = {(x, y) for x in xs for y in ys}
    if pos_set != expected:
        missing = expected - pos_set
        raise ValueError(
            f"{label} positions do not form a valid AOD Cartesian product. "
            f"Missing positions: {sorted(missing)}"
        )


# ── Grid query helpers ──


class _SiteGridQuery:
    """Query site indices within a word shape.

    Site index is computed as ``x + y * num_x`` (row-major in x).
    """

    def __init__(self, word_shape: tuple[int, int]):
        self._nx, self._ny = word_shape

    def __getitem__(self, key: tuple[slice | int | list[int], slice | int | list[int]]) -> list[int]:  # type: ignore[override]
        x_idx, y_idx = key
        xs = _normalize_index(x_idx, self._nx)
        ys = _normalize_index(y_idx, self._ny)
        return sorted(x + y * self._nx for x in xs for y in ys)


class _WordGridQuery:
    """Query word indices by grid region on a ZoneBuilder."""

    def __init__(self, zone: ZoneBuilder):
        self._zone = zone

    def __getitem__(self, key: tuple[slice | int | list[int], slice | int | list[int]]) -> tuple[str, list[int]]:  # type: ignore[override]
        x_idx, y_idx = key
        xs = _normalize_index(x_idx, self._zone._grid.num_x)
        ys = _normalize_index(y_idx, self._zone._grid.num_y)
        query_positions = {(x, y) for x in xs for y in ys}
        hits: set[int] = set()
        for pos, word_id in self._zone._position_to_word.items():
            if pos in query_positions:
                hits.add(word_id)
        return (self._zone._name, sorted(hits))


# ── ZoneBuilder ──


class ZoneBuilder:
    """Build a single zone with its words, grid, and buses.

    All indices are zone-local. Words are placed on the zone's grid
    and validated for shape and overlap. Buses are validated for AOD
    Cartesian product compliance.
    """

    def __init__(self, name: str, grid: _RustGrid, word_shape: tuple[int, int]):
        """Initialize a zone.

        Args:
            name: Human-readable zone name (stored in Rust Zone).
            grid: Coordinate grid for this zone.
            word_shape: (num_x_sites, num_y_sites) — uniform shape for
                all words in this zone. sites_per_word = product of shape.
        """
        self._name = name
        self._grid = grid
        self._word_shape = word_shape
        self._words: list[list[tuple[int, int]]] = []
        self._position_to_word: dict[tuple[int, int], int] = {}
        self._site_buses: list[tuple[list[int], list[int]]] = []
        self._word_buses: list[tuple[list[int], list[int]]] = []
        self._entangling_pairs: list[tuple[int, int]] = []

    @property
    def name(self) -> str:
        """Zone name."""
        return self._name

    @property
    def word_shape(self) -> tuple[int, int]:
        """(num_x_sites, num_y_sites) for each word."""
        return self._word_shape

    @property
    def sites_per_word(self) -> int:
        """Total sites per word (product of word_shape)."""
        return self._word_shape[0] * self._word_shape[1]

    @property
    def num_words(self) -> int:
        """Number of words added so far."""
        return len(self._words)

    def add_word(
        self,
        x_sites: slice | list[int],
        y_sites: slice | list[int],
    ) -> int:
        """Add a word occupying the given grid positions.

        The number of x-indices and y-indices must match word_shape.
        Grid positions must not overlap with any existing word.

        Returns:
            Zone-local word index.

        Raises:
            ValueError: Shape mismatch or grid position overlap.
            IndexError: Indices out of range for this zone's grid.
        """
        xs = _normalize_index(x_sites, self._grid.num_x)
        ys = _normalize_index(y_sites, self._grid.num_y)

        if len(xs) != self._word_shape[0]:
            raise ValueError(
                f"x_sites has {len(xs)} indices but word_shape requires "
                f"{self._word_shape[0]}"
            )
        if len(ys) != self._word_shape[1]:
            raise ValueError(
                f"y_sites has {len(ys)} indices but word_shape requires "
                f"{self._word_shape[1]}"
            )

        for x in xs:
            if x < 0 or x >= self._grid.num_x:
                raise IndexError(
                    f"x index {x} out of range for grid with "
                    f"{self._grid.num_x} x-positions"
                )
        for y in ys:
            if y < 0 or y >= self._grid.num_y:
                raise IndexError(
                    f"y index {y} out of range for grid with "
                    f"{self._grid.num_y} y-positions"
                )

        positions = [(x, y) for y in ys for x in xs]
        for pos in positions:
            if pos in self._position_to_word:
                owner = self._position_to_word[pos]
                raise ValueError(
                    f"Grid position (x={pos[0]}, y={pos[1]}) "
                    f"already belongs to word {owner}"
                )

        word_id = len(self._words)
        self._words.append(positions)
        for pos in positions:
            self._position_to_word[pos] = word_id
        return word_id

    def add_site_bus(self, src: list[int], dst: list[int]) -> None:
        """Add a site bus (intra-word movement).

        src/dst are site indices within word_shape (0..sites_per_word).
        Validates that src and dst positions each form a valid AOD
        Cartesian product on the word grid.
        """
        total = self.sites_per_word
        nx = self._word_shape[0]
        for s in src:
            if s < 0 or s >= total:
                raise ValueError(f"site index {s} out of range [0, {total})")
        for d in dst:
            if d < 0 or d >= total:
                raise ValueError(f"site index {d} out of range [0, {total})")

        src_positions = [(s % nx, s // nx) for s in src]
        dst_positions = [(d % nx, d // nx) for d in dst]
        _validate_aod_rectangle(src_positions, "Site bus src")
        _validate_aod_rectangle(dst_positions, "Site bus dst")
        self._site_buses.append((list(src), list(dst)))

    def add_word_bus(self, src: list[int], dst: list[int]) -> None:
        """Add a word bus (intra-zone movement).

        src/dst are zone-local word indices. Validates that src and dst
        word positions each form a valid AOD Cartesian product on the
        zone grid.
        """
        n = len(self._words)
        for s in src:
            if s < 0 or s >= n:
                raise ValueError(f"word index {s} out of range [0, {n})")
        for d in dst:
            if d < 0 or d >= n:
                raise ValueError(f"word index {d} out of range [0, {n})")

        src_positions = [self._word_origin(s) for s in src]
        dst_positions = [self._word_origin(d) for d in dst]
        _validate_aod_rectangle(src_positions, "Word bus src")
        _validate_aod_rectangle(dst_positions, "Word bus dst")
        self._word_buses.append((list(src), list(dst)))

    def add_entangling_pair(self, word_a: int, word_b: int) -> None:
        """Mark two zone-local words as a CZ pair."""
        n = len(self._words)
        if word_a < 0 or word_a >= n:
            raise ValueError(f"word index {word_a} out of range [0, {n})")
        if word_b < 0 or word_b >= n:
            raise ValueError(f"word index {word_b} out of range [0, {n})")
        self._entangling_pairs.append((word_a, word_b))

    @property
    def words(self) -> _WordGridQuery:
        """Query word indices by grid region.

        Returns ``(zone_name, list[int])`` — the zone name and zone-local
        word indices whose sites intersect the queried region.

        The returned tuple can be passed directly to ``ArchBuilder.connect()``.
        """
        return _WordGridQuery(self)

    @property
    def sites(self) -> _SiteGridQuery:
        """Query site indices within the word shape."""
        return _SiteGridQuery(self._word_shape)

    def _word_origin(self, word_id: int) -> tuple[int, int]:
        """Get the (min_x, min_y) origin of a word for AOD validation."""
        positions = self._words[word_id]
        return (min(p[0] for p in positions), min(p[1] for p in positions))


# ── ArchBuilder ──


class ArchBuilder:
    """Compose ``ZoneBuilder``s into a complete ``ArchSpec``.

    Each zone added gets assigned global word IDs. Inter-zone connections
    go into ``zone_buses``. Calls Rust validation on ``build()``.
    """

    def __init__(self) -> None:
        self._zones: list[ZoneBuilder] = []
        self._zone_name_to_id: dict[str, int] = {}
        self._word_id_offsets: list[int] = []
        self._connections: list[tuple[tuple[str, list[int]], tuple[str, list[int]]]] = (
            []
        )
        self._modes: list[tuple[str, list[str]]] = []
        self._total_words: int = 0

    def add_zone(self, zone: ZoneBuilder) -> int:
        """Add a zone. Returns zone_id. Assigns global word IDs.

        Validates that sites_per_word matches across all zones.
        """
        if zone.name in self._zone_name_to_id:
            raise ValueError(f"Duplicate zone name: '{zone.name}'")
        if self._zones:
            existing_spw = self._zones[0].sites_per_word
            if zone.sites_per_word != existing_spw:
                raise ValueError(
                    f"sites_per_word mismatch: zone '{zone.name}' has "
                    f"{zone.sites_per_word} but existing zones have "
                    f"{existing_spw}"
                )
        zone_id = len(self._zones)
        self._zone_name_to_id[zone.name] = zone_id
        self._word_id_offsets.append(self._total_words)
        self._total_words += zone.num_words
        self._zones.append(zone)
        return zone_id

    def connect(
        self,
        src: tuple[str, list[int]],
        dst: tuple[str, list[int]],
    ) -> None:
        """Add an inter-zone bus (zone_buses).

        Args:
            src: ``(zone_name, zone_local_word_indices)`` — typically
                from ``zone.words[...]``.
            dst: ``(zone_name, zone_local_word_indices)`` — same format.

        Validates AOD Cartesian product across the two zone grids.
        """
        src_name, src_words = src
        dst_name, dst_words = dst
        if src_name not in self._zone_name_to_id:
            raise ValueError(f"Unknown zone: '{src_name}'")
        if dst_name not in self._zone_name_to_id:
            raise ValueError(f"Unknown zone: '{dst_name}'")

        src_zone = self._zones[self._zone_name_to_id[src_name]]
        dst_zone = self._zones[self._zone_name_to_id[dst_name]]
        src_positions = [src_zone._word_origin(w) for w in src_words]
        dst_positions = [dst_zone._word_origin(w) for w in dst_words]
        _validate_aod_rectangle(src_positions, "Zone bus src")
        _validate_aod_rectangle(dst_positions, "Zone bus dst")

        self._connections.append((src, dst))

    def add_mode(self, name: str, zones: list[str]) -> None:
        """Add an operational mode.

        Args:
            name: Mode name (e.g. "all", "gate", "measure").
            zones: Zone names to include in this mode.
        """
        for z in zones:
            if z not in self._zone_name_to_id:
                raise ValueError(f"Unknown zone: '{z}'")
        self._modes.append((name, list(zones)))

    def build(self) -> ArchSpec:
        """Assemble the ArchSpec and validate via Rust.

        Raises:
            ValueError: If Rust validation fails.
        """
        # 1. Collect all words with global IDs.
        all_words: list[Word] = []
        for zone in self._zones:
            for positions in zone._words:
                all_words.append(Word(tuple(positions)))

        # 2. Build Rust Zone objects.
        rust_zones: list[_RustZone] = []
        for zone in self._zones:
            site_buses = [_RustSiteBus(src=s, dst=d) for s, d in zone._site_buses]
            word_buses = [_RustWordBus(src=s, dst=d) for s, d in zone._word_buses]
            words_with_site_buses = list(range(zone.num_words)) if site_buses else []
            sites_with_word_buses = (
                list(range(zone.sites_per_word)) if word_buses else []
            )
            rust_zones.append(
                _RustZone(
                    name=zone.name,
                    grid=zone._grid,
                    site_buses=site_buses,
                    word_buses=word_buses,
                    words_with_site_buses=words_with_site_buses,
                    sites_with_word_buses=sites_with_word_buses,
                    entangling_pairs=zone._entangling_pairs,
                )
            )

        # 3. Build zone_buses from connect() calls.
        zone_buses: list[_RustZoneBus] = []
        for (src_name, src_words), (dst_name, dst_words) in self._connections:
            src_zid = self._zone_name_to_id[src_name]
            dst_zid = self._zone_name_to_id[dst_name]
            src_offset = self._word_id_offsets[src_zid]
            dst_offset = self._word_id_offsets[dst_zid]
            zone_buses.append(
                _RustZoneBus(
                    src=[(src_zid, src_offset + w) for w in src_words],
                    dst=[(dst_zid, dst_offset + w) for w in dst_words],
                )
            )

        # 4. Build modes.
        modes: list[_RustMode] = []
        for mode_name, zone_names in self._modes:
            zone_ids = [self._zone_name_to_id[z] for z in zone_names]
            bitstring_order: list[_RustLocAddr] = []
            for zid in zone_ids:
                offset = self._word_id_offsets[zid]
                zone = self._zones[zid]
                for w in range(zone.num_words):
                    for s in range(zone.sites_per_word):
                        bitstring_order.append(_RustLocAddr(zid, offset + w, s))
            modes.append(
                _RustMode(
                    name=mode_name,
                    zones=zone_ids,
                    bitstring_order=bitstring_order,
                )
            )

        # 5. Assemble and validate.
        return ArchSpec.from_components(
            words=tuple(all_words),
            zones=tuple(rust_zones),
            modes=modes,
            zone_buses=zone_buses,
        )
