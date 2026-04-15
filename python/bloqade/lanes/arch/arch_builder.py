"""Zone-aware architecture builder.

Provides ``ZoneBuilder`` (single-zone construction with validation) and
``ArchBuilder`` (multi-zone composition) as the low-level building blocks.
The high-level ``build_arch()`` function uses these internally.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
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

# All internal lengths are stored as integer nm counts so that distance
# comparisons for blockade-radius inference are exact (no floating-point
# edge cases when a site sits right at the radius boundary).
_NM_PER_UM = 1000


def _to_nm(value_um: float, name: str) -> int:
    """Convert a µm length to an integer nm count, validating precision.

    Raises ``ValueError`` if *value_um* is NaN, infinite, or has sub-nm
    resolution.
    """
    if not math.isfinite(value_um):
        raise ValueError(f"{name} must be finite, got {value_um}")
    scaled = value_um * _NM_PER_UM
    rounded = round(scaled)
    if abs(scaled - rounded) > 1e-6:
        raise ValueError(f"{name} {value_um} µm is not representable at 1 nm precision")
    return int(rounded)


def _normalize_index(idx: slice | int | Sequence[int], size: int) -> list[int]:
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

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> list[int]:
        x_idx, y_idx = key
        xs = _normalize_index(x_idx, self._nx)
        ys = _normalize_index(y_idx, self._ny)
        return sorted(x + y * self._nx for x in xs for y in ys)


class _WordGridQuery:
    """Query word indices by grid region on a ZoneBuilder."""

    def __init__(self, zone: ZoneBuilder):
        self._zone = zone

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> tuple[str, list[int]]:
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
        self._blockade_radius_nm: int | None = None

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
        x_sites: slice | Sequence[int],
        y_sites: slice | Sequence[int],
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

    def add_site_bus(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a site bus (intra-word movement).

        src/dst are site indices within word_shape (0..sites_per_word).
        Must have equal length. Validates that src and dst positions each
        form a valid AOD Cartesian product on the word grid.
        """
        if len(src) != len(dst):
            raise ValueError(
                f"Site bus src has {len(src)} entries but dst has {len(dst)}"
            )
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

    def add_word_bus(self, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a word bus (intra-zone movement).

        src/dst are zone-local word indices. Must have equal length.
        Validates that src and dst word positions each form a valid AOD
        Cartesian product on the zone grid.
        """
        if len(src) != len(dst):
            raise ValueError(
                f"Word bus src has {len(src)} entries but dst has {len(dst)}"
            )
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

    def add_entangling_pairs(
        self, words_a: Sequence[int], words_b: Sequence[int]
    ) -> None:
        """Mark paired zone-local words as CZ pairs.

        ``words_a[i]`` is paired with ``words_b[i]``. The two sequences
        must have the same length.

        For most users, prefer :meth:`set_blockade_radius` — it derives
        the pair list directly from geometry and validates the word
        layout against the CZ-pairing convention.

        Any ``blockade_radius`` previously recorded on this zone (via
        :meth:`set_blockade_radius`) is cleared, since a manual append
        means the pair list is no longer purely radius-derived.
        """
        if len(words_a) != len(words_b):
            raise ValueError(
                f"words_a has {len(words_a)} entries but words_b has " f"{len(words_b)}"
            )
        n = len(self._words)
        for a, b in zip(words_a, words_b):
            if a < 0 or a >= n:
                raise ValueError(f"word index {a} out of range [0, {n})")
            if b < 0 or b >= n:
                raise ValueError(f"word index {b} out of range [0, {n})")
            self._entangling_pairs.append((a, b))
        # Pair list was just manually modified; the cached radius no
        # longer describes its full state.
        self._blockade_radius_nm = None

    @property
    def blockade_radius(self) -> float | None:
        """Rydberg blockade radius (µm) used to derive entangling pairs, or None."""
        if self._blockade_radius_nm is None:
            return None
        return self._blockade_radius_nm / _NM_PER_UM

    def set_blockade_radius(self, radius: float) -> None:
        """Derive entangling word pairs from the Rydberg blockade radius.

        Scans every pair of distinct words in the zone and classifies
        each under the matching-site-index CZ convention:

        * All matching-index site distances ``<= radius`` and all
          non-matching-index site distances ``> radius``: valid CZ pair.
        * Some matching-index distances within radius, some outside:
          ``ValueError`` (partial blockade — the word layout doesn't
          cleanly map onto the CZ-pairing convention).
        * Any non-matching-index site distance within radius (regardless
          of whether the matching-index distances also fall within):
          ``ValueError`` (crossed-index — two words are arranged such
          that site ``i`` of one word sits next to site ``j != i`` of
          the other, violating the exclusivity the convention requires).
        * All distances outside radius: words ignore each other, no
          pair recorded.

        After classification, every word must appear in **at most one**
        valid pair; multiple partners raise ``ValueError``.

        This call **overwrites** ``_entangling_pairs`` with the scan
        result and stores the radius on the zone.  To have it flow into
        the final ``ArchSpec.blockade_radius``, either call
        :meth:`ArchBuilder.set_blockade_radius` (which applies to every
        zone and records the value at builder scope) or, for a single
        zone already set via ``ZoneBuilder.set_blockade_radius``,
        ``ArchBuilder.build()`` will pick up a consistent zone-level
        radius automatically.

        Args:
            radius: Blockade radius in micrometers. Must be positive and
                representable at 1 nm precision.

        Raises:
            ValueError: if the layout is inconsistent with the radius
                (partial blockade / crossed-index / multi-partner) or
                if ``radius`` is not positive / nm-precise.
        """
        if radius <= 0:
            raise ValueError(f"blockade_radius must be positive, got {radius}")
        radius_nm = _to_nm(radius, "blockade_radius")
        self._entangling_pairs = self._scan_blockade_pairs(radius_nm)
        self._blockade_radius_nm = radius_nm

    def _site_nm(self, word_id: int, site_id: int) -> tuple[int, int]:
        """Physical (x, y) position of a site, in nm integers."""
        x_idx, y_idx = self._words[word_id][site_id]
        return (
            _to_nm(self._grid.x_positions[x_idx], "grid x-position"),
            _to_nm(self._grid.y_positions[y_idx], "grid y-position"),
        )

    def _scan_blockade_pairs(self, radius_nm: int) -> list[tuple[int, int]]:
        """Scan word pairs and classify under the matching-index CZ rule.

        Returns the list of valid ``(a, b)`` word pairs with ``a < b``.
        Raises ``ValueError`` on partial blockade, crossed-index, or
        multi-partner cases.

        Uses ``scipy.spatial.KDTree.query_pairs`` to enumerate only the
        site pairs within ``radius_nm`` (O(n log n + k) over all sites
        in the zone, rather than O(n² · spw²) all-to-all).  Coordinates
        are fed in as nm integers so the ``<= radius_nm`` cutoff lands
        on exact boundaries without float drift.
        """
        n = self.num_words
        spw = self.sites_per_word
        if n < 2:
            return []

        from scipy.spatial import KDTree

        # Flatten every site into one KDTree, tracking (word, site_index)
        # so we can classify each returned pair.
        positions: list[tuple[int, int]] = []
        owners: list[tuple[int, int]] = []
        for w in range(n):
            for s in range(spw):
                positions.append(self._site_nm(w, s))
                owners.append((w, s))

        tree = KDTree(positions)
        # query_pairs returns (i, j) with i < j and dist(p_i, p_j) <= radius_nm.
        raw_pairs = tree.query_pairs(radius_nm, output_type="set")

        # For each cross-word pair within the radius, record which matching
        # site-indices fell within (and bail immediately on crossed-index).
        matching_sites: dict[tuple[int, int], set[int]] = {}
        radius_um = radius_nm / _NM_PER_UM
        for i, j in raw_pairs:
            w1, s1 = owners[i]
            w2, s2 = owners[j]
            if w1 == w2:
                # Two sites of the same word — not a CZ pair relationship.
                continue
            # Canonicalize (a, b) with a < b; track which site-index of `a`
            # matches which of `b`.
            if w1 < w2:
                a, sa, b, sb = w1, s1, w2, s2
            else:
                a, sa, b, sb = w2, s2, w1, s1
            if sa != sb:
                raise ValueError(
                    f"Zone '{self._name}' blockade scan: words "
                    f"{a} and {b} have a non-matching-index site pair "
                    f"(site {sa} ↔ site {sb}) within radius "
                    f"{radius_um} µm (crossed-index blockade). The "
                    f"layout cannot be cleanly paired under the CZ "
                    f"matching-index convention."
                )
            matching_sites.setdefault((a, b), set()).add(sa)

        # A clean CZ pair has all `spw` matching-index site-pairs within
        # radius; fewer is a partial blockade.
        valid_pairs: list[tuple[int, int]] = []
        for (a, b), sites in matching_sites.items():
            if len(sites) != spw:
                raise ValueError(
                    f"Zone '{self._name}' blockade scan: words "
                    f"{a} and {b} have {len(sites)}/{spw} "
                    f"matching-index site pairs within radius "
                    f"{radius_um} µm (partial blockade). The layout "
                    f"cannot be cleanly paired under the CZ "
                    f"matching-index convention."
                )
            valid_pairs.append((a, b))

        # Deterministic order (`query_pairs` returns a set).
        valid_pairs.sort()

        # Each word must appear in at most one valid pair.
        partner: dict[int, int] = {}
        for a, b in valid_pairs:
            for x, y in ((a, b), (b, a)):
                if x in partner and partner[x] != y:
                    raise ValueError(
                        f"Zone '{self._name}' blockade scan: word {x} "
                        f"has multiple blockade partners: "
                        f"{partner[x]}, {y}. Tighten blockade_radius "
                        f"or adjust the word layout."
                    )
                partner[x] = y

        return valid_pairs

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
        self._connections: list[
            tuple[tuple[str, Sequence[int]], tuple[str, Sequence[int]]]
        ] = []
        self._modes: list[tuple[str, list[str]]] = []
        self._total_words: int = 0
        self._blockade_radius: float | None = None

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
        src: tuple[str, Sequence[int]],
        dst: tuple[str, Sequence[int]],
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

    def add_mode(self, name: str, zones: Sequence[str]) -> None:
        """Add an operational mode.

        Args:
            name: Mode name (e.g. "all", "gate", "measure").
            zones: Zone names to include in this mode.
        """
        for z in zones:
            if z not in self._zone_name_to_id:
                raise ValueError(f"Unknown zone: '{z}'")
        self._modes.append((name, list(zones)))

    def set_blockade_radius(self, radius: float) -> None:
        """Apply ``radius`` to every zone by calling
        :meth:`ZoneBuilder.set_blockade_radius` on each.

        Overwrites every zone's entangling pairs with the scan result.
        The radius is stored on the builder and flows to
        ``ArchSpec.blockade_radius`` at :meth:`build` time.

        The radius is validated up-front (positive, finite, nm-precise)
        *before* any zone is touched, and the scan is run two-phase:
        every zone is scanned before any pair list is overwritten, so
        a layout error in a later zone cannot leave earlier zones in a
        partially-updated state.

        Args:
            radius: Rydberg blockade radius in micrometers.

        Raises:
            ValueError: if ``radius`` itself is invalid (non-positive,
                non-finite, sub-nm), or if any zone's layout is
                inconsistent with the radius. The error message
                includes the zone name and offending word IDs.
        """
        if radius <= 0:
            raise ValueError(f"blockade_radius must be positive, got {radius}")
        radius_nm = _to_nm(radius, "blockade_radius")

        # Phase 1: scan every zone.  Any zone-level failure raises here
        # before we've mutated anything.
        scan_results: list[list[tuple[int, int]]] = [
            zone._scan_blockade_pairs(radius_nm) for zone in self._zones
        ]

        # Phase 2: commit.  No further failures possible.
        for zone, pairs in zip(self._zones, scan_results):
            zone._entangling_pairs = pairs
            zone._blockade_radius_nm = radius_nm
        self._blockade_radius = radius

    @property
    def blockade_radius(self) -> float | None:
        """Rydberg blockade radius (µm) applied to all zones, or None."""
        return self._blockade_radius

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
        # Zone-local word IDs must be translated to global IDs for the Rust
        # ArchSpec, which uses global word indices everywhere.
        rust_zones: list[_RustZone] = []
        for zone_idx, zone in enumerate(self._zones):
            offset = self._word_id_offsets[zone_idx]
            site_buses = [_RustSiteBus(src=s, dst=d) for s, d in zone._site_buses]
            word_buses = [
                _RustWordBus(
                    src=[offset + w for w in s],
                    dst=[offset + w for w in d],
                )
                for s, d in zone._word_buses
            ]
            words_with_site_buses = (
                [offset + w for w in range(zone.num_words)] if site_buses else []
            )
            sites_with_word_buses = (
                list(range(zone.sites_per_word)) if word_buses else []
            )
            entangling_pairs = [
                (offset + a, offset + b) for a, b in zone._entangling_pairs
            ]
            rust_zones.append(
                _RustZone(
                    name=zone.name,
                    grid=zone._grid,
                    site_buses=site_buses,
                    word_buses=word_buses,
                    words_with_site_buses=words_with_site_buses,
                    sites_with_word_buses=sites_with_word_buses,
                    entangling_pairs=entangling_pairs,
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

        # 5. Determine the blockade radius to record on the ArchSpec.
        # Builder-level radius (set via ArchBuilder.set_blockade_radius)
        # takes precedence.  Otherwise, pick up a zone-level radius if
        # every zone with a radius agrees on the value (if some zones
        # have a radius and others don't, or zones disagree, error out
        # — the single-spec blockade_radius field can't represent that).
        blockade_radius = self._resolve_blockade_radius()

        # 6. Assemble and validate.
        return ArchSpec.from_components(
            words=tuple(all_words),
            zones=tuple(rust_zones),
            modes=modes,
            zone_buses=zone_buses,
            blockade_radius=blockade_radius,
        )

    def _resolve_blockade_radius(self) -> float | None:
        """Pick the blockade_radius value for the final ArchSpec.

        Precedence:

        1. ``self._blockade_radius`` (builder-scope value from
           :meth:`set_blockade_radius`) — always authoritative when set.
        2. A unique radius shared by all zones that have one set; zones
           without a radius must also agree (i.e., no zone opt-out) for
           this branch to apply.
        3. ``None`` otherwise.

        Raises ``ValueError`` if zones disagree on a radius or if some
        zones have a radius set and others don't.
        """
        if self._blockade_radius is not None:
            return self._blockade_radius
        zone_radii = [zone.blockade_radius for zone in self._zones]
        if all(r is None for r in zone_radii):
            return None
        missing = [z.name for z, r in zip(self._zones, zone_radii) if r is None]
        if missing:
            raise ValueError(
                "blockade_radius is set on some zones but not others; "
                f"missing on: {missing}. Either call "
                "ArchBuilder.set_blockade_radius to apply uniformly, "
                "or leave it unset on every zone."
            )
        # Every zone has a non-None radius here.
        present = [r for r in zone_radii if r is not None]
        unique = sorted(set(present))
        if len(unique) > 1:
            raise ValueError(
                f"Zones disagree on blockade_radius: {unique}. "
                "The ArchSpec can only carry a single value."
            )
        return unique[0]
