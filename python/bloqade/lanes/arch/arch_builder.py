"""Zone-aware architecture builder.

Provides ``ZoneBuilder`` (single-zone construction with validation) and
``ArchBuilder`` (multi-zone composition) as the low-level building blocks.
The high-level ``build_arch()`` function uses these internally.
"""

from __future__ import annotations

import math
import warnings
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import rustworkx as rx

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
from bloqade.lanes.layout.encoding import Direction, LaneAddress, MoveType
from bloqade.lanes.layout.word import Word

if TYPE_CHECKING:
    pass


# ── Helpers ──

# Internal length unit: 1 nm.  All user-facing lengths are in µm;
# internally we convert to nm integers so that path search (hashing,
# set membership, equality) and blockade-radius distance comparisons
# are exact without floating-point hazards.
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
    """Query word indices by grid region on a ZoneBuilder.

    Returns a plain ``list[int]`` of zone-local word IDs for intra-zone
    operations like ``add_word_bus`` and ``add_entangling_pairs``.  For
    cross-zone operations that need a name-qualified reference (e.g.,
    ``ArchBuilder.connect``), use ``zone[...]`` directly — that form
    returns ``(name, list[int])``.
    """

    def __init__(self, zone: ZoneBuilder):
        self._zone = zone

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> list[int]:
        x_idx, y_idx = key
        xs = _normalize_index(x_idx, self._zone._grid.num_x)
        ys = _normalize_index(y_idx, self._zone._grid.num_y)
        query_positions = {(x, y) for x in xs for y in ys}
        hits: set[int] = set()
        for pos, word_id in self._zone._position_to_word.items():
            if pos in query_positions:
                hits.add(word_id)
        return sorted(hits)


# ── ZoneBuilder ──


class ZoneBuilder:
    """Build a single zone with its words, grid, and buses.

    All indices are zone-local. Words are placed on the zone's grid
    and validated for shape and overlap. Buses are validated for AOD
    Cartesian product compliance.
    """

    def __init__(
        self,
        name: str,
        grid: _RustGrid,
        word_shape: tuple[int, int],
        *,
        x_clearance: float,
        y_clearance: float,
    ):
        """Initialize a zone.

        Args:
            name: Human-readable zone name (stored in Rust Zone).
            grid: Coordinate grid for this zone.  Every x- and y-position
                must be representable at 1 nm precision (i.e., at most 3
                decimal places when given in µm).
            word_shape: (num_x_sites, num_y_sites) — uniform shape for
                all words in this zone. sites_per_word = product of shape.
            x_clearance: Minimum physical distance (> 0, µm) from grid
                lines that path waypoints must maintain on the x-axis.
                Must be representable at 1 nm precision.
            y_clearance: Same as ``x_clearance``, applied to the y-axis.
                Allowing separate values is useful when row and column
                spacings differ substantially (e.g., tight intra-pair x
                spacing but wide row spacing).

        Raises:
            ValueError: If either clearance is not positive or any
                position / clearance value is not nm-precise.
        """
        if x_clearance <= 0:
            raise ValueError(f"x_clearance must be positive, got {x_clearance}")
        if y_clearance <= 0:
            raise ValueError(f"y_clearance must be positive, got {y_clearance}")
        self._name = name
        self._grid = grid
        self._word_shape = word_shape
        # Internal nm-integer representation for exact path search.
        self._x_clearance_nm = _to_nm(x_clearance, "x_clearance")
        self._y_clearance_nm = _to_nm(y_clearance, "y_clearance")
        self._grid_x_nm: list[int] = [
            _to_nm(x, "grid x-position") for x in grid.x_positions
        ]
        self._grid_y_nm: list[int] = [
            _to_nm(y, "grid y-position") for y in grid.y_positions
        ]
        self._words: list[list[tuple[int, int]]] = []
        self._word_has_site_bus: list[bool] = []
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
    def x_clearance(self) -> float:
        """Minimum x-axis clearance (µm) from grid lines for waypoints."""
        return self._x_clearance_nm / _NM_PER_UM

    @property
    def y_clearance(self) -> float:
        """Minimum y-axis clearance (µm) from grid lines for waypoints."""
        return self._y_clearance_nm / _NM_PER_UM

    @property
    def num_words(self) -> int:
        """Number of words added so far."""
        return len(self._words)

    def add_word(
        self,
        x_sites: slice | Sequence[int],
        y_sites: slice | Sequence[int],
        *,
        has_site_bus: bool = True,
    ) -> int:
        """Add a word occupying the given grid positions.

        The number of x-indices and y-indices must match word_shape.
        Grid positions must not overlap with any existing word.

        Args:
            x_sites: Grid x-indices for the word's sites.
            y_sites: Grid y-indices for the word's sites.
            has_site_bus: Whether this word participates in site-bus
                transport. Feeds the zone-level
                ``words_with_site_buses`` list on the final ``ArchSpec``
                — only words with ``has_site_bus=True`` are eligible to
                have site buses applied to them. Defaults to ``True``,
                which preserves the historical "all words opt-in"
                behavior. Set to ``False`` on storage words that
                shouldn't participate in site-level routing.

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
        self._word_has_site_bus.append(has_site_bus)
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
        if n * spw < 2:
            # Need at least two sites anywhere in the zone before a pair
            # can even exist; skip the KDTree build.
            return []

        from scipy.spatial import KDTree

        # Flatten every site into one KDTree, tracking (word, site_index)
        # so we can classify each returned pair.
        positions = [self._site_nm(w, s) for w in range(n) for s in range(spw)]
        owners = [(w, s) for w in range(n) for s in range(spw)]

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
                # Two sites of the same word are within blockade — the
                # layout can't support a CZ at this radius (it would
                # entangle atoms inside a single word).
                raise ValueError(
                    f"Zone '{self._name}' blockade scan: word {w1} "
                    f"has two intra-word sites ({s1} and {s2}) within "
                    f"radius {radius_um} µm. Entanglement within a "
                    f"single word is not allowed — tighten "
                    f"blockade_radius or space the word's sites apart."
                )
            if s1 != s2:
                raise ValueError(
                    f"Zone '{self._name}' blockade scan: words "
                    f"{min(w1, w2)} and {max(w1, w2)} have a "
                    f"non-matching-index site pair (site {s1} ↔ "
                    f"site {s2}) within radius {radius_um} µm "
                    f"(crossed-index blockade). The layout cannot "
                    f"be cleanly paired under the CZ matching-index "
                    f"convention."
                )
            # Matching-index pair. Canonicalize only the word pair.
            matching_sites.setdefault((min(w1, w2), max(w1, w2)), set()).add(s1)

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
        """Query word indices by grid region for intra-zone use.

        Returns a plain ``list[int]`` of zone-local word indices whose
        sites intersect the queried region — suitable for passing
        directly to ``add_word_bus`` / ``add_entangling_pairs``.

        For cross-zone references (e.g. ``ArchBuilder.connect``), index
        the zone itself (``zone[region]``) to get a name-qualified
        ``(name, list[int])`` tuple.
        """
        return _WordGridQuery(self)

    def __getitem__(
        self, key: tuple[slice | int | Sequence[int], slice | int | Sequence[int]]
    ) -> tuple[str, list[int]]:
        """Query word indices by grid region, name-qualified for cross-zone use.

        Returns ``(self.name, zone.words[key])`` so the result can be
        passed directly to ``ArchBuilder.connect(src=..., dst=...)``,
        which expects ``(zone_name, zone_local_indices)``.

        For intra-zone use (passing indices to ``add_word_bus`` etc.),
        use ``zone.words[key]`` which returns just the index list.
        """
        return (self._name, self.words[key])

    @property
    def sites(self) -> _SiteGridQuery:
        """Query site indices within the word shape."""
        return _SiteGridQuery(self._word_shape)

    def _word_origin(self, word_id: int) -> tuple[int, int]:
        """Get the (min_x, min_y) origin of a word for AOD validation."""
        positions = self._words[word_id]
        return (min(p[0] for p in positions), min(p[1] for p in positions))

    # ── Path computation ──
    #
    # All internal geometry is stored as nm-integer values so that set
    # membership, tuple equality, and candidate hashing are exact.  User-
    # facing APIs accept and return µm floats; conversion happens at the
    # boundary.

    def _site_nm(self, word_id: int, site_id: int) -> tuple[int, int]:
        """Physical (x, y) position of a site, in nm integers."""
        x_idx, y_idx = self._words[word_id][site_id]
        return (self._grid_x_nm[x_idx], self._grid_y_nm[y_idx])

    def _enumerate_safe_positions(
        self,
        grid_positions: list[int],
        source_positions: list[int],
        min_cl_nm: int,
    ) -> list[int]:
        """Enumerate reference positions on one axis that keep the bus clear.

        Returns sorted integer positions ``p`` (nm) such that, when the
        reference atom is at ``p`` and every other atom follows by the
        same shift (AOD invariant), every atom is at least ``min_cl_nm``
        from every grid line on this axis.

        Candidate positions are chosen to maximize clearance rather than
        sitting exactly on the ``min_cl_nm`` threshold:

        * **Midpoints** between consecutive grid lines.  When a shifted
          atom lands at a midpoint, its distance to the two neighboring
          grid lines is half the grid gap — which is typically larger
          than ``min_cl_nm`` on non-uniform grids, giving extra breathing
          room.
        * **Boundary edges** at ``min_grid - min_cl_nm`` and
          ``max_grid + min_cl_nm``, so the search can route around the
          outside of the grid when the interior is too crowded.

        All candidates are filtered by ``_valid`` (distance to every grid
        line, for every offset in the bus, must be ``>= min_cl_nm``).
        """
        if not source_positions:
            return []

        ref_src = source_positions[0]
        offsets = sorted({s - ref_src for s in source_positions})
        sorted_grid = sorted(set(grid_positions))

        candidates: set[int] = set()

        # Midpoints between consecutive grid lines, shifted per offset.
        # ``p = mid - off`` places the atom at offset ``off`` exactly at the
        # midpoint.
        for i in range(len(sorted_grid) - 1):
            mid = (sorted_grid[i] + sorted_grid[i + 1]) // 2
            for off in offsets:
                candidates.add(mid - off)

        # Outer boundary edges for routing around the grid.
        if sorted_grid:
            for off in offsets:
                candidates.add(sorted_grid[0] - off - min_cl_nm)
                candidates.add(sorted_grid[-1] - off + min_cl_nm)

        def _valid(p: int) -> bool:
            for off in offsets:
                shifted = p + off
                for g in grid_positions:
                    if abs(shifted - g) < min_cl_nm:
                        return False
            return True

        return sorted(c for c in candidates if _valid(c))

    def _search_path(
        self,
        ref_src: tuple[int, int],
        ref_dst: tuple[int, int],
        bus_src_positions: list[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...] | None:
        """Graph-based shortest path for a bus's reference atom, in nm-integer space.

        Builds a position graph where nodes are safe waypoint positions
        and edges are axis-aligned moves validated against bus-level grid
        crossings.  Dijkstra's algorithm finds the shortest-distance
        path, and a merge pass collapses consecutive same-axis segments.

        Returns a waypoint sequence ``[ref_src, ..., ref_dst]`` or
        ``None`` if no valid path exists.
        """
        src_xs = [p[0] for p in bus_src_positions]
        src_ys = [p[1] for p in bus_src_positions]
        safe_xs = set(
            self._enumerate_safe_positions(
                self._grid_x_nm, src_xs, self._x_clearance_nm
            )
        )
        safe_ys = set(
            self._enumerate_safe_positions(
                self._grid_y_nm, src_ys, self._y_clearance_nm
            )
        )

        x_candidates = sorted({ref_src[0], ref_dst[0], *safe_xs})
        y_candidates = sorted({ref_src[1], ref_dst[1], *safe_ys})

        offsets = [(p[0] - ref_src[0], p[1] - ref_src[1]) for p in bus_src_positions]
        grid_xs = self._grid_x_nm
        grid_ys = self._grid_y_nm
        grid_xs_set = set(grid_xs)
        grid_ys_set = set(grid_ys)

        # ── Build graph nodes ──
        # A node is valid if it is src/dst or a safe middle waypoint
        # (x in safe_xs OR y in safe_ys).
        pos_to_idx: dict[tuple[int, int], int] = {}
        idx_to_pos: list[tuple[int, int]] = []

        for x in x_candidates:
            x_safe = x in safe_xs
            for y in y_candidates:
                pos = (x, y)
                if pos == ref_src or pos == ref_dst or x_safe or y in safe_ys:
                    pos_to_idx[pos] = len(idx_to_pos)
                    idx_to_pos.append(pos)

        if ref_src not in pos_to_idx or ref_dst not in pos_to_idx:
            return None

        graph: rx.PyGraph = rx.PyGraph()
        graph.add_nodes_from(range(len(idx_to_pos)))

        # ── Build edges via blocking-position sweep ──
        # Group nodes by row (y) for horizontal edges, by column (x) for
        # vertical edges.
        rows: dict[int, list[int]] = defaultdict(list)
        cols: dict[int, list[int]] = defaultdict(list)
        for pos, idx in pos_to_idx.items():
            rows[pos[1]].append(idx)
            cols[pos[0]].append(idx)

        # Horizontal edges: for each row, compute blocking x-positions
        # from bus offsets, then connect adjacent candidates without a
        # blocker strictly between them.
        for y, node_indices in rows.items():
            # Blocking ref-x positions: grid x values shifted by -off_x
            # for each offset whose shifted y lands on a grid row.
            blockers: list[int] = []
            for off_x, off_y in offsets:
                if (y + off_y) in grid_ys_set:
                    for g_x in grid_xs:
                        blockers.append(g_x - off_x)
            sorted_blockers = sorted(set(blockers))

            # Sort nodes on this row by x-coordinate.
            node_indices.sort(key=lambda i: idx_to_pos[i][0])

            for a, b in zip(node_indices, node_indices[1:]):
                x_a = idx_to_pos[a][0]
                x_b = idx_to_pos[b][0]
                # Check if any blocker lies strictly between x_a and x_b.
                lo = bisect_right(sorted_blockers, x_a)
                hi = bisect_left(sorted_blockers, x_b)
                if lo >= hi:
                    # No blocker in (x_a, x_b) → safe edge.
                    graph.add_edge(a, b, x_b - x_a)

        # Vertical edges: same logic transposed.
        for x, node_indices in cols.items():
            blockers = []
            for off_x, off_y in offsets:
                if (x + off_x) in grid_xs_set:
                    for g_y in grid_ys:
                        blockers.append(g_y - off_y)
            sorted_blockers = sorted(set(blockers))

            node_indices.sort(key=lambda i: idx_to_pos[i][1])

            for a, b in zip(node_indices, node_indices[1:]):
                y_a = idx_to_pos[a][1]
                y_b = idx_to_pos[b][1]
                lo = bisect_right(sorted_blockers, y_a)
                hi = bisect_left(sorted_blockers, y_b)
                if lo >= hi:
                    graph.add_edge(a, b, y_b - y_a)

        # ── Dijkstra's shortest path ──
        src_idx = pos_to_idx[ref_src]
        dst_idx = pos_to_idx[ref_dst]

        paths = rx.dijkstra_shortest_paths(
            graph, src_idx, target=dst_idx, weight_fn=float
        )
        if dst_idx not in paths:
            return None

        raw_path = [idx_to_pos[i] for i in paths[dst_idx]]

        # ── Merge consecutive same-axis segments ──
        # Dijkstra's path uses fine-grained adjacent-candidate steps.
        # Collapse runs on the same axis where the direct segment does
        # not cross any grid atom for the bus.
        def _segment_safe(start: tuple[int, int], end: tuple[int, int]) -> bool:
            if start[1] == end[1]:
                for off_x, off_y in offsets:
                    if (start[1] + off_y) not in grid_ys_set:
                        continue
                    lo = min(start[0], end[0]) + off_x
                    hi = max(start[0], end[0]) + off_x
                    for g in grid_xs:
                        if lo < g < hi:
                            return False
            else:
                for off_x, off_y in offsets:
                    if (start[0] + off_x) not in grid_xs_set:
                        continue
                    lo = min(start[1], end[1]) + off_y
                    hi = max(start[1], end[1]) + off_y
                    for g in grid_ys:
                        if lo < g < hi:
                            return False
            return True

        merged = self._merge_collinear(raw_path, _segment_safe)
        return merged

    @staticmethod
    def _merge_collinear(
        path: list[tuple[int, int]],
        segment_safe: Callable[[tuple[int, int], tuple[int, int]], bool],
    ) -> tuple[tuple[int, int], ...]:
        """Collapse consecutive same-axis waypoints into longer segments.

        Walks the path and, for each axis-aligned run, extends the anchor
        to the farthest reachable point whose direct segment passes
        ``segment_safe``.  Positions within an axis run are monotonic
        (shortest-path guarantee), so once a merge is blocked all
        subsequent points on the same axis are also blocked.
        """
        if len(path) <= 2:
            return tuple(path)

        merged: list[tuple[int, int]] = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = i + 1
            # Determine which coordinate must stay constant (the axis).
            is_horizontal = path[j][1] == merged[-1][1]
            coord = 1 if is_horizontal else 0
            anchor_val = merged[-1][coord]

            # Extend as far as possible on the current axis.
            best = j
            for k in range(j + 1, len(path)):
                if path[k][coord] != anchor_val:
                    break
                if segment_safe(merged[-1], path[k]):
                    best = k
                else:
                    break

            merged.append(path[best])
            i = best

        return tuple(merged)

    def _apply_deltas(
        self,
        lane_src: tuple[int, int],
        ref_waypoints: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Build a lane's waypoint sequence by shifting the reference path."""
        ref_src = ref_waypoints[0]
        dx0 = lane_src[0] - ref_src[0]
        dy0 = lane_src[1] - ref_src[1]
        return tuple((w[0] + dx0, w[1] + dy0) for w in ref_waypoints)

    @staticmethod
    def _path_nm_to_um(
        path_nm: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[float, float], ...]:
        """Convert an nm-integer waypoint sequence to µm floats."""
        return tuple((w[0] / _NM_PER_UM, w[1] / _NM_PER_UM) for w in path_nm)

    def _compute_paths(
        self, zone_id: int, word_offset: int
    ) -> dict[LaneAddress, tuple[tuple[float, float], ...]]:
        """Compute axis-aligned AOD waypoint paths for all buses.

        Uses a DFS path search at the bus level over nm-integer positions.
        For each bus, the reference atom's path is searched; every other
        lane derives its waypoints by applying the same per-segment deltas.

        Site bus paths are intra-word; word bus paths are intra-zone.
        Zone buses are NOT included (inter-zone routing is separate).

        Returns:
            Dict mapping LaneAddress to waypoint tuples (µm floats) for
            both directions.
        """
        paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}

        # ── Site bus paths (intra-word) ──
        for bus_id, (src_sites, dst_sites) in enumerate(self._site_buses):
            # AOD invariant: every (src_site, dst_site) pair must have the
            # same physical displacement, because the AOD applies one
            # uniform delta per segment to the entire bus.
            displacements = {
                (
                    self._site_nm(0, ds)[0] - self._site_nm(0, ss)[0],
                    self._site_nm(0, ds)[1] - self._site_nm(0, ss)[1],
                )
                for ss, ds in zip(src_sites, dst_sites)
            }
            if len(displacements) > 1:
                warnings.warn(
                    f"Zone '{self._name}' site bus {bus_id}: inconsistent "
                    f"site displacements {sorted(displacements)} violate "
                    f"the AOD single-shift invariant. "
                    f"Skipping path generation for this bus.",
                    stacklevel=3,
                )
                continue

            bus_src_atoms = [
                self._site_nm(w, s) for w in range(self.num_words) for s in src_sites
            ]

            ref_src = self._site_nm(0, src_sites[0])
            ref_dst = self._site_nm(0, dst_sites[0])

            if ref_src == ref_dst:
                ref_waypoints: tuple[tuple[int, int], ...] = (ref_src, ref_dst)
            else:
                result = self._search_path(ref_src, ref_dst, bus_src_atoms)
                if result is None:
                    warnings.warn(
                        f"Zone '{self._name}' site bus {bus_id}: no valid "
                        f"path found (x_clearance={self.x_clearance}, "
                        f"y_clearance={self.y_clearance}). "
                        f"Skipping path generation for this bus.",
                        stacklevel=3,
                    )
                    continue
                ref_waypoints = result

            for local_word in range(self.num_words):
                for src_s in src_sites:
                    lane_src = self._site_nm(local_word, src_s)
                    lane_path_nm = self._apply_deltas(lane_src, ref_waypoints)
                    lane_path = self._path_nm_to_um(lane_path_nm)
                    global_word = word_offset + local_word
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        lane = LaneAddress(
                            MoveType.SITE,
                            global_word,
                            src_s,
                            bus_id,
                            direction,
                            zone_id,
                        )
                        paths[lane] = (
                            lane_path
                            if direction == Direction.FORWARD
                            else lane_path[::-1]
                        )

        # ── Word bus paths (intra-zone) ──
        for bus_id, (src_words, dst_words) in enumerate(self._word_buses):
            spw = range(self.sites_per_word)

            # AOD invariant: every (src_word, dst_word) pair must have
            # the same physical displacement.  Inconsistent spacings
            # (e.g., differing source vs destination grid layouts)
            # cannot be represented by a single uniform shift sequence.
            displacements = {
                (
                    self._site_nm(dw, 0)[0] - self._site_nm(sw, 0)[0],
                    self._site_nm(dw, 0)[1] - self._site_nm(sw, 0)[1],
                )
                for sw, dw in zip(src_words, dst_words)
            }
            if len(displacements) > 1:
                warnings.warn(
                    f"Zone '{self._name}' word bus {bus_id}: inconsistent "
                    f"word displacements {sorted(displacements)} violate "
                    f"the AOD single-shift invariant. "
                    f"Skipping path generation for this bus.",
                    stacklevel=3,
                )
                continue

            bus_src_atoms = [self._site_nm(w, s) for w in src_words for s in spw]

            ref_src = self._site_nm(src_words[0], 0)
            ref_dst = self._site_nm(dst_words[0], 0)

            if ref_src == ref_dst:
                ref_waypoints = (ref_src, ref_dst)
            else:
                result = self._search_path(ref_src, ref_dst, bus_src_atoms)
                if result is None:
                    warnings.warn(
                        f"Zone '{self._name}' word bus {bus_id}: no valid "
                        f"path found (x_clearance={self.x_clearance}, "
                        f"y_clearance={self.y_clearance}). "
                        f"Skipping path generation for this bus.",
                        stacklevel=3,
                    )
                    continue
                ref_waypoints = result

            for src_w in src_words:
                for site_id in spw:
                    lane_src = self._site_nm(src_w, site_id)
                    lane_path_nm = self._apply_deltas(lane_src, ref_waypoints)
                    lane_path = self._path_nm_to_um(lane_path_nm)
                    global_src = word_offset + src_w
                    for direction in (Direction.FORWARD, Direction.BACKWARD):
                        lane = LaneAddress(
                            MoveType.WORD,
                            global_src,
                            site_id,
                            bus_id,
                            direction,
                            zone_id,
                        )
                        paths[lane] = (
                            lane_path
                            if direction == Direction.FORWARD
                            else lane_path[::-1]
                        )

        return paths


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
                from ``zone[...]`` (indexing the zone itself, which
                returns a name-qualified tuple).
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

    def build(
        self,
        feed_forward: bool = False,
        atom_reloading: bool = False,
        blockade_radius: float | None = None,
    ) -> ArchSpec:
        """Assemble the ArchSpec and validate via Rust.

        Args:
            feed_forward: Whether the device supports feed-forward.
            atom_reloading: Whether the device supports atom reloading.
            blockade_radius: Explicit blockade radius (µm). If provided,
                overrides both builder-level and zone-level radii.

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
            # Only words flagged at add_word(has_site_bus=True) are
            # eligible for site-bus transport. Default is True, so the
            # historical "all words opt-in when any site bus exists"
            # behavior is preserved unless the caller overrides.
            words_with_site_buses = (
                [
                    offset + w
                    for w in range(zone.num_words)
                    if zone._word_has_site_bus[w]
                ]
                if site_buses
                else []
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

        # 5. Compute AOD waypoint paths for site and word buses.
        all_paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
        for zone_idx, zone in enumerate(self._zones):
            offset = self._word_id_offsets[zone_idx]
            all_paths.update(zone._compute_paths(zone_idx, offset))

        # 6. Determine the blockade radius to record on the ArchSpec.
        # Precedence: explicit build(blockade_radius=...) argument >
        # builder-level set_blockade_radius > zone-level agreement > None.
        resolved_radius = (
            blockade_radius
            if blockade_radius is not None
            else self._resolve_blockade_radius()
        )

        # 7. Assemble and validate.
        return ArchSpec.from_components(
            words=tuple(all_words),
            zones=tuple(rust_zones),
            modes=modes,
            zone_buses=zone_buses,
            paths=all_paths or None,
            feed_forward=feed_forward,
            atom_reloading=atom_reloading,
            blockade_radius=resolved_radius,
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
