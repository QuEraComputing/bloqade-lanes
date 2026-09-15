"""Spec-shaped architecture builder.

``ArchBuilder`` owns everything the :class:`ArchSpec` owns at spec scope —
the word template and the grid index space it indexes — and treats a zone
as what the spec says it is: a name, a set of coordinates for that shared
index space, and the buses that run on it.

The API is phased, and the phase order is what makes every check
decidable at the call that raises it:

1. **Words.**  ``add_word`` defines the spec-wide template.  A word is a
   set of grid ``(row, column)`` index pairs, so the template is
   independent of any zone's coordinates.  Every index-space API here is
   ``(rows, columns)``, row-major, including the ``words[...]`` and
   ``sites[...]`` selections.
2. **Zones.**  ``add_zone`` supplies coordinates for that index space,
   plus which words and sites take part in transport there.  The template
   is frozen from the first zone on, because a later word would change the
   occupancy and the participant set of every zone already added.
3. **Buses.**  Always zone-qualified.  By this point the carried-atom set
   of every bus is fully determined, so realizability is checked here
   rather than deferred to build time.
4. **Spec scope.**  Blockade radius, modes, capabilities, then ``build``.

See ``docs/superpowers/specs/2026-09-15-arch-builder-redesign-design.md``
for why the previous ``ZoneBuilder`` / ``ArchBuilder`` split could not be
made consistent.
"""

from __future__ import annotations

import operator
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from bloqade.lanes.bytecode._native import (
    Grid as _RustGrid,
    LocationAddress as _RustLocAddr,
    Mode as _RustMode,
    SiteBus as _RustSiteBus,
    WordBus as _RustWordBus,
    Zone as _RustZone,
    ZoneBus as _RustZoneBus,
)
from bloqade.lanes.arch.build._geometry import (
    ZoneGeometry,
    compute_transport_paths,
    scan_blockade_pairs,
    to_nm,
)
from bloqade.lanes.arch.build.v2._aod import check_aod_transport
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LaneAddress, MoveType
from bloqade.lanes.bytecode.word import Word

Index = slice | int | Sequence[int]


# ── Index normalization ──


def _query_axis(idx: Index, size: int, axis: str, what: str) -> list[int]:
    """Normalize one axis of a grid query, rejecting bad indices.

    The selection is returned **in the order given** — a sequence
    verbatim, a slice as it expands (including a negative step).  A query
    as a whole then behaves like ``arr[np.ix_(y_values, x_values)].ravel()``:
    the order within each axis is the caller's, and "row-major" describes
    only the nesting, with y as the outer loop.

    Out-of-range, negative and repeated indices raise rather than
    selecting nothing, so a typo cannot silently shorten a selection into
    a smaller-but-still-valid bus.  Slices are clamped by ``slice.indices``
    and are always valid.
    """
    if isinstance(idx, slice):
        return list(range(*idx.indices(size)))
    values = [idx] if isinstance(idx, int) else list(idx)
    out_of_range = sorted({i for i in values if not 0 <= i < size})
    if out_of_range:
        raise IndexError(f"{axis} {what} {out_of_range} out of range [0, {size})")
    repeated = sorted(i for i, n in Counter(values).items() if n > 1)
    if repeated:
        raise ValueError(
            f"{axis} {what} {repeated} repeated in {values}; each position is "
            "selected once, so a repeat cannot select anything new"
        )
    return values


def _checked_subset(values: Sequence[int], size: int, what: str) -> tuple[int, ...]:
    """Validate an explicit participation subset: in range and unique."""
    out = list(values)
    bad = sorted({v for v in out if not 0 <= v < size})
    if bad:
        raise IndexError(f"{what} {bad} out of range [0, {size})")
    repeated = sorted(v for v, n in Counter(out).items() if n > 1)
    if repeated:
        raise ValueError(f"{what} {repeated} repeated in {out}")
    return tuple(out)


# ── Template queries ──


class _SiteQuery:
    """Select site indices within the word shape, as ``[rows, columns]``.

    Traversal is row-major over the selected positions — rows the outer
    loop, columns the inner — with each axis visited in the order given.
    Site index is ``column + row * num_columns``.
    """

    def __init__(self, word_shape: tuple[int, int]):
        self._rows, self._cols = word_shape

    def __getitem__(self, key: tuple[Index, Index]) -> list[int]:
        row_idx, col_idx = key
        rows = _query_axis(row_idx, self._rows, "row", "site index")
        cols = _query_axis(col_idx, self._cols, "column", "site index")
        return [c + r * self._cols for r in rows for c in cols]


class _WordQuery:
    """Select word IDs by grid region, as ``[rows, columns]``.

    The selected positions are traversed row-major — rows the outer loop,
    columns the inner, each axis in the order given — and each word is
    taken at its first arrival, since a word spans several grid positions.

    Ordering by traversal rather than by word ID matters whenever word IDs
    are not monotone in the grid, which is the normal case for interleaved
    CZ layouts: scanning a row visits words 0, 1, 2, 3, 0, 1, 2, 3, ..., so
    a partial column selection reaches them out of ID order.  Because a bus's
    ``src`` and ``dst`` are related by a separable, order-preserving AOD
    transport, selecting both endpoints the same way pairs ``src[i]`` with
    ``dst[i]`` correctly.

    The query lives on the builder rather than on a zone because the
    template is spec-wide: the answer is the same for every zone.
    """

    def __init__(self, builder: ArchBuilder):
        self._builder = builder

    def __getitem__(self, key: tuple[Index, Index]) -> list[int]:
        row_idx, col_idx = key
        n_rows, n_cols = self._builder.grid_shape
        rows = _query_axis(row_idx, n_rows, "row", "grid index")
        cols = _query_axis(col_idx, n_cols, "column", "grid index")
        result: list[int] = []
        seen: set[int] = set()
        for r in rows:
            for c in cols:
                word_id = self._builder._position_to_word.get((c, r))
                if word_id is not None and word_id not in seen:
                    seen.add(word_id)
                    result.append(word_id)
        return result


# ── Zone ──


@dataclass
class _Zone:
    """One zone: coordinates for the shared index space, plus its buses.

    A plain value. It carries no validation logic, because every fact that
    decides whether its buses are legal is owned by the builder.
    """

    name: str
    grid: _RustGrid
    grid_x_nm: tuple[int, ...]
    grid_y_nm: tuple[int, ...]
    x_clearance_nm: int | None
    y_clearance_nm: int | None
    words_with_site_buses: tuple[int, ...]
    sites_with_word_buses: tuple[int, ...]
    site_buses: list[tuple[list[int], list[int]]] = field(default_factory=list)
    word_buses: list[tuple[list[int], list[int]]] = field(default_factory=list)
    entangling_pairs: list[tuple[int, int]] = field(default_factory=list)
    blockade_radius_nm: int | None = None


# ── ArchBuilder ──


class ArchBuilder:
    """Build an :class:`ArchSpec` from a shared word template and zones."""

    def __init__(
        self,
        grid_shape: tuple[int, int],
        word_shape: tuple[int, int],
    ):
        """Initialize the builder.

        Args:
            grid_shape: ``(num_rows, num_columns)`` size of the grid index
                space that word templates index into.  Every zone supplies
                exactly this many coordinates on each axis.
            word_shape: ``(num_rows, num_columns)`` of sites in every word;
                ``sites_per_word`` is their product.

        Raises:
            ValueError: If either shape is not a pair of positive integers.
        """

        def _shape(label: str, shape: tuple[int, int]) -> tuple[int, int]:
            # ``int(2.5)`` would silently accept a malformed index space, so
            # require values that are integers already.
            if len(shape) != 2:
                raise ValueError(f"{label} must be two positive ints, got {shape!r}")
            out = []
            for v in shape:
                if isinstance(v, bool) or not hasattr(v, "__index__"):
                    raise ValueError(
                        f"{label} must be two positive ints, got {shape!r}"
                    )
                i = operator.index(v)
                if i < 1:
                    raise ValueError(
                        f"{label} must be two positive ints, got {shape!r}"
                    )
                out.append(i)
            return (out[0], out[1])

        self._grid_shape = _shape("grid_shape", grid_shape)
        self._word_shape = _shape("word_shape", word_shape)
        self._words: list[list[tuple[int, int]]] = []
        self._position_to_word: dict[tuple[int, int], int] = {}
        self._zones: list[_Zone] = []
        self._zone_ids: dict[str, int] = {}
        self._connections: list[tuple[tuple[int, list[int]], tuple[int, list[int]]]] = (
            []
        )
        self._modes: list[tuple[str, list[str], list | None]] = []
        self._blockade_radius: float | None = None
        self._feed_forward = False
        self._atom_reloading = False
        # Paths carried in from an existing spec, plus the bus structure
        # they were computed for.  Bundled architectures ship hardware-
        # derived lane geometry that this path search does not reproduce,
        # and lane durations (hence fidelity estimates) are read off those
        # segment lengths — so a round-trip must not silently replace them.
        self._inherited_paths: (
            dict[LaneAddress, tuple[tuple[float, float], ...]] | None
        ) = None
        # Per zone: (n_site_buses, n_word_buses, movers, carried) as of
        # from_spec, so lanes for buses that existed then stay inheritable
        # even after new buses are appended.
        self._inherited_shape: dict[
            str, tuple[int, int, tuple[int, ...], tuple[int, ...]]
        ] = {}

    # ── Shape ──

    @property
    def grid_shape(self) -> tuple[int, int]:
        """``(num_rows, num_columns)`` of the shared grid index space."""
        return self._grid_shape

    @property
    def word_shape(self) -> tuple[int, int]:
        """``(num_rows, num_columns)`` of sites in every word."""
        return self._word_shape

    @property
    def sites_per_word(self) -> int:
        """Total sites per word."""
        return self._word_shape[0] * self._word_shape[1]

    @property
    def num_words(self) -> int:
        """Words defined in the template so far."""
        return len(self._words)

    @property
    def num_zones(self) -> int:
        """Zones added so far."""
        return len(self._zones)

    @property
    def words(self) -> _WordQuery:
        """Select word IDs by ``[rows, columns]`` of the grid index space."""
        return _WordQuery(self)

    @property
    def sites(self) -> _SiteQuery:
        """Select site indices by ``[rows, columns]`` of the word shape."""
        return _SiteQuery(self._word_shape)

    # ── Phase 1: the word template ──

    def add_word(self, rows: Index, columns: Index) -> int:
        """Add a word to the spec-wide template.

        Args:
            rows: Grid row indices for the word's sites; must number
                ``word_shape[0]``.
            columns: Grid column indices; must number ``word_shape[1]``.

        Returns:
            The new word ID.

        Raises:
            ValueError: If the index count does not match ``word_shape``,
                an index repeats, a grid position is already taken by
                another word, or a zone has already been added.
            IndexError: If an index falls outside ``grid_shape``.
        """
        if self._zones:
            raise ValueError(
                f"the word template is frozen: {len(self._zones)} zone(s) "
                "already resolve it against their own coordinates, so adding "
                "a word now would change their occupancy and their transport "
                "participants. Add every word before the first add_zone."
            )
        n_rows, n_cols = self._grid_shape
        rs = _query_axis(rows, n_rows, "row", "grid index")
        cs = _query_axis(columns, n_cols, "column", "grid index")
        if len(rs) != self._word_shape[0]:
            raise ValueError(
                f"rows has {len(rs)} indices but word_shape requires "
                f"{self._word_shape[0]}"
            )
        if len(cs) != self._word_shape[1]:
            raise ValueError(
                f"columns has {len(cs)} indices but word_shape requires "
                f"{self._word_shape[1]}"
            )

        # Stored in the spec's own ``(x_idx, y_idx)`` order, which is
        # ``(column, row)``; the row-major traversal is what fixes site IDs.
        positions = [(c, r) for r in rs for c in cs]
        for pos in positions:
            owner = self._position_to_word.get(pos)
            if owner is not None:
                raise ValueError(
                    f"grid position (row={pos[1]}, column={pos[0]}) already "
                    f"belongs to word {owner}"
                )

        word_id = len(self._words)
        self._words.append(positions)
        for pos in positions:
            self._position_to_word[pos] = word_id
        return word_id

    # ── Phase 2: zones ──

    def add_zone(
        self,
        name: str,
        rows: Sequence[float],
        columns: Sequence[float],
        *,
        x_clearance: float | None,
        y_clearance: float | None,
        words_with_site_buses: Sequence[int] | None = None,
        sites_with_word_buses: Sequence[int] | None = None,
    ) -> int:
        """Add a zone: coordinates for the shared index space.

        Args:
            name: Zone name; must be unique.
            rows: ``grid_shape[0]`` row positions — the y-coordinate of
                each grid row, in µm, each representable at 1 nm precision.
            columns: ``grid_shape[1]`` column positions — the x-coordinate
                of each grid column, in µm.
            x_clearance: Minimum x-axis distance (> 0, µm) that path
                waypoints keep from every grid line.  ``None`` only when
                paths are inherited from an existing spec via
                :meth:`from_spec`, since the search cannot run without it.
            y_clearance: Same, for the y-axis.
            words_with_site_buses: Words eligible for site-bus transport in
                this zone.  Defaults to every word.
            sites_with_word_buses: Sites carried by this zone's word buses.
                Defaults to every site.

        Returns:
            The new zone ID.

        Raises:
            ValueError: On a duplicate name, a coordinate count that does
                not match ``grid_shape``, a non-positive or non-nm-precise
                clearance, no words defined yet, or a repeated
                participation entry.
            IndexError: If a participation entry is out of range.
        """
        if not self._words:
            raise ValueError(
                "define the word template with add_word before adding a zone; "
                "a zone resolves that template against its own coordinates."
            )
        if name in self._zone_ids:
            raise ValueError(f"duplicate zone name: '{name}'")
        n_rows, n_cols = self._grid_shape
        if len(rows) != n_rows or len(columns) != n_cols:
            raise ValueError(
                f"zone '{name}' supplies {len(rows)} rows x {len(columns)} "
                f"columns but grid_shape is {n_rows}x{n_cols}; every zone "
                "indexes the same shared grid index space."
            )
        if x_clearance is not None and x_clearance <= 0:
            raise ValueError(f"x_clearance must be positive, got {x_clearance}")
        if y_clearance is not None and y_clearance <= 0:
            raise ValueError(f"y_clearance must be positive, got {y_clearance}")

        movers = (
            tuple(range(self.num_words))
            if words_with_site_buses is None
            else _checked_subset(
                words_with_site_buses, self.num_words, "words_with_site_buses"
            )
        )
        carried = (
            tuple(range(self.sites_per_word))
            if sites_with_word_buses is None
            else _checked_subset(
                sites_with_word_buses, self.sites_per_word, "sites_with_word_buses"
            )
        )

        zone = _Zone(
            name=name,
            grid=_RustGrid.from_positions(list(columns), list(rows)),
            grid_x_nm=tuple(to_nm(v, "column position") for v in columns),
            grid_y_nm=tuple(to_nm(v, "row position") for v in rows),
            x_clearance_nm=(
                None if x_clearance is None else to_nm(x_clearance, "x_clearance")
            ),
            y_clearance_nm=(
                None if y_clearance is None else to_nm(y_clearance, "y_clearance")
            ),
            words_with_site_buses=movers,
            sites_with_word_buses=carried,
        )
        zone_id = len(self._zones)
        self._zone_ids[name] = zone_id
        self._zones.append(zone)
        return zone_id

    # ── Phase 3: buses ──

    def add_site_bus(self, zone: str, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a site bus (intra-word movement) to one zone.

        The atoms it carries are every participating word's ``src`` sites,
        which is fixed once the template is frozen and the zone declares
        its participants — so realizability is decided here.

        Args:
            zone: Zone name.
            src: Source site indices within the word shape.
            dst: Destination site indices, parallel to ``src``.

        Raises:
            ValueError: On mismatched or empty sequences, an out-of-range
                site index, or a transport no AOD can perform.
        """
        z = self._zone(zone)
        self._check_bus_endpoints(src, dst, self.sites_per_word, "site")
        if not z.words_with_site_buses:
            raise ValueError(
                f"zone '{zone}' has no words_with_site_buses, so a site bus "
                "there would carry nothing. Declare the participating words "
                "on add_zone."
            )
        bus_id = len(z.site_buses)
        self._reject_duplicate_bus(
            z.site_buses, src, dst, "site", f"site bus {bus_id} on zone '{zone}'"
        )
        geom = self._geometry(z)
        check_aod_transport(
            [
                (geom.site_nm(w, s), geom.site_nm(w, d))
                for w in z.words_with_site_buses
                for s, d in zip(src, dst)
            ]
            or [],
            occupied=self._occupied(geom),
            label=f"site bus {bus_id} on zone '{zone}'",
        )
        z.site_buses.append((list(src), list(dst)))

    def _reject_duplicate_bus(
        self,
        existing: Sequence[tuple[list[int], list[int]]],
        src: Sequence[int],
        dst: Sequence[int],
        kind: str,
        label: str,
    ) -> None:
        """Reject a bus that repeats an existing ordered mapping.

        A bus's index is its ``bus_id`` and every lane address is built
        from it, so a duplicate adds a second set of lanes that move the
        same atoms along the same route — wasted lanes that also make
        ``bus_id`` ambiguous to anything reasoning about the spec.
        """
        pair = (list(src), list(dst))
        for i, (e_src, e_dst) in enumerate(existing):
            if [list(e_src), list(e_dst)] == list(pair):
                raise ValueError(
                    f"{label}: {kind} bus {i} already has this exact ordered "
                    "src/dst mapping. Buses are addressed by index, so a "
                    "duplicate only adds redundant lanes."
                )

    def add_word_bus(self, zone: str, src: Sequence[int], dst: Sequence[int]) -> None:
        """Add a word bus (intra-zone movement) to one zone.

        Args:
            zone: Zone name.
            src: Source word IDs.
            dst: Destination word IDs, parallel to ``src``.

        Raises:
            ValueError: On mismatched or empty sequences, an out-of-range
                word ID, or a transport no AOD can perform.
        """
        z = self._zone(zone)
        self._check_bus_endpoints(src, dst, self.num_words, "word")
        if not z.sites_with_word_buses:
            raise ValueError(
                f"zone '{zone}' has no sites_with_word_buses, so a word bus "
                "there would carry nothing. Declare the participating sites "
                "on add_zone."
            )
        bus_id = len(z.word_buses)
        self._reject_duplicate_bus(
            z.word_buses, src, dst, "word", f"word bus {bus_id} on zone '{zone}'"
        )
        geom = self._geometry(z)
        check_aod_transport(
            [
                (geom.site_nm(sw, s), geom.site_nm(dw, s))
                for sw, dw in zip(src, dst)
                for s in z.sites_with_word_buses
            ],
            occupied=self._occupied(geom),
            label=f"word bus {bus_id} on zone '{zone}'",
        )
        z.word_buses.append((list(src), list(dst)))

    def connect(
        self,
        src: tuple[str, Sequence[int]],
        dst: tuple[str, Sequence[int]],
    ) -> None:
        """Add an inter-zone word bus.

        Args:
            src: ``(zone_name, word_ids)`` for the source endpoint.
            dst: ``(zone_name, word_ids)`` for the destination endpoint.

        Raises:
            ValueError: On an unknown zone, mismatched or empty sequences,
                or a transport no AOD can perform.
            IndexError: If a word ID is out of range.
        """
        src_name, src_words = src
        dst_name, dst_words = dst
        src_zone = self._zone(src_name)
        dst_zone = self._zone(dst_name)
        if src_name == dst_name:
            raise ValueError(
                f"connect adds an inter-zone bus, but '{src_name}' is both "
                "endpoints. Use add_word_bus for movement inside one zone."
            )
        self._check_bus_endpoints(src_words, dst_words, self.num_words, "word")

        src_geom = self._geometry(src_zone)
        dst_geom = self._geometry(dst_zone)
        bus_id = len(self._connections)
        check_aod_transport(
            [
                (src_geom.site_nm(sw, s), dst_geom.site_nm(dw, s))
                for sw, dw in zip(src_words, dst_words)
                for s in range(self.sites_per_word)
            ],
            occupied=self._occupied(src_geom),
            label=f"zone bus {bus_id} from '{src_name}' to '{dst_name}'",
        )
        self._connections.append(
            (
                (self._zone_ids[src_name], list(src_words)),
                (self._zone_ids[dst_name], list(dst_words)),
            )
        )

    # ── Phase 4: spec scope ──

    def add_entangling_pairs(
        self, zone: str, words_a: Sequence[int], words_b: Sequence[int]
    ) -> None:
        """Mark paired words as CZ pairs within one zone.

        ``words_a[i]`` is paired with ``words_b[i]``.  Any blockade radius
        previously recorded on this zone is cleared, since a manual append
        means the pair list is no longer purely radius-derived.
        """
        z = self._zone(zone)
        if len(words_a) != len(words_b):
            raise ValueError(
                f"words_a has {len(words_a)} entries but words_b has " f"{len(words_b)}"
            )
        for a, b in zip(words_a, words_b):
            for w in (a, b):
                if not 0 <= w < self.num_words:
                    raise IndexError(
                        f"word index {w} out of range [0, {self.num_words})"
                    )
            z.entangling_pairs.append((a, b))
        z.blockade_radius_nm = None

    def set_blockade_radius(self, radius: float) -> None:
        """Derive entangling word pairs in every zone from the radius.

        The radius is a spec-scope value, but the pairs it implies are
        per-zone: the same radius against different coordinates yields
        different pairs.  Every zone is scanned before any pair list is
        overwritten, so a layout error in a later zone cannot leave earlier
        zones partially updated.

        Raises:
            ValueError: If ``radius`` is not positive or nm-precise, or if
                any zone's layout is inconsistent with it.
        """
        if radius <= 0:
            raise ValueError(f"blockade_radius must be positive, got {radius}")
        radius_nm = to_nm(radius, "blockade_radius")
        scanned = [
            scan_blockade_pairs(self._geometry(z), radius_nm) for z in self._zones
        ]
        for z, pairs in zip(self._zones, scanned):
            z.entangling_pairs = pairs
            z.blockade_radius_nm = radius_nm
        self._blockade_radius = radius

    @property
    def blockade_radius(self) -> float | None:
        """Blockade radius (µm) applied to every zone, or ``None``."""
        return self._blockade_radius

    def add_mode(
        self,
        name: str,
        zones: Sequence[str],
        *,
        bitstring_order: Sequence[_RustLocAddr] | None = None,
    ) -> None:
        """Add an operational mode over a subset of zones.

        Args:
            name: Mode name.
            zones: Zone names active in this mode.
            bitstring_order: Explicit measurement bit ordering.  Defaults to
                every ``(zone, word, site)`` of the listed zones in order;
                an ``ArchSpec`` may carry any ordering, so :meth:`from_spec`
                passes the original through rather than regenerating it.
        """
        for z in zones:
            if z not in self._zone_ids:
                raise ValueError(f"unknown zone: '{z}'")
        self._modes.append(
            (
                name,
                list(zones),
                None if bitstring_order is None else list(bitstring_order),
            )
        )

    def build(
        self,
        *,
        feed_forward: bool | None = None,
        atom_reloading: bool | None = None,
        blockade_radius: float | None = None,
        recompute_paths: bool = False,
    ) -> ArchSpec:
        """Assemble the :class:`ArchSpec` and validate it via Rust.

        Args:
            feed_forward: Whether the device supports feed-forward.  Defaults
                to the value carried in by :meth:`from_spec`, else ``False``.
            atom_reloading: Whether the device supports atom reloading, with
                the same default.
            blockade_radius: Explicit radius (µm), overriding the value
                recorded by :meth:`set_blockade_radius`.
            recompute_paths: Search every lane afresh, discarding inherited
                paths even for buses that predate any edit.  Off by default,
                so neither a round-trip nor an extension silently replaces
                hardware-derived lane geometry.

        Raises:
            ValueError: If no words or zones were defined, or if Rust
                validation fails.
        """
        if not self._words:
            raise ValueError("no words defined; call add_word before build")
        if not self._zones:
            raise ValueError("no zones defined; call add_zone before build")

        words = tuple(Word(tuple(positions)) for positions in self._words)

        rust_zones = [
            _RustZone(
                name=z.name,
                grid=z.grid,
                site_buses=[_RustSiteBus(src=s, dst=d) for s, d in z.site_buses],
                word_buses=[
                    _RustWordBus(src=list(s), dst=list(d)) for s, d in z.word_buses
                ],
                words_with_site_buses=(
                    list(z.words_with_site_buses) if z.site_buses else []
                ),
                sites_with_word_buses=(
                    list(z.sites_with_word_buses) if z.word_buses else []
                ),
                entangling_pairs=list(z.entangling_pairs),
            )
            for z in self._zones
        ]

        zone_buses = [
            _RustZoneBus(
                src=[(src_zid, w) for w in src_words],
                dst=[(dst_zid, w) for w in dst_words],
            )
            for (src_zid, src_words), (dst_zid, dst_words) in self._connections
        ]

        modes: list[_RustMode] = []
        for mode_name, zone_names, order in self._modes:
            zone_ids = [self._zone_ids[z] for z in zone_names]
            modes.append(
                _RustMode(
                    name=mode_name,
                    zones=zone_ids,
                    bitstring_order=(
                        list(order)
                        if order is not None
                        else [
                            _RustLocAddr(zid, w, s)
                            for zid in zone_ids
                            for w in range(self.num_words)
                            for s in range(self.sites_per_word)
                        ]
                    ),
                )
            )

        # Reuse inherited paths when nothing about the buses has changed.
        # Bundled architectures ship hardware-derived lane geometry that
        # this search does not reproduce, and move durations are measured
        # off those segment lengths, so recomputing silently would shift
        # fidelity estimates and routing metrics.
        paths: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
        for zone_id, z in enumerate(self._zones):
            shape = self._inherited_shape.get(z.name)
            unchanged = shape is not None and shape == self._zone_shape(z)
            if not recompute_paths and unchanged:
                # Nothing about this zone's buses moved, so its inherited
                # lanes still describe it exactly; no search, and no
                # clearances needed.
                paths.update(self._reusable_inherited(zone_id, z))
                continue
            computed = compute_transport_paths(
                self._routing_geometry(z),
                site_buses=z.site_buses,
                word_buses=z.word_buses,
                site_bus_words=z.words_with_site_buses,
                zone_id=zone_id,
                warn_stacklevel=3,
            )
            if not recompute_paths:
                # Buses that existed at from_spec keep their inherited
                # geometry; only the appended ones take search results.
                computed.update(self._reusable_inherited(zone_id, z))
            paths.update(computed)

        return ArchSpec.from_components(
            words=words,
            zones=tuple(rust_zones),
            modes=modes,
            zone_buses=zone_buses,
            paths=paths or None,
            feed_forward=(self._feed_forward if feed_forward is None else feed_forward),
            atom_reloading=(
                self._atom_reloading if atom_reloading is None else atom_reloading
            ),
            blockade_radius=(
                blockade_radius
                if blockade_radius is not None
                else self._blockade_radius
            ),
        )

    # ── from_spec ──

    @classmethod
    def from_spec(
        cls,
        spec: ArchSpec,
        *,
        x_clearance: float | None = None,
        y_clearance: float | None = None,
    ) -> ArchBuilder:
        """Rebuild a builder from an existing :class:`ArchSpec`.

        Every zone is replayed in phase order, so a spec that violates the
        builder's rules — a bus no AOD can perform, say — is rejected here
        rather than restored silently.

        The spec's transport paths are carried over and reused verbatim by
        :meth:`build`, per bus: appending a bus searches paths for that bus
        alone and leaves every pre-existing lane exactly as it was.  That
        matters: the bundled architectures ship hardware-derived lane
        geometry that this path search does not reproduce (no clearance
        value regenerates it), and move durations — hence fidelity
        estimates and the committed benchmark baselines — are read off
        those segment lengths.

        Clearances are **not** part of an ``ArchSpec``: they are inputs to
        the path search, not properties of the architecture, so a
        round-trip cannot recover them.  Supply them if you intend to edit
        the buses, which makes the inherited paths stale and forces a
        search; without them, such an edit raises rather than silently
        re-routing.  Passing them does not by itself discard the inherited
        paths — use ``build(recompute_paths=True)`` for that.

        Args:
            spec: The spec to rebuild.
            x_clearance: Minimum x-axis waypoint clearance (µm), or
                ``None`` to keep the spec's own paths.
            y_clearance: Same, for the y-axis.

        Raises:
            ValueError: If the spec has no words or zones, if its zones
                disagree on grid dimensions, or if any bus is not
                realizable under the builder's rules.
        """
        inner = spec._inner
        if not inner.words:
            raise ValueError("spec has no words")
        if not inner.zones:
            raise ValueError("spec has no zones")

        def _factor(sites: list[tuple[int, ...]]) -> tuple[list[int], list[int]]:
            """Split a word's sites into (rows, columns), or reject it.

            A word here is a row-major Cartesian product, because that is
            what ``word_shape`` and the ``column + row * num_columns`` site
            numbering mean.  ``ArchSpec`` does not require that, so a spec
            can hold words this builder cannot express — say so instead of
            canonicalizing them, since site IDs address bus endpoints and
            inherited lanes and renumbering them would re-map routing.
            """
            rows = list(dict.fromkeys(y for _, y in sites))
            cols = list(dict.fromkeys(x for x, _ in sites))
            if [(c, r) for r in rows for c in cols] != sites:
                raise ValueError(
                    f"word sites {sites} are not a row-major grid of "
                    f"{len(rows)} row(s) x {len(cols)} column(s). This builder "
                    "numbers sites as column + row * num_columns, so it cannot "
                    "represent this word without renumbering it — which would "
                    "silently re-map every bus endpoint and inherited lane."
                )
            return rows, cols

        sites = [tuple(s) for s in inner.words[0].sites]
        first_rows, first_cols = _factor(sites)
        word_shape = (len(first_rows), len(first_cols))

        shapes = {(z.grid.num_y, z.grid.num_x) for z in inner.zones}
        if len(shapes) > 1:
            raise ValueError(
                f"zones disagree on grid dimensions (rows, columns): "
                f"{sorted(shapes)}. The word "
                "template indexes one shared index space, so every zone's grid "
                "must have the same dimensions."
            )
        grid_shape = shapes.pop()

        builder = cls(grid_shape=grid_shape, word_shape=word_shape)
        for word in inner.words:
            rows, cols = _factor([tuple(s) for s in word.sites])
            builder.add_word(rows=rows, columns=cols)

        for zone in inner.zones:
            builder.add_zone(
                zone.name,
                rows=list(zone.grid.y_positions),
                columns=list(zone.grid.x_positions),
                x_clearance=x_clearance,
                y_clearance=y_clearance,
                # Participation is passed through verbatim: an empty list
                # alongside buses is a real value and must not be rebuilt as
                # "everything". With no buses of that kind, ``build`` emits
                # ``[]`` whatever the builder holds, so the spec's empty list
                # carries no information and defaulting to all keeps the zone
                # extensible.
                words_with_site_buses=(
                    list(zone.words_with_site_buses)
                    if zone.site_buses
                    else list(range(len(inner.words)))
                ),
                sites_with_word_buses=(
                    list(zone.sites_with_word_buses)
                    if zone.word_buses
                    else list(range(word_shape[0] * word_shape[1]))
                ),
            )

        for zone in inner.zones:
            for bus in zone.site_buses:
                builder.add_site_bus(zone.name, list(bus.src), list(bus.dst))
            for bus in zone.word_buses:
                builder.add_word_bus(zone.name, list(bus.src), list(bus.dst))
            if zone.entangling_pairs:
                pairs = [tuple(p) for p in zone.entangling_pairs]
                builder.add_entangling_pairs(
                    zone.name, [a for a, _ in pairs], [b for _, b in pairs]
                )

        names = [z.name for z in inner.zones]
        for bus_id, bus in enumerate(inner.zone_buses):
            endpoints = []
            for side, entries in (("src", bus.src), ("dst", bus.dst)):
                pairs = [tuple(e) for e in entries]
                if not pairs:
                    raise ValueError(
                        f"zone bus {bus_id} has an empty {side} endpoint; a bus "
                        "must move at least one word."
                    )
                zone_ids = {z for z, _ in pairs}
                if len(zone_ids) != 1:
                    raise ValueError(
                        f"zone bus {bus_id}'s {side} endpoint spans zones "
                        f"{sorted(zone_ids)}. connect addresses one zone per "
                        "side, so this builder cannot express it."
                    )
                endpoints.append((names[zone_ids.pop()], [w for _, w in pairs]))
            builder.connect(endpoints[0], endpoints[1])

        for mode in inner.modes:
            builder.add_mode(
                mode.name,
                [names[z] for z in mode.zones],
                bitstring_order=list(mode.bitstring_order),
            )

        builder._inherited_paths = dict(spec.paths) if spec.paths else None
        builder._inherited_shape = {
            z.name: builder._zone_shape(z) for z in builder._zones
        }
        builder._feed_forward = spec.feed_forward
        builder._atom_reloading = spec.atom_reloading
        builder._blockade_radius = spec.blockade_radius
        if spec.blockade_radius is not None:
            radius_nm = to_nm(spec.blockade_radius, "blockade_radius")
            for z in builder._zones:
                z.blockade_radius_nm = radius_nm
        return builder

    # ── Internals ──

    def _zone(self, name: str) -> _Zone:
        """Look up a zone by name."""
        if name not in self._zone_ids:
            raise ValueError(f"unknown zone: '{name}'")
        return self._zones[self._zone_ids[name]]

    def _geometry(self, zone: _Zone) -> ZoneGeometry:
        """Resolve the shared template against one zone's coordinates.

        Bus validation and the blockade scan read only positions, so a
        zone without clearances still resolves; :meth:`_routing_geometry`
        is the accessor that insists on them.
        """
        return ZoneGeometry(
            name=zone.name,
            grid_x_nm=zone.grid_x_nm,
            grid_y_nm=zone.grid_y_nm,
            x_clearance_nm=zone.x_clearance_nm or 0,
            y_clearance_nm=zone.y_clearance_nm or 0,
            words=tuple(tuple(w) for w in self._words),
        )

    def _routing_geometry(self, zone: _Zone) -> ZoneGeometry:
        """Like :meth:`_geometry`, but require clearances for path search."""
        if zone.x_clearance_nm is None or zone.y_clearance_nm is None:
            raise ValueError(
                f"zone '{zone.name}' has no clearances, so transport paths "
                "cannot be searched for it. It came from a spec whose paths "
                "were inherited; supply x_clearance/y_clearance to from_spec "
                "to recompute them."
            )
        return self._geometry(zone)

    def _zone_shape(self, zone: _Zone) -> tuple[int, int, tuple, tuple]:
        """Bus counts and participation, the things inherited lanes depend on."""
        return (
            len(zone.site_buses),
            len(zone.word_buses),
            zone.words_with_site_buses,
            zone.sites_with_word_buses,
        )

    def _reusable_inherited(
        self, zone_id: int, zone: _Zone
    ) -> dict[LaneAddress, tuple[tuple[float, float], ...]]:
        """Inherited lanes still valid for this zone.

        Buses are only ever appended, so a bus that existed at
        ``from_spec`` keeps its ID and its lanes stay meaningful.  What
        invalidates them is a change of participation: ``words_with_site_buses``
        decides which words get site lanes, ``sites_with_word_buses`` which
        sites get word lanes.
        """
        shape = self._inherited_shape.get(zone.name)
        if shape is None or self._inherited_paths is None:
            return {}
        n_site, n_word, movers, carried = shape
        site_ok = movers == zone.words_with_site_buses
        word_ok = carried == zone.sites_with_word_buses
        site_cap = min(n_site, len(zone.site_buses))
        word_cap = min(n_word, len(zone.word_buses))
        out: dict[LaneAddress, tuple[tuple[float, float], ...]] = {}
        for lane, path in self._inherited_paths.items():
            if lane.zone_id != zone_id:
                continue
            if lane.move_type == MoveType.SITE:
                if site_ok and lane.bus_id < site_cap:
                    out[lane] = path
            elif word_ok and lane.bus_id < word_cap:
                out[lane] = path
        return out

    @staticmethod
    def _occupied(geom: ZoneGeometry) -> frozenset[tuple[int, int]]:
        """Every atom position in a zone, in nm."""
        return frozenset(
            geom.site_nm(w, s)
            for w in range(geom.num_words)
            for s in range(geom.sites_per_word)
        )

    def _check_bus_endpoints(
        self,
        src: Sequence[int],
        dst: Sequence[int],
        size: int,
        kind: str,
    ) -> None:
        """Shared endpoint checks: equal length, non-empty, in range."""
        if len(src) != len(dst):
            raise ValueError(
                f"{kind} bus src has {len(src)} entries but dst has {len(dst)}"
            )
        if not src:
            raise ValueError(
                f"a {kind} bus must move at least one {kind}; an empty bus "
                "would reach the spec as a bus that transports nothing"
            )
        for label, seq in (("src", src), ("dst", dst)):
            bad = sorted({i for i in seq if not 0 <= i < size})
            if bad:
                raise IndexError(
                    f"{kind} bus {label} {kind} index {bad} out of range "
                    f"[0, {size})"
                )


__all__ = ["ArchBuilder"]
