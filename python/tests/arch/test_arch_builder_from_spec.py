"""Tests for restoring a materialized ``ArchSpec`` into the imperative builder.

Covers ``ZoneBuilder.from_zone`` / ``set_path``, ``ArchBuilder.from_spec`` /
``zone``, the order-preserving ``zone.words.ordered`` query, the append-only
bus guard, and ``ArchResult.builder``.
"""

from __future__ import annotations

import warnings

import pytest

from bloqade.lanes.arch.build.blueprint import ArchResult
from bloqade.lanes.arch.build.imperative import (
    ArchBuilder,
    ZoneBuilder,
    _infer_word_shape,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.arch.gemini_full import get_arch as get_gemini_full
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import (
    Grid,
    SiteBus as _RustSiteBus,
    WordBus as _RustWordBus,
    Zone as _RustZone,
)
from bloqade.lanes.bytecode.encoding import Direction, LaneAddress, MoveType
from bloqade.lanes.bytecode.word import Word

_CL = 0.25


def _make_grid(nx: int, ny: int, *, y_offset: float = 0.0) -> Grid:
    xs = [float(i) for i in range(nx)]
    ys = [y_offset + float(j) for j in range(ny)]
    return Grid.from_positions(xs, ys)


def _assert_spec_equal(a: ArchSpec, b: ArchSpec) -> None:
    """Structural equality, ignoring ``Mode.bitstring_order``.

    ``build()`` regenerates ``bitstring_order`` from the zone/word template
    (the Rust core only validates it, never consumes it), so it is the one
    field a restore-then-build round trip is documented not to copy.
    """
    assert a.words == b.words
    assert a.zones == b.zones
    assert [(zb.src, zb.dst) for zb in a.zone_buses] == [
        (zb.src, zb.dst) for zb in b.zone_buses
    ]
    assert [(m.name, m.zones) for m in a.modes] == [(m.name, m.zones) for m in b.modes]
    assert dict(a.paths) == dict(b.paths)
    assert a.feed_forward == b.feed_forward
    assert a.atom_reloading == b.atom_reloading
    assert a.blockade_radius == b.blockade_radius


def _two_zone_builder() -> ArchBuilder:
    """Two zones sharing a 4-word template, with site/word/zone buses.

    Grid per zone: 4 x-positions × 2 rows, words of shape (2, 1):
    word 0 = (0..1, row 0), word 1 = (2..3, row 0), word 2 = (0..1, row 1),
    word 3 = (2..3, row 1).  Word 1 opts out of site-bus transport so the
    ``words_with_site_buses`` restoration path is exercised.
    """
    builder = ArchBuilder()
    for name, y_off in (("proc", 0.0), ("store", 10.0)):
        zone = ZoneBuilder(
            name,
            _make_grid(4, 2, y_offset=y_off),
            word_shape=(2, 1),
            x_clearance=_CL,
            y_clearance=_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0], has_site_bus=False)
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus([0], [1])
        zone.add_word_bus([0, 1], [2, 3])
        builder.add_zone(zone)
    builder.connect(("proc", [0, 1]), ("store", [0, 1]))
    builder.add_mode("all", ["proc", "store"])
    builder.add_mode("proc", ["proc"])
    return builder


# ── Gemini physical round trip ──


@pytest.fixture(scope="module")
def physical() -> ArchSpec:
    return get_arch_spec()


class TestPhysicalRoundTrip:
    def test_round_trip_is_identical(self, physical: ArchSpec):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            builder = ArchBuilder.from_spec(physical, x_clearance=0.5, y_clearance=3.0)
            rebuilt = builder.build()
        _assert_spec_equal(physical, rebuilt)
        assert len(rebuilt.paths) == 1120

    def test_restored_zone_state(self, physical: ArchSpec):
        builder = ArchBuilder.from_spec(physical, x_clearance=0.5, y_clearance=3.0)
        gate = builder.zone("gate")
        assert gate.name == "gate"
        assert gate.word_shape == (8, 1)
        assert gate.num_words == 20
        assert len(gate._site_buses) == 3
        assert len(gate._word_buses) == 19
        # Only the odd words carry site buses in the shipped spec.
        assert gate._word_has_site_bus == [w % 2 == 1 for w in range(20)]
        assert gate.blockade_radius is None
        assert gate.x_clearance == 0.5
        assert gate.y_clearance == 3.0

    def test_add_word_bus_routes_only_new_lanes(self, physical: ArchSpec):
        builder = ArchBuilder.from_spec(physical, x_clearance=0.5, y_clearance=3.0)
        gate = builder.zone("gate")
        n_before = len(gate._word_buses)
        # Word 0 (row 0) → word 4 (row 1): a single vertical shift.
        gate.add_word_bus(src=[0], dst=[4])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            extended = builder.build()

        old_paths = dict(physical.paths)
        new_paths = dict(extended.paths)
        for lane, waypoints in old_paths.items():
            assert new_paths[lane] == waypoints
        added = {lane for lane in new_paths if lane not in old_paths}
        expected = {
            LaneAddress(MoveType.WORD, 0, site, n_before, direction, 0)
            for site in range(8)
            for direction in (Direction.FORWARD, Direction.BACKWARD)
        }
        assert added == expected
        assert len(extended.zones[0].word_buses) == n_before + 1

    def test_set_path_replaces_preserved(self, physical: ArchSpec):
        builder = ArchBuilder.from_spec(physical, x_clearance=0.5, y_clearance=3.0)
        gate = builder.zone("gate")
        lane = next(iter(physical.paths))
        original = physical.paths[lane]
        replacement = (
            original[0],
            (original[0][0] + 1.0, original[0][1]),
            original[-1],
        )
        assert replacement != original
        gate.set_path(lane, replacement)
        rebuilt = builder.build()
        assert rebuilt.paths[lane] == replacement
        for other, waypoints in physical.paths.items():
            if other != lane:
                assert rebuilt.paths[other] == waypoints

    def test_zone_unknown_raises(self, physical: ArchSpec):
        builder = ArchBuilder.from_spec(physical, x_clearance=0.5, y_clearance=3.0)
        with pytest.raises(ValueError, match="Unknown zone: 'nope'"):
            builder.zone("nope")


# ── Gemini full: blockade rescan surfaces the crossed-index layout ──


class TestGeminiFullRescan:
    def test_from_spec_raises_crossed_index(self):
        spec = get_gemini_full().arch
        assert spec.blockade_radius is not None
        with pytest.raises(ValueError, match="crossed-index"):
            ArchBuilder.from_spec(spec, x_clearance=0.5, y_clearance=3.0)


# ── Two-zone round trip: zone buses, modes, blockade radius ──


class TestTwoZoneRoundTrip:
    def test_round_trip_is_identical(self):
        original = _two_zone_builder().build(feed_forward=True, atom_reloading=True)
        builder = ArchBuilder.from_spec(original, x_clearance=_CL, y_clearance=_CL)
        rebuilt = builder.build(feed_forward=True, atom_reloading=True)
        _assert_spec_equal(original, rebuilt)
        assert len(original.zone_buses) == 1
        assert [(m.name, m.zones) for m in rebuilt.modes] == [
            ("all", [0, 1]),
            ("proc", [0]),
        ]

    def test_words_with_site_buses_restored(self):
        original = _two_zone_builder().build()
        builder = ArchBuilder.from_spec(original, x_clearance=_CL, y_clearance=_CL)
        assert builder.zone("proc")._word_has_site_bus == [True, False, True, True]

    def test_words_with_site_buses_fallback_without_site_buses(self):
        builder = ArchBuilder()
        zone = ZoneBuilder(
            "z", _make_grid(4, 1), word_shape=(2, 1), x_clearance=_CL, y_clearance=_CL
        )
        zone.add_word(slice(0, 2), [0], has_site_bus=False)
        zone.add_word(slice(2, 4), [0], has_site_bus=False)
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        spec = builder.build()
        assert spec.zones[0].words_with_site_buses == []
        restored = ArchBuilder.from_spec(spec, x_clearance=_CL, y_clearance=_CL)
        # Ambiguous with no site buses → the add_word default (all True).
        assert restored.zone("z")._word_has_site_bus == [True, True]
        _assert_spec_equal(spec, restored.build())

    def test_blockade_radius_rescan_reproduces_pairs(self):
        builder = ArchBuilder()
        zone = ZoneBuilder(
            "cz",
            Grid.from_positions([0.0, 1.0, 5.0, 6.0], [0.0]),
            word_shape=(1, 1),
            x_clearance=_CL,
            y_clearance=_CL,
        )
        for x in range(4):
            zone.add_word([x], [0])
        builder.add_zone(zone)
        builder.add_mode("all", ["cz"])
        builder.set_blockade_radius(1.5)
        spec = builder.build()
        assert spec.zones[0].entangling_pairs == [(0, 1), (2, 3)]
        assert spec.blockade_radius == 1.5

        restored = ArchBuilder.from_spec(spec, x_clearance=_CL, y_clearance=_CL)
        assert restored.zone("cz").blockade_radius == 1.5
        _assert_spec_equal(spec, restored.build())

    def test_extend_two_zone_spec(self):
        original = _two_zone_builder().build()
        builder = ArchBuilder.from_spec(original, x_clearance=_CL, y_clearance=_CL)
        store = builder.zone("store")
        store.add_word_bus([2], [3])
        builder.connect(("proc", [2, 3]), ("store", [2, 3]))
        extended = builder.build()
        assert len(extended.zones[1].word_buses) == 2
        assert len(extended.zone_buses) == 2
        for lane, waypoints in original.paths.items():
            assert extended.paths[lane] == waypoints


# ── from_zone validation ──


class TestFromZoneValidation:
    def test_infer_word_shape_2d(self):
        words = [
            Word(((0, 0), (1, 0), (0, 1), (1, 1))),
            Word(((2, 0), (3, 0), (2, 1), (3, 1))),
        ]
        assert _infer_word_shape(words) == (2, 2)

    def test_infer_word_shape_non_rectangular_raises(self):
        with pytest.raises(ValueError, match="rectangular"):
            _infer_word_shape([Word(((0, 0), (1, 1)))])

    def test_infer_word_shape_non_uniform_raises(self):
        with pytest.raises(ValueError, match="non-uniform"):
            _infer_word_shape([Word(((0, 0),)), Word(((1, 0), (2, 0)))])

    def test_infer_word_shape_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _infer_word_shape([])

    def test_site_order_preserved_verbatim(self):
        """A word's site order is its site_id semantics; keep it as recorded."""
        words = [Word(((1, 0), (0, 0))), Word(((3, 0), (2, 0)))]
        zone = _RustZone(
            name="z",
            grid=_make_grid(4, 1),
            site_buses=[],
            word_buses=[],
            words_with_site_buses=[],
            sites_with_word_buses=[],
        )
        zb = ZoneBuilder.from_zone(zone, words, x_clearance=_CL, y_clearance=_CL)
        assert zb._words == [[(1, 0), (0, 0)], [(3, 0), (2, 0)]]
        assert zb._position_to_word == {(1, 0): 0, (0, 0): 0, (3, 0): 1, (2, 0): 1}

    def test_partial_sites_with_word_buses_raises(self):
        words = [Word(((0, 0), (1, 0))), Word(((2, 0), (3, 0)))]
        zone = _RustZone(
            name="z",
            grid=_make_grid(4, 1),
            site_buses=[],
            word_buses=[_RustWordBus(src=[0], dst=[1])],
            words_with_site_buses=[],
            sites_with_word_buses=[0],
        )
        with pytest.raises(ValueError, match="sites_with_word_buses"):
            ZoneBuilder.from_zone(zone, words, x_clearance=_CL, y_clearance=_CL)

    def test_non_rectangular_bus_rejected(self):
        """Restoration goes through add_word_bus, so AOD validation applies."""
        words = [Word(((x, y),)) for y in range(2) for x in range(2)]
        zone = _RustZone(
            name="z",
            grid=_make_grid(2, 2),
            site_buses=[],
            # src words 0, 1, 2 = (0,0), (1,0), (0,1): an L, not a rectangle.
            word_buses=[_RustWordBus(src=[0, 1, 2], dst=[3, 2, 1])],
            words_with_site_buses=[],
            sites_with_word_buses=[0],
        )
        with pytest.raises(ValueError, match="Cartesian product"):
            ZoneBuilder.from_zone(zone, words, x_clearance=_CL, y_clearance=_CL)

    def test_paths_seed_overrides(self):
        words = [Word(((0, 0), (1, 0)))]
        zone = _RustZone(
            name="z",
            grid=_make_grid(2, 1),
            site_buses=[_RustSiteBus(src=[0], dst=[1])],
            word_buses=[],
            words_with_site_buses=[0],
            sites_with_word_buses=[],
        )
        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        zb = ZoneBuilder.from_zone(
            zone,
            words,
            x_clearance=_CL,
            y_clearance=_CL,
            paths={lane: [(0.0, 0.0), (0.0, 0.5), (1.0, 0.5), (1.0, 0.0)]},
        )
        assert zb._path_overrides == {
            lane: ((0.0, 0.0), (0.0, 0.5), (1.0, 0.5), (1.0, 0.0))
        }


# ── set_path ──


class TestSetPath:
    def _zone(self) -> ZoneBuilder:
        zone = ZoneBuilder(
            "z", _make_grid(4, 2), word_shape=(2, 1), x_clearance=_CL, y_clearance=_CL
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus([0], [1])
        zone.add_word_bus([0, 1], [2, 3])
        return zone

    def test_too_few_waypoints_raises(self):
        zone = self._zone()
        lane = LaneAddress(MoveType.SITE, 0, 0, 0)
        with pytest.raises(ValueError, match="at least 2 waypoints"):
            zone.set_path(lane, [(0.0, 0.0)])

    def test_non_finite_raises(self):
        zone = self._zone()
        lane = LaneAddress(MoveType.SITE, 0, 0, 0)
        with pytest.raises(ValueError, match="non-finite"):
            zone.set_path(lane, [(0.0, 0.0), (float("nan"), 0.0)])

    def test_bus_out_of_range_raises(self):
        zone = self._zone()
        with pytest.raises(ValueError, match="word bus index 5"):
            zone.set_path(LaneAddress(MoveType.WORD, 0, 0, 5), [(0, 0), (0, 1)])
        with pytest.raises(ValueError, match="site bus index 1"):
            zone.set_path(LaneAddress(MoveType.SITE, 0, 0, 1), [(0, 0), (1, 0)])

    def test_word_out_of_range_raises(self):
        zone = self._zone()
        with pytest.raises(ValueError, match="word index 9"):
            zone.set_path(LaneAddress(MoveType.WORD, 9, 0, 0), [(0, 0), (0, 1)])

    def test_zone_id_mismatch_raises_at_build(self):
        zone = self._zone()
        zone.set_path(LaneAddress(MoveType.WORD, 0, 0, 0, zone_id=3), [(0, 0), (0, 1)])
        builder = ArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        with pytest.raises(ValueError, match="zone_id is 3"):
            builder.build()

    def test_pinned_lane_beats_search_and_others_are_computed(self):
        zone = self._zone()
        pinned = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        detour = ((0.0, 0.0), (0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0))
        zone.set_path(pinned, detour)
        builder = ArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        spec = builder.build()
        assert spec.paths[pinned] == detour
        # The rest of the bus was still routed by the search.
        sibling = LaneAddress(MoveType.WORD, 1, 0, 0, Direction.FORWARD, 0)
        assert sibling in spec.paths
        assert spec.paths[sibling] != detour
        backward = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.BACKWARD, 0)
        assert backward in spec.paths

    def test_fully_pinned_bus_skips_single_shift_warning(self):
        """A bus whose lanes are all pinned is neither searched nor checked."""
        zone = ZoneBuilder(
            "z", _make_grid(4, 2), word_shape=(1, 1), x_clearance=_CL, y_clearance=_CL
        )
        for y in range(2):
            for x in range(4):
                zone.add_word([x], [y])
        # (0,0)→(1,1) and (1,0)→(3,1): different displacements.
        zone.add_word_bus([0, 1], [5, 7])
        for word, dst in ((0, (1.0, 1.0)), (1, (3.0, 1.0))):
            src = (float(word), 0.0)
            fwd = LaneAddress(MoveType.WORD, word, 0, 0, Direction.FORWARD, 0)
            bwd = LaneAddress(MoveType.WORD, word, 0, 0, Direction.BACKWARD, 0)
            zone.set_path(fwd, [src, dst])
            zone.set_path(bwd, [dst, src])
        builder = ArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            spec = builder.build()
        assert len(spec.paths) == 4


# ── Append-only guard ──


class TestAppendOnlyGuard:
    def test_removing_restored_word_bus_raises(self):
        builder = ArchBuilder.from_spec(
            _two_zone_builder().build(), x_clearance=_CL, y_clearance=_CL
        )
        builder.zone("proc")._word_buses.pop()
        with pytest.raises(ValueError, match="append-only"):
            builder.build()

    def test_reordering_restored_site_buses_raises(self):
        builder = ArchBuilder.from_spec(
            _two_zone_builder().build(), x_clearance=_CL, y_clearance=_CL
        )
        proc = builder.zone("proc")
        proc.add_site_bus([1], [0])
        proc._site_buses.reverse()
        with pytest.raises(ValueError, match="append-only"):
            builder.build()

    def test_removing_restored_zone_bus_raises(self):
        builder = ArchBuilder.from_spec(
            _two_zone_builder().build(), x_clearance=_CL, y_clearance=_CL
        )
        builder._connections.clear()
        with pytest.raises(ValueError, match="append-only"):
            builder.build()

    def test_appending_is_fine(self):
        builder = ArchBuilder.from_spec(
            _two_zone_builder().build(), x_clearance=_CL, y_clearance=_CL
        )
        builder.zone("proc").add_word_bus([0], [2])
        builder.connect(("proc", [2]), ("store", [2]))
        builder.build()


# ── Order-preserving word selection ──


class TestOrderedWordSelection:
    def _zone(self) -> ZoneBuilder:
        zone = ZoneBuilder(
            "z", _make_grid(4, 2), word_shape=(1, 1), x_clearance=_CL, y_clearance=_CL
        )
        for y in range(2):
            for x in range(4):
                zone.add_word([x], [y])  # word id = x + 4*y
        return zone

    def test_sequence_order_is_kept(self):
        zone = self._zone()
        assert zone.words.ordered[[2, 0, 3, 1], 0] == [2, 0, 3, 1]
        assert zone.words[[2, 0, 3, 1], 0] == [0, 1, 2, 3]

    def test_slice_expands_ascending(self):
        zone = self._zone()
        assert zone.words.ordered[:, 1] == [4, 5, 6, 7]
        assert zone.words.ordered[slice(3, None, -1), 1] == [7, 6, 5, 4]

    def test_x_outer_y_inner(self):
        zone = self._zone()
        assert zone.words.ordered[[1, 0], [1, 0]] == [5, 1, 4, 0]

    def test_word_spanning_positions_reported_once(self):
        zone = ZoneBuilder(
            "z", _make_grid(4, 1), word_shape=(2, 1), x_clearance=_CL, y_clearance=_CL
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        assert zone.words.ordered[[3, 2, 1, 0], 0] == [1, 0]

    def test_empty_positions_skipped(self):
        zone = self._zone()
        assert zone.words.ordered[[7, 2], 0] == [2]

    def test_permutation_bus_sorted_cannot_express(self):
        zone = self._zone()
        src = zone.words.ordered[[0, 1, 2, 3], 0]
        dst = zone.words.ordered[[3, 2, 1, 0], 1]
        zone.add_word_bus(src, dst)
        assert zone._word_buses[-1] == ([0, 1, 2, 3], [7, 6, 5, 4])
        # The sorted query can only pair src[i] with the i-th smallest dst.
        assert zone.words[[3, 2, 1, 0], 1] == [4, 5, 6, 7]
        builder = ArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        # The permutation has distinct per-pair displacements, so the bus is
        # legal but unroutable by a single AOD shift.
        with pytest.warns(UserWarning, match="single-shift invariant"):
            spec = builder.build()
        assert spec.zones[0].word_buses[0].dst == [7, 6, 5, 4]


# ── Site-bus paths honour has_site_bus ──


class TestSiteBusPathsHonourHasSiteBus:
    def test_no_paths_for_words_without_site_bus(self):
        zone = ZoneBuilder(
            "z", _make_grid(4, 1), word_shape=(2, 1), x_clearance=_CL, y_clearance=_CL
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0], has_site_bus=False)
        zone.add_site_bus([0], [1])
        builder = ArchBuilder()
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        spec = builder.build()
        site_lanes = {lane for lane in spec.paths if lane.move_type == MoveType.SITE}
        assert {lane.word_id for lane in site_lanes} == {0}
        assert spec.zones[0].words_with_site_buses == [0]

    def test_word_zero_opting_out_keeps_movers_clear(self):
        """The bus reference atom must be one that actually moves.

        ``_enumerate_safe_positions`` derives the bus offset set from
        ``bus_src_atoms[0]``, so if the reference is word 0 while word 0 has
        opted out, every safe-waypoint candidate is evaluated in a frame
        shifted by (first mover - word 0) and the clearance guarantee stops
        describing the atoms actually being transported.  This is the shape
        of the shipped Gemini physical spec, whose ``words_with_site_buses``
        is the odd words only.
        """
        # Irregular bases so that a +2 µm frame shift is not a symmetry.
        bases = [1.0, 3.0, 4.5]
        pitch = 6.0
        xs = sorted({b for b in bases} | {b + pitch for b in bases})
        ys = [-2.0, 0.0, 2.0]
        x_cl, y_cl = 0.5, 1.5

        zone = ZoneBuilder(
            "z",
            Grid.from_positions(xs, ys),
            word_shape=(2, 1),
            x_clearance=x_cl,
            y_clearance=y_cl,
        )
        for i, base in enumerate(bases):
            zone.add_word(
                [xs.index(base), xs.index(base + pitch)], [1], has_site_bus=i != 0
            )
        zone.add_site_bus([0], [1])

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            paths = zone._compute_paths(0)

        forward = {
            lane.word_id: pts
            for lane, pts in paths.items()
            if lane.direction == Direction.FORWARD
        }
        assert set(forward) == {1, 2}, "only the opted-in words move"

        # Every lane must land on its own destination site.
        for word_id, pts in forward.items():
            dst_x, dst_y = zone._site_nm(word_id, 1)
            assert pts[-1] == (dst_x / 1000, dst_y / 1000)

        # At each intermediate waypoint the whole bus shifts together, so
        # every mover must clear the grid on at least one axis.
        n_waypoints = {len(pts) for pts in forward.values()}
        assert len(n_waypoints) == 1
        for i in range(1, n_waypoints.pop() - 1):
            at_step = [pts[i] for pts in forward.values()]
            x_clear = all(abs(x - g) >= x_cl for x, _ in at_step for g in xs)
            y_clear = all(abs(y - g) >= y_cl for _, y in at_step for g in ys)
            assert x_clear or y_clear, (
                f"waypoint {i} {at_step} puts a moving atom inside the "
                f"clearance of a static grid site"
            )


# ── Device capabilities survive a round trip ──


class TestCapabilitiesRoundTrip:
    def _spec(self, **kwargs) -> ArchSpec:
        builder = ArchBuilder()
        zone = ZoneBuilder(
            "z", _make_grid(4, 1), word_shape=(2, 1), x_clearance=_CL, y_clearance=_CL
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word_bus([0], [1])
        builder.add_zone(zone)
        builder.add_mode("all", ["z"])
        return builder.build(**kwargs)

    def test_restored_capabilities_are_not_dropped(self):
        original = self._spec(feed_forward=True, atom_reloading=True)
        restored = ArchBuilder.from_spec(original, x_clearance=_CL, y_clearance=_CL)
        rebuilt = restored.build()
        assert (rebuilt.feed_forward, rebuilt.atom_reloading) == (True, True)
        _assert_spec_equal(original, rebuilt)

    def test_explicit_argument_overrides_restored(self):
        original = self._spec(feed_forward=True, atom_reloading=True)
        restored = ArchBuilder.from_spec(original, x_clearance=_CL, y_clearance=_CL)
        rebuilt = restored.build(feed_forward=False)
        assert (rebuilt.feed_forward, rebuilt.atom_reloading) == (False, True)

    def test_fresh_builder_still_defaults_to_false(self):
        spec = self._spec()
        assert (spec.feed_forward, spec.atom_reloading) == (False, False)


# ── ArchResult.builder ──


class TestArchResultBuilder:
    def test_builder_reproduces_arch(self):
        result: ArchResult = get_gemini_full()
        assert isinstance(result.builder, ArchBuilder)
        rebuilt = result.builder.build(
            feed_forward=result.arch.feed_forward,
            atom_reloading=result.arch.atom_reloading,
            blockade_radius=result.arch.blockade_radius,
        )
        _assert_spec_equal(result.arch, rebuilt)

    def test_builder_can_be_extended(self):
        result = get_gemini_full()
        names = list(result.zone_indices)
        zone = result.builder.zone(names[0])
        n_before = len(zone._word_buses)
        # A word bus back onto itself is a self-loop; pick two distinct
        # words that exist in every zone of the shared template.
        zone.add_word_bus([0], [1])
        extended = result.builder.build(blockade_radius=result.arch.blockade_radius)
        assert len(extended.zones[result.zone_indices[names[0]]].word_buses) == (
            n_before + 1
        )
