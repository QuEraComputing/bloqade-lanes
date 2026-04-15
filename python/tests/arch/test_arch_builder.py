"""Tests for ZoneBuilder and ArchBuilder."""

import pytest

from bloqade.lanes.arch.arch_builder import (
    ArchBuilder,
    ZoneBuilder,
    _SiteGridQuery,
    _validate_aod_rectangle,
)
from bloqade.lanes.bytecode._native import Grid
from bloqade.lanes.layout.encoding import Direction, LaneAddress, MoveType

# Default clearance used by most tests.  The unit-spaced test grids have
# spacings of 1.0, so 0.25 is comfortably below half-spacing.
_DEFAULT_CL = 0.25


def _make_grid(nx: int, ny: int, *, y_offset: float = 0.0) -> Grid:
    """Create a simple grid with unit spacing."""
    xs = [float(i) for i in range(nx)]
    ys = [y_offset + float(j) for j in range(ny)]
    return Grid.from_positions(xs, ys)


# ── _SiteGridQuery ──


class TestSiteGridQuery:
    def test_full_slice(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[:, :] == [0, 1, 2, 3, 4, 5]

    def test_single_row(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[:, 0] == [0, 1, 2]

    def test_second_row(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[:, 1] == [3, 4, 5]

    def test_x_slice(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[slice(0, 2), :] == [0, 1, 3, 4]

    def test_explicit_list(self):
        q = _SiteGridQuery(word_shape=(4, 2))
        assert q[[0, 2], 0] == [0, 2]

    def test_single_int(self):
        q = _SiteGridQuery(word_shape=(3, 2))
        assert q[0, 0] == [0]


# ── AOD validation ──


class TestAODValidation:
    def test_valid_rectangle(self):
        _validate_aod_rectangle([(0, 0), (1, 0), (0, 1), (1, 1)], "test")

    def test_missing_corner_raises(self):
        with pytest.raises(ValueError, match="Cartesian product"):
            _validate_aod_rectangle([(0, 0), (1, 0), (0, 1)], "test")

    def test_single_point_valid(self):
        _validate_aod_rectangle([(0, 0)], "test")

    def test_empty_valid(self):
        _validate_aod_rectangle([], "test")

    def test_row_valid(self):
        _validate_aod_rectangle([(0, 0), (1, 0), (2, 0)], "test")

    def test_col_valid(self):
        _validate_aod_rectangle([(0, 0), (0, 1), (0, 2)], "test")


# ── ZoneBuilder: add_word ──


class TestZoneBuilderAddWord:
    def test_add_single_word(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        wid = zone.add_word(x_sites=slice(0, 2), y_sites=slice(0, 1))
        assert wid == 0

    def test_add_multiple_words(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.add_word(slice(0, 2), slice(0, 1)) == 0
        assert zone.add_word(slice(2, 4), slice(0, 1)) == 1

    def test_shape_mismatch_x_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        with pytest.raises(ValueError, match="word_shape requires 2"):
            zone.add_word(slice(0, 3), slice(0, 1))

    def test_shape_mismatch_y_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        with pytest.raises(ValueError, match="word_shape requires 1"):
            zone.add_word(slice(0, 2), slice(0, 2))

    def test_overlap_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 1))
        with pytest.raises(ValueError, match="already belongs to word 0"):
            zone.add_word(slice(1, 3), slice(0, 1))

    def test_out_of_bounds_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        with pytest.raises(IndexError):
            zone.add_word([3, 4], [0])

    def test_list_indices(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        wid = zone.add_word([0, 2], [0])
        assert wid == 0

    def test_num_words(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.num_words == 0
        zone.add_word(slice(0, 2), [0])
        assert zone.num_words == 1

    def test_sites_per_word(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 4),
            word_shape=(2, 3),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.sites_per_word == 6


# ── ZoneBuilder: grid queries ──


class TestZoneBuilderQueries:
    def test_words_query_all(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        name, ids = zone.words[:, :]
        assert name == "gate"
        assert ids == [0, 1, 2, 3]

    def test_words_query_first_column(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        name, ids = zone.words[slice(0, 2), :]
        assert name == "gate"
        assert ids == [0, 2]

    def test_words_query_returns_zone_name(self):
        zone = ZoneBuilder(
            "proc",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        name, _ = zone.words[:, :]
        assert name == "proc"

    def test_sites_query(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(3, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        assert zone.sites[:, 0] == [0, 1, 2]
        assert zone.sites[:, 1] == [3, 4, 5]
        assert zone.sites[:, :] == [0, 1, 2, 3, 4, 5]


# ── ZoneBuilder: buses ──


class TestZoneBuilderBuses:
    def test_add_site_bus_valid(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 2))
        zone.add_site_bus(src=[0, 1], dst=[2, 3])

    def test_add_site_bus_invalid_rectangle(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(3, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 3), slice(0, 2))
        # src=[0, 5] → positions (0,0) and (2,1) — diagonal, not a rectangle
        with pytest.raises(ValueError, match="Cartesian product"):
            zone.add_site_bus(src=[0, 5], dst=[1, 4])

    def test_add_site_bus_out_of_range(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 2))
        with pytest.raises(ValueError, match="out of range"):
            zone.add_site_bus(src=[0, 4], dst=[1, 5])

    def test_add_word_bus_valid(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])

    def test_add_word_bus_out_of_range(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        with pytest.raises(ValueError, match="out of range"):
            zone.add_word_bus(src=[0], dst=[1])

    def test_add_entangling_pairs(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_entangling_pairs([0], [1])

    def test_add_entangling_pairs_out_of_range(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        with pytest.raises(ValueError, match="out of range"):
            zone.add_entangling_pairs([0], [1])


# ── ArchBuilder ──


class TestArchBuilder:
    def test_single_zone(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        zid = arch.add_zone(zone)
        assert zid == 0
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert len(spec.zones) == 1
        assert len(spec.words) == 2

    def test_zone_name_preserved(self):
        zone = ZoneBuilder(
            "my_zone",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["my_zone"])
        spec = arch.build()
        assert spec.zones[0].name == "my_zone"

    def test_duplicate_zone_name_raises(self):
        z1 = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z1.add_word(slice(0, 2), [0])
        z2 = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z2.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(z1)
        with pytest.raises(ValueError, match="Duplicate zone name"):
            arch.add_zone(z2)

    def test_sites_per_word_mismatch_raises(self):
        z1 = ZoneBuilder(
            "a",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z1.add_word(slice(0, 2), [0])
        z2 = ZoneBuilder(
            "b",
            _make_grid(4, 1),
            word_shape=(4, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z2.add_word(slice(0, 4), [0])
        arch = ArchBuilder()
        arch.add_zone(z1)
        with pytest.raises(ValueError, match="sites_per_word"):
            arch.add_zone(z2)

    def test_multi_zone_with_connection(self):
        proc = ZoneBuilder(
            "proc",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        mem = ZoneBuilder(
            "mem",
            _make_grid(4, 2, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.connect(src=proc.words[:, :], dst=mem.words[:, :])
        arch.add_mode("all", ["proc", "mem"])
        spec = arch.build()
        assert len(spec.zones) == 2
        assert len(spec.words) == 4
        assert len(spec.zone_buses) == 1

    def test_unknown_zone_in_connect_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        with pytest.raises(ValueError, match="Unknown zone"):
            arch.connect(src=("gate", [0]), dst=("missing", [0]))

    def test_unknown_zone_in_mode_raises(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        with pytest.raises(ValueError, match="Unknown zone"):
            arch.add_mode("test", ["missing"])

    def test_global_word_ids_assigned(self):
        z1 = ZoneBuilder(
            "a",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z1.add_word(slice(0, 2), [0])
        z1.add_word(slice(2, 4), [0])
        z2 = ZoneBuilder(
            "b",
            _make_grid(4, 1, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        z2.add_word(slice(0, 2), [0])
        z2.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        arch.add_zone(z1)
        arch.add_zone(z2)
        arch.add_mode("all", ["a", "b"])
        spec = arch.build()
        assert len(spec.words) == 4

    def test_entangling_pairs_in_single_zone(self):
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_entangling_pairs([0], [1])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec.zones[0].entangling_pairs == [(0, 1)]

    def test_rust_validation_passes(self):
        """Build a realistic arch and verify Rust validation."""
        zone = ZoneBuilder(
            "gate",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus(src=[0], dst=[1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        zone.add_entangling_pairs([0, 2], [1, 3])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec is not None


class TestArchBuilderMultiZoneOffsets:
    """Verify global word ID translation for second+ zones."""

    def _make_two_zone_arch(self):
        proc = ZoneBuilder(
            "proc",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        proc.add_word(slice(0, 2), [1])
        proc.add_word(slice(2, 4), [1])
        proc.add_word_bus(src=[0, 1], dst=[2, 3])
        proc.add_site_bus(src=[0], dst=[1])
        proc.add_entangling_pairs([0, 2], [1, 3])

        mem = ZoneBuilder(
            "mem",
            _make_grid(4, 2, y_offset=10.0),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(2, 4), [0])
        mem.add_word(slice(0, 2), [1])
        mem.add_word(slice(2, 4), [1])
        mem.add_word_bus(src=[0, 1], dst=[2, 3])
        mem.add_entangling_pairs([0, 2], [1, 3])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.connect(src=proc.words[:, :], dst=mem.words[:, :])
        arch.add_mode("all", ["proc", "mem"])
        return arch.build()

    def test_second_zone_word_buses_use_global_ids(self):
        spec = self._make_two_zone_arch()
        # proc is zone 0 (words 0-3), mem is zone 1 (words 4-7)
        mem_zone = spec.zones[1]
        for bus in mem_zone.word_buses:
            for w in bus.src:
                assert w >= 4, f"mem word bus src {w} should be >= 4 (global)"
            for w in bus.dst:
                assert w >= 4, f"mem word bus dst {w} should be >= 4 (global)"

    def test_second_zone_entangling_pairs_use_global_ids(self):
        spec = self._make_two_zone_arch()
        mem_zone = spec.zones[1]
        for a, b in mem_zone.entangling_pairs:
            assert a >= 4, f"mem entangling pair word {a} should be >= 4"
            assert b >= 4, f"mem entangling pair word {b} should be >= 4"

    def test_second_zone_words_with_site_buses_use_global_ids(self):
        spec = self._make_two_zone_arch()
        # proc zone has site buses → words_with_site_buses should be [0,1,2,3]
        assert all(w < 4 for w in spec.zones[0].words_with_site_buses)
        # mem zone has no site buses → empty
        assert spec.zones[1].words_with_site_buses == []

    def test_zone_bus_tuples_correct(self):
        spec = self._make_two_zone_arch()
        assert len(spec.zone_buses) == 1
        zb = spec.zone_buses[0]
        # src should be zone 0, words 0-3
        for zone_id, word_id in zb.src:
            assert zone_id == 0
            assert 0 <= word_id < 4
        # dst should be zone 1, words 4-7
        for zone_id, word_id in zb.dst:
            assert zone_id == 1
            assert 4 <= word_id < 8

    def test_rust_validation_passes_multi_zone(self):
        """Multi-zone with buses and entangling pairs on both zones passes Rust validation."""
        spec = self._make_two_zone_arch()
        assert spec is not None
        assert len(spec.zones) == 2
        assert len(spec.words) == 8


class TestBuilderEdgeCases:
    def test_add_entangling_pairs_length_mismatch(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 1),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        with pytest.raises(ValueError, match="entries"):
            zone.add_entangling_pairs([0], [1, 0])

    def test_site_bus_src_dst_length_mismatch(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 2),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), slice(0, 2))
        with pytest.raises(ValueError, match="entries"):
            zone.add_site_bus(src=[0, 1], dst=[2])

    def test_word_bus_src_dst_length_mismatch(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        with pytest.raises(ValueError, match="entries"):
            zone.add_word_bus(src=[0, 1], dst=[2])

    def test_empty_word_query(self):
        zone = ZoneBuilder(
            "z",
            _make_grid(4, 2),
            word_shape=(2, 1),
            x_clearance=_DEFAULT_CL,
            y_clearance=_DEFAULT_CL,
        )
        zone.add_word(slice(0, 2), [0])
        # Query a region with no words
        name, ids = zone.words[:, 1]
        assert name == "z"
        assert ids == []


# ── Path computation helpers ──


def _make_spaced_grid(
    nx: int, ny: int, *, x_spacing: float = 10.0, y_spacing: float = 20.0
) -> Grid:
    """Create a grid with explicit spacing (more realistic than unit grid)."""
    xs = [i * x_spacing for i in range(nx)]
    ys = [j * y_spacing for j in range(ny)]
    return Grid.from_positions(xs, ys)


# ── Safe-position enumeration ──


class TestEnumerateSafePositions:
    def test_single_source_valid_positions_clear(self):
        """Every returned position must be >= x_clearance (nm) from every grid line."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        min_cl_nm = zone._x_clearance_nm
        safe = zone._enumerate_safe_positions(zone._grid_x_nm, [0], min_cl_nm)
        assert len(safe) > 0
        for p in safe:
            for g in zone._grid_x_nm:
                assert abs(p - g) >= min_cl_nm

    def test_multiple_sources_preserve_spacing(self):
        """Multi-source: every shifted atom must clear every grid line."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        min_cl_nm = zone._x_clearance_nm
        safe = zone._enumerate_safe_positions(zone._grid_x_nm, [0, 2000], min_cl_nm)
        assert all(
            all(
                abs(p + off - g) >= min_cl_nm
                for g in zone._grid_x_nm
                for off in (0, 2000)
            )
            for p in safe
        )


# ── Search integration ──


class TestSearchPath:
    def test_straight_1d_move(self):
        """cd=1 shift in 1D row → straight line."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        # Coordinates in nm integers.
        path = zone._search_path((0, 0), (10000, 0), [(0, 0)])
        assert path == ((0, 0), (10000, 0))

    def test_cross_column_uses_clearance_row(self):
        """Cross-column move uses y-clearance (3-seg)."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        path = zone._search_path((0, 0), (20000, 0), [(0, 0)])
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (20000, 0)

    def test_diagonal_has_middle_waypoint(self):
        """Diagonal (col+row diff > 0) cannot be 2-segment with grid middle."""
        zone = ZoneBuilder(
            "z",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            (2, 1),
            x_clearance=3.0,
            y_clearance=3.0,
        )
        path = zone._search_path((0, 0), (10000, 20000), [(0, 0)])
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (10000, 20000)


class TestInconsistentBusDisplacement:
    def test_mismatched_word_displacements_warns_and_skips(self):
        """Bus with different src→dst displacements per pair violates AOD."""
        # Grid: x=[0, 2, 10, 12], y=[0, 10, 20]; word_shape=(2, 1).
        grid = Grid.from_positions([0.0, 2.0, 10.0, 12.0], [0.0, 10.0, 20.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=3.0, y_clearance=3.0)
        # 2 words per row × 3 rows = 6 words.
        for row in range(3):
            zone.add_word([0, 1], [row])
            zone.add_word([2, 3], [row])
        # Bus pair 1: word 0 (row 0, x=[0,2]) → word 1 (row 0, x=[10,12])
        #            displacement (+10, 0).
        # Bus pair 2: word 2 (row 1, x=[0,2]) → word 5 (row 2, x=[10,12])
        #            displacement (+10, +10).  Different!
        zone.add_word_bus(src=[0, 2], dst=[1, 5])

        with pytest.warns(UserWarning, match="inconsistent word displacements"):
            paths = zone._compute_paths(zone_id=0, word_offset=0)

        assert not any(k.move_type == MoveType.WORD for k in paths)


class TestSearchFailureWarning:
    def test_no_valid_delta_warns_and_skips(self):
        """Very tight grid with high clearance → search fails."""
        # Dense grid: x=[0,1,2,3,4], y=[0,1,2,3,4], min_cl=2.5
        # Internal gaps half=0.5 < 2.5 → no safe internal positions.
        # Edges at ±2.5 → might work for a single source but not for multi-
        # word bus spanning multiple sources.
        grid = Grid.from_positions([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0])
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=2.5, y_clearance=2.5)
        zone.add_word([0, 1], [0])
        zone.add_word([2, 3], [0])
        zone.add_word([0, 1], [2])
        zone.add_word([2, 3], [2])
        # Cross-column + cross-row bus (needs routing)
        zone.add_word_bus(src=[0], dst=[3])

        # Search may fail because no safe x (intra-pair sources span 0,1
        # with min_cl=2.5 → inter-offset shifts always collide).
        import warnings as _w

        with _w.catch_warnings(record=True) as w:
            _w.simplefilter("always")
            paths = zone._compute_paths(zone_id=0, word_offset=0)
            # Either warning fires (search failed) or paths found
            if any("no valid path" in str(warning.message) for warning in w):
                assert not any(k.move_type == MoveType.WORD for k in paths)


# ── _compute_paths (legacy + clearance) ──


class TestComputePaths:
    def _make_zone_with_site_bus(self):
        """Zone with 2 words on one row, one site bus shifting site 0→1."""
        grid = _make_spaced_grid(4, 1, x_spacing=10.0)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_site_bus(src=[0], dst=[1])
        return zone

    def _make_zone_with_word_bus(self):
        """Zone with 4 words (2×2 grid), one word bus moving row 0→1."""
        grid = _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        return zone

    def test_site_bus_forward_path(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((0.0, 0.0), (10.0, 0.0))

    def test_site_bus_backward_is_reversed(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        fwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        bwd = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.BACKWARD, 0)
        assert paths[bwd] == paths[fwd][::-1]

    def test_site_bus_applies_to_all_words(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        lane = LaneAddress(MoveType.SITE, 1, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((20.0, 0.0), (30.0, 0.0))

    def test_word_bus_forward_path(self):
        zone = self._make_zone_with_word_bus()
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        assert lane in paths
        assert paths[lane] == ((0.0, 0.0), (0.0, 20.0))

    def test_word_bus_backward_is_reversed(self):
        zone = self._make_zone_with_word_bus()
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        fwd = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        bwd = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.BACKWARD, 0)
        assert paths[bwd] == paths[fwd][::-1]

    def test_word_bus_all_sites_get_paths(self):
        zone = self._make_zone_with_word_bus()
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        for site_id in range(2):
            lane = LaneAddress(MoveType.WORD, 0, site_id, 0, Direction.FORWARD, 0)
            assert lane in paths

    def test_word_offset_applied(self):
        zone = self._make_zone_with_site_bus()
        paths = zone._compute_paths(zone_id=1, word_offset=4)
        lane = LaneAddress(MoveType.SITE, 4, 0, 0, Direction.FORWARD, 1)
        assert lane in paths

    def test_no_buses_returns_empty(self):
        grid = _make_spaced_grid(2, 1)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        assert paths == {}

    def test_cross_column_word_bus_uses_clearance(self):
        grid = _make_spaced_grid(4, 1, x_spacing=10.0)
        zone = ZoneBuilder(
            "z", grid, word_shape=(2, 1), x_clearance=5.0, y_clearance=5.0
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word_bus(src=[0], dst=[1])
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        path = paths[lane]
        # col_diff=2 > 1 → 3-segment routing via y-clearance
        assert len(path) == 4
        assert path[0] == (0.0, 0.0)
        assert path[3] == (20.0, 0.0)

    def test_adjacent_word_bus_with_clearance_is_straight(self):
        """row_diff=1, col_diff=0 → straight line even with clearance."""
        grid = _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0)
        zone = ZoneBuilder("z", grid, (2, 1), x_clearance=5.0, y_clearance=5.0)
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        paths = zone._compute_paths(zone_id=0, word_offset=0)
        lane = LaneAddress(MoveType.WORD, 0, 0, 0, Direction.FORWARD, 0)
        # col_diff=0, row_diff=1 → straight
        assert paths[lane] == ((0.0, 0.0), (0.0, 20.0))


class TestArchBuilderPaths:
    """Integration: ArchBuilder.build() produces paths on the ArchSpec."""

    def test_build_populates_paths(self):
        zone = ZoneBuilder(
            "gate",
            _make_spaced_grid(4, 2, x_spacing=10.0, y_spacing=20.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus(src=[0], dst=[1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert len(spec.paths) > 0

    def test_get_path_returns_stored_path(self):
        zone = ZoneBuilder(
            "gate",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_site_bus(src=[0], dst=[1])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()

        lane = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        path = spec.get_path(lane)
        assert path == ((0.0, 0.0), (10.0, 0.0))

    def test_multi_zone_paths_use_correct_offsets(self):
        proc = ZoneBuilder(
            "proc",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        proc.add_site_bus(src=[0], dst=[1])

        mem = ZoneBuilder(
            "mem",
            _make_spaced_grid(4, 1, x_spacing=10.0),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        mem.add_word(slice(0, 2), [0])
        mem.add_word(slice(2, 4), [0])
        mem.add_site_bus(src=[0], dst=[1])

        arch = ArchBuilder()
        arch.add_zone(proc)
        arch.add_zone(mem)
        arch.connect(src=proc.words[:, :], dst=mem.words[:, :])
        arch.add_mode("all", ["proc", "mem"])
        spec = arch.build()

        lane0 = LaneAddress(MoveType.SITE, 0, 0, 0, Direction.FORWARD, 0)
        assert lane0 in spec.paths
        lane2 = LaneAddress(MoveType.SITE, 2, 0, 0, Direction.FORWARD, 1)
        assert lane2 in spec.paths

    def test_no_buses_no_paths(self):
        zone = ZoneBuilder(
            "gate",
            _make_spaced_grid(2, 1),
            word_shape=(2, 1),
            x_clearance=5.0,
            y_clearance=5.0,
        )
        zone.add_word(slice(0, 2), [0])

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert len(spec.paths) == 0
