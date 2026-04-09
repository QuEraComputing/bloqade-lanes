"""Tests for ZoneBuilder and ArchBuilder."""

import pytest

from bloqade.lanes.arch.arch_builder import (
    ArchBuilder,
    ZoneBuilder,
    _SiteGridQuery,
    _validate_aod_rectangle,
)
from bloqade.lanes.bytecode._native import Grid


def _make_grid(nx: int, ny: int) -> Grid:
    """Create a simple grid with unit spacing."""
    xs = [float(i) for i in range(nx)]
    ys = [float(j) for j in range(ny)]
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
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        wid = zone.add_word(x_sites=slice(0, 2), y_sites=slice(0, 1))
        assert wid == 0

    def test_add_multiple_words(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        assert zone.add_word(slice(0, 2), slice(0, 1)) == 0
        assert zone.add_word(slice(2, 4), slice(0, 1)) == 1

    def test_shape_mismatch_x_raises(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        with pytest.raises(ValueError, match="word_shape requires 2"):
            zone.add_word(slice(0, 3), slice(0, 1))

    def test_shape_mismatch_y_raises(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        with pytest.raises(ValueError, match="word_shape requires 1"):
            zone.add_word(slice(0, 2), slice(0, 2))

    def test_overlap_raises(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        zone.add_word(slice(0, 2), slice(0, 1))
        with pytest.raises(ValueError, match="already belongs to word 0"):
            zone.add_word(slice(1, 3), slice(0, 1))

    def test_out_of_bounds_raises(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        with pytest.raises(IndexError):
            zone.add_word([3, 4], [0])

    def test_list_indices(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        wid = zone.add_word([0, 2], [0])
        assert wid == 0

    def test_num_words(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        assert zone.num_words == 0
        zone.add_word(slice(0, 2), [0])
        assert zone.num_words == 1

    def test_sites_per_word(self):
        zone = ZoneBuilder("gate", _make_grid(4, 4), word_shape=(2, 3))
        assert zone.sites_per_word == 6


# ── ZoneBuilder: grid queries ──


class TestZoneBuilderQueries:
    def test_words_query_all(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        name, ids = zone.words[:, :]
        assert name == "gate"
        assert ids == [0, 1, 2, 3]

    def test_words_query_first_column(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        name, ids = zone.words[slice(0, 2), :]
        assert name == "gate"
        assert ids == [0, 2]

    def test_words_query_returns_zone_name(self):
        zone = ZoneBuilder("proc", _make_grid(2, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        name, _ = zone.words[:, :]
        assert name == "proc"

    def test_sites_query(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(3, 2))
        assert zone.sites[:, 0] == [0, 1, 2]
        assert zone.sites[:, 1] == [3, 4, 5]
        assert zone.sites[:, :] == [0, 1, 2, 3, 4, 5]


# ── ZoneBuilder: buses ──


class TestZoneBuilderBuses:
    def test_add_site_bus_valid(self):
        zone = ZoneBuilder("z", _make_grid(4, 2), word_shape=(2, 2))
        zone.add_word(slice(0, 2), slice(0, 2))
        zone.add_site_bus(src=[0, 1], dst=[2, 3])

    def test_add_site_bus_invalid_rectangle(self):
        zone = ZoneBuilder("z", _make_grid(4, 2), word_shape=(3, 2))
        zone.add_word(slice(0, 3), slice(0, 2))
        # src=[0, 5] → positions (0,0) and (2,1) — diagonal, not a rectangle
        with pytest.raises(ValueError, match="Cartesian product"):
            zone.add_site_bus(src=[0, 5], dst=[1, 4])

    def test_add_site_bus_out_of_range(self):
        zone = ZoneBuilder("z", _make_grid(4, 2), word_shape=(2, 2))
        zone.add_word(slice(0, 2), slice(0, 2))
        with pytest.raises(ValueError, match="out of range"):
            zone.add_site_bus(src=[0, 4], dst=[1, 5])

    def test_add_word_bus_valid(self):
        zone = ZoneBuilder("z", _make_grid(4, 2), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])

    def test_add_word_bus_out_of_range(self):
        zone = ZoneBuilder("z", _make_grid(4, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        with pytest.raises(ValueError, match="out of range"):
            zone.add_word_bus(src=[0], dst=[1])

    def test_add_entangling_pair(self):
        zone = ZoneBuilder("z", _make_grid(4, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_entangling_pair(0, 1)

    def test_add_entangling_pair_out_of_range(self):
        zone = ZoneBuilder("z", _make_grid(4, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        with pytest.raises(ValueError, match="out of range"):
            zone.add_entangling_pair(0, 1)


# ── ArchBuilder ──


class TestArchBuilder:
    def test_single_zone(self):
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
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
        zone = ZoneBuilder("my_zone", _make_grid(2, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["my_zone"])
        spec = arch.build()
        assert spec.zones[0].name == "my_zone"

    def test_duplicate_zone_name_raises(self):
        z1 = ZoneBuilder("gate", _make_grid(2, 1), word_shape=(2, 1))
        z1.add_word(slice(0, 2), [0])
        z2 = ZoneBuilder("gate", _make_grid(2, 1), word_shape=(2, 1))
        z2.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(z1)
        with pytest.raises(ValueError, match="Duplicate zone name"):
            arch.add_zone(z2)

    def test_sites_per_word_mismatch_raises(self):
        z1 = ZoneBuilder("a", _make_grid(4, 1), word_shape=(2, 1))
        z1.add_word(slice(0, 2), [0])
        z2 = ZoneBuilder("b", _make_grid(4, 1), word_shape=(4, 1))
        z2.add_word(slice(0, 4), [0])
        arch = ArchBuilder()
        arch.add_zone(z1)
        with pytest.raises(ValueError, match="sites_per_word"):
            arch.add_zone(z2)

    def test_multi_zone_with_connection(self):
        proc = ZoneBuilder("proc", _make_grid(4, 2), word_shape=(2, 1))
        proc.add_word(slice(0, 2), [0])
        proc.add_word(slice(2, 4), [0])
        mem = ZoneBuilder("mem", _make_grid(4, 2), word_shape=(2, 1))
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
        zone = ZoneBuilder("gate", _make_grid(2, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        with pytest.raises(ValueError, match="Unknown zone"):
            arch.connect(src=("gate", [0]), dst=("missing", [0]))

    def test_unknown_zone_in_mode_raises(self):
        zone = ZoneBuilder("gate", _make_grid(2, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        arch = ArchBuilder()
        arch.add_zone(zone)
        with pytest.raises(ValueError, match="Unknown zone"):
            arch.add_mode("test", ["missing"])

    def test_global_word_ids_assigned(self):
        z1 = ZoneBuilder("a", _make_grid(4, 1), word_shape=(2, 1))
        z1.add_word(slice(0, 2), [0])
        z1.add_word(slice(2, 4), [0])
        z2 = ZoneBuilder("b", _make_grid(4, 1), word_shape=(2, 1))
        z2.add_word(slice(0, 2), [0])
        z2.add_word(slice(2, 4), [0])

        arch = ArchBuilder()
        arch.add_zone(z1)
        arch.add_zone(z2)
        arch.add_mode("all", ["a", "b"])
        spec = arch.build()
        assert len(spec.words) == 4

    def test_entangling_pairs_in_single_zone(self):
        zone = ZoneBuilder("gate", _make_grid(4, 1), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_entangling_pair(0, 1)

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec.zones[0].entangling_pairs == [(0, 1)]

    def test_rust_validation_passes(self):
        """Build a realistic arch and verify Rust validation."""
        zone = ZoneBuilder("gate", _make_grid(4, 2), word_shape=(2, 1))
        zone.add_word(slice(0, 2), [0])
        zone.add_word(slice(2, 4), [0])
        zone.add_word(slice(0, 2), [1])
        zone.add_word(slice(2, 4), [1])
        zone.add_site_bus(src=[0], dst=[1])
        zone.add_word_bus(src=[0, 1], dst=[2, 3])
        zone.add_entangling_pair(0, 1)
        zone.add_entangling_pair(2, 3)

        arch = ArchBuilder()
        arch.add_zone(zone)
        arch.add_mode("all", ["gate"])
        spec = arch.build()
        assert spec is not None
