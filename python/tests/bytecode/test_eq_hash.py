"""Tests for equality and hash contracts across all Rust-backed types.

Verifies:
- Equal objects have equal hashes
- Non-equal objects (where possible) have different hashes
- -0.0 vs 0.0 produces equal objects with equal hashes
- NaN/Inf are rejected at construction
- Objects work correctly as dict keys and set members
"""

import pytest

from bloqade.lanes.bytecode._native import (
    ArchSpec,
    Bus,
    Buses,
    Direction,
    Geometry,
    Grid,
    LaneAddress,
    LocationAddress,
    MoveType,
    TransportPath,
    Word,
    Zone,
    ZoneAddress,
)

# ── Direction / MoveType ──


class TestDirectionEqHash:
    def test_equal(self):
        assert Direction.FORWARD == Direction.FORWARD
        assert Direction.BACKWARD == Direction.BACKWARD

    def test_not_equal(self):
        assert Direction.FORWARD != Direction.BACKWARD

    def test_hash_equal(self):
        assert hash(Direction.FORWARD) == hash(Direction.FORWARD)

    def test_hash_different(self):
        assert hash(Direction.FORWARD) != hash(Direction.BACKWARD)

    def test_as_dict_key(self):
        d = {Direction.FORWARD: "fwd", Direction.BACKWARD: "bwd"}
        assert d[Direction.FORWARD] == "fwd"

    def test_as_set_member(self):
        s = {Direction.FORWARD, Direction.BACKWARD, Direction.FORWARD}
        assert len(s) == 2


class TestMoveTypeEqHash:
    def test_equal(self):
        assert MoveType.SITE == MoveType.SITE

    def test_hash_equal(self):
        assert hash(MoveType.SITE) == hash(MoveType.SITE)

    def test_as_set_member(self):
        s = {MoveType.SITE, MoveType.WORD, MoveType.SITE}
        assert len(s) == 2


# ── LocationAddress ──


class TestLocationAddressEqHash:
    def test_equal(self):
        a = LocationAddress(1, 2)
        b = LocationAddress(1, 2)
        assert a == b

    def test_not_equal_word(self):
        assert LocationAddress(0, 1) != LocationAddress(1, 1)

    def test_not_equal_site(self):
        assert LocationAddress(1, 0) != LocationAddress(1, 1)

    def test_hash_equal(self):
        a = LocationAddress(1, 2)
        b = LocationAddress(1, 2)
        assert hash(a) == hash(b)

    def test_hash_different(self):
        assert hash(LocationAddress(0, 0)) != hash(LocationAddress(0, 1))

    def test_as_dict_key(self):
        loc = LocationAddress(3, 7)
        d = {loc: "value"}
        assert d[LocationAddress(3, 7)] == "value"

    def test_zero_ids(self):
        a = LocationAddress(0, 0)
        b = LocationAddress(0, 0)
        assert a == b
        assert hash(a) == hash(b)

    def test_max_ids(self):
        a = LocationAddress(0xFFFF, 0xFFFF)
        b = LocationAddress(0xFFFF, 0xFFFF)
        assert a == b
        assert hash(a) == hash(b)


# ── LaneAddress ──


class TestLaneAddressEqHash:
    def test_equal(self):
        a = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD)
        b = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_direction(self):
        a = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.FORWARD)
        b = LaneAddress(MoveType.SITE, 1, 2, 3, Direction.BACKWARD)
        assert a != b

    def test_different_move_type(self):
        a = LaneAddress(MoveType.SITE, 1, 2, 3)
        b = LaneAddress(MoveType.WORD, 1, 2, 3)
        assert a != b

    def test_as_dict_key(self):
        lane = LaneAddress(MoveType.WORD, 0, 1, 0)
        d = {lane: "x"}
        assert d[LaneAddress(MoveType.WORD, 0, 1, 0)] == "x"


# ── ZoneAddress ──


class TestZoneAddressEqHash:
    def test_equal(self):
        a = ZoneAddress(5)
        b = ZoneAddress(5)
        assert a == b
        assert hash(a) == hash(b)

    def test_not_equal(self):
        assert ZoneAddress(0) != ZoneAddress(1)

    def test_as_set_member(self):
        s = {ZoneAddress(0), ZoneAddress(1), ZoneAddress(0)}
        assert len(s) == 2


# ── Grid ──


class TestGridEqHash:
    def test_equal(self):
        a = Grid(1.0, 2.0, [3.0], [4.0])
        b = Grid(1.0, 2.0, [3.0], [4.0])
        assert a == b
        assert hash(a) == hash(b)

    def test_different_start(self):
        a = Grid(1.0, 2.0, [3.0], [4.0])
        b = Grid(1.5, 2.0, [3.0], [4.0])
        assert a != b

    def test_different_spacing(self):
        a = Grid(1.0, 2.0, [3.0], [4.0])
        b = Grid(1.0, 2.0, [3.5], [4.0])
        assert a != b

    def test_negative_zero_equals_zero(self):
        a = Grid(0.0, 0.0, [1.0], [1.0])
        b = Grid(-0.0, -0.0, [1.0], [1.0])
        assert a == b
        assert hash(a) == hash(b)

    def test_negative_zero_in_spacing(self):
        a = Grid(1.0, 2.0, [0.0, 1.0], [0.0])
        b = Grid(1.0, 2.0, [-0.0, 1.0], [-0.0])
        assert a == b
        assert hash(a) == hash(b)

    def test_empty_spacing(self):
        a = Grid(1.0, 2.0, [], [])
        b = Grid(1.0, 2.0, [], [])
        assert a == b
        assert hash(a) == hash(b)

    def test_nan_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            Grid(float("nan"), 0.0, [], [])

    def test_inf_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            Grid(0.0, 0.0, [float("inf")], [])

    def test_neg_inf_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            Grid(0.0, 0.0, [], [float("-inf")])

    def test_as_dict_key(self):
        g = Grid(1.0, 2.0, [3.0], [4.0])
        d = {g: "grid"}
        assert d[Grid(1.0, 2.0, [3.0], [4.0])] == "grid"


# ── Bus ──


class TestBusEqHash:
    def test_equal(self):
        a = Bus(src=[0, 1], dst=[2, 3])
        b = Bus(src=[0, 1], dst=[2, 3])
        assert a == b
        assert hash(a) == hash(b)

    def test_different_src(self):
        a = Bus(src=[0, 1], dst=[2, 3])
        b = Bus(src=[0, 2], dst=[2, 3])
        assert a != b

    def test_different_dst(self):
        a = Bus(src=[0, 1], dst=[2, 3])
        b = Bus(src=[0, 1], dst=[2, 4])
        assert a != b

    def test_empty(self):
        a = Bus(src=[], dst=[])
        b = Bus(src=[], dst=[])
        assert a == b
        assert hash(a) == hash(b)

    def test_as_set_member(self):
        s = {Bus(src=[0], dst=[1]), Bus(src=[0], dst=[1]), Bus(src=[1], dst=[0])}
        assert len(s) == 2


# ── TransportPath ──


class TestTransportPathEqHash:
    def _lane(self):
        return LaneAddress(MoveType.SITE, 0, 0, 0)

    def test_equal(self):
        a = TransportPath(self._lane(), [(1.0, 2.0), (3.0, 4.0)])
        b = TransportPath(self._lane(), [(1.0, 2.0), (3.0, 4.0)])
        assert a == b
        assert hash(a) == hash(b)

    def test_different_waypoints(self):
        a = TransportPath(self._lane(), [(1.0, 2.0), (3.0, 4.0)])
        b = TransportPath(self._lane(), [(1.0, 2.0), (3.0, 5.0)])
        assert a != b

    def test_negative_zero_waypoints(self):
        a = TransportPath(self._lane(), [(0.0, 0.0)])
        b = TransportPath(self._lane(), [(-0.0, -0.0)])
        assert a == b
        assert hash(a) == hash(b)

    def test_nan_waypoint_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            TransportPath(self._lane(), [(float("nan"), 1.0)])

    def test_inf_waypoint_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            TransportPath(self._lane(), [(1.0, float("inf"))])


# ── ArchSpec ──


def _minimal_arch_spec(
    entangling_zones: list[list[tuple[int, int]]] | None = None,
) -> ArchSpec:
    """Build a minimal valid ArchSpec for testing."""
    grid = Grid(0.0, 0.0, [], [])
    word = Word(positions=grid, site_indices=[(0, 0)])
    return ArchSpec(
        version=(2, 0),
        geometry=Geometry(sites_per_word=1, words=[word]),
        buses=Buses(site_buses=[], word_buses=[]),
        words_with_site_buses=[],
        sites_with_word_buses=[],
        zones=[Zone(words=[0])],
        entangling_zones=entangling_zones if entangling_zones is not None else [],
        measurement_mode_zones=[0],
    )


class TestArchSpecEqHash:
    def test_equal(self):
        a = _minimal_arch_spec()
        b = _minimal_arch_spec()
        assert a == b
        assert hash(a) == hash(b)

    def test_different_entangling_zones(self):
        a = _minimal_arch_spec(entangling_zones=[])
        b = _minimal_arch_spec(entangling_zones=[[(0, 0)]])
        assert a != b

    def test_as_dict_key(self):
        spec = _minimal_arch_spec()
        d = {spec: "arch"}
        assert d[_minimal_arch_spec()] == "arch"
