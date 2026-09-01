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
    Direction,
    Grid,
    LaneAddress,
    LocationAddress,
    Mode,
    MoveType,
    SiteBus,
    TransportPath,
    Word,
    WordBus,
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
        a = LocationAddress(0, 1, 2)
        b = LocationAddress(0, 1, 2)
        assert a == b

    def test_not_equal_word(self):
        assert LocationAddress(0, 0, 1) != LocationAddress(0, 1, 1)

    def test_not_equal_site(self):
        assert LocationAddress(0, 1, 0) != LocationAddress(0, 1, 1)

    def test_hash_equal(self):
        a = LocationAddress(0, 1, 2)
        b = LocationAddress(0, 1, 2)
        assert hash(a) == hash(b)

    def test_hash_different(self):
        assert hash(LocationAddress(0, 0, 0)) != hash(LocationAddress(0, 0, 1))

    def test_as_dict_key(self):
        loc = LocationAddress(0, 3, 7)
        d = {loc: "value"}
        assert d[LocationAddress(0, 3, 7)] == "value"

    def test_zero_ids(self):
        a = LocationAddress(0, 0, 0)
        b = LocationAddress(0, 0, 0)
        assert a == b
        assert hash(a) == hash(b)

    def test_max_ids(self):
        a = LocationAddress(0xFF, 0xFFFF, 0xFFFF)
        b = LocationAddress(0xFF, 0xFFFF, 0xFFFF)
        assert a == b
        assert hash(a) == hash(b)


# ── LaneAddress ──


class TestLaneAddressEqHash:
    def test_equal(self):
        a = LaneAddress(MoveType.SITE, 0, 1, 2, 3, Direction.FORWARD)
        b = LaneAddress(MoveType.SITE, 0, 1, 2, 3, Direction.FORWARD)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_direction(self):
        a = LaneAddress(MoveType.SITE, 0, 1, 2, 3, Direction.FORWARD)
        b = LaneAddress(MoveType.SITE, 0, 1, 2, 3, Direction.BACKWARD)
        assert a != b

    def test_different_move_type(self):
        a = LaneAddress(MoveType.SITE, 0, 1, 2, 3)
        b = LaneAddress(MoveType.WORD, 0, 1, 2, 3)
        assert a != b

    def test_as_dict_key(self):
        lane = LaneAddress(MoveType.WORD, 0, 0, 1, 0)
        d = {lane: "x"}
        assert d[LaneAddress(MoveType.WORD, 0, 0, 1, 0)] == "x"


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

    def test_from_positions_nan_rejected(self):
        """``from_positions`` must guard finiteness just like ``__init__``.

        A NaN grid is unequal to itself, which breaks the ``Eq`` contract the
        Rust ``Grid`` asserts — and any ``Zone`` embedding it (see #476).
        """
        with pytest.raises(ValueError, match="x_positions.*non-finite"):
            Grid.from_positions([0.0, float("nan"), 2.0], [0.0])

    def test_from_positions_inf_rejected(self):
        with pytest.raises(ValueError, match="y_positions.*non-finite"):
            Grid.from_positions([0.0], [0.0, float("inf")])

    def test_as_dict_key(self):
        g = Grid(1.0, 2.0, [3.0], [4.0])
        d = {g: "grid"}
        assert d[Grid(1.0, 2.0, [3.0], [4.0])] == "grid"


# ── Word ──


class TestWordEqHash:
    def test_equal(self):
        a = Word(sites=[(0, 0), (1, 0)])
        b = Word(sites=[(0, 0), (1, 0)])
        assert a == b
        assert hash(a) == hash(b)

    def test_different_site(self):
        a = Word(sites=[(0, 0), (1, 0)])
        b = Word(sites=[(0, 0), (1, 1)])
        assert a != b

    def test_site_order_matters(self):
        a = Word(sites=[(0, 0), (1, 0)])
        b = Word(sites=[(1, 0), (0, 0)])
        assert a != b

    def test_empty(self):
        assert Word(sites=[]) == Word(sites=[])
        assert hash(Word(sites=[])) == hash(Word(sites=[]))

    def test_not_equal_to_other_type(self):
        assert Word(sites=[(0, 0)]) != 5

    def test_as_dict_key(self):
        d = {Word(sites=[(0, 0)]): "w"}
        assert d[Word(sites=[(0, 0)])] == "w"

    def test_as_set_member(self):
        s = {Word(sites=[(0, 0)]), Word(sites=[(0, 0)]), Word(sites=[(1, 1)])}
        assert len(s) == 2


# ── SiteBus ──


class TestSiteBusEqHash:
    def test_equal(self):
        a = SiteBus(src=[0, 1], dst=[2, 3])
        b = SiteBus(src=[0, 1], dst=[2, 3])
        assert a == b
        assert hash(a) == hash(b)

    def test_different_src(self):
        a = SiteBus(src=[0, 1], dst=[2, 3])
        b = SiteBus(src=[0, 2], dst=[2, 3])
        assert a != b

    def test_different_dst(self):
        a = SiteBus(src=[0, 1], dst=[2, 3])
        b = SiteBus(src=[0, 1], dst=[2, 4])
        assert a != b

    def test_empty(self):
        a = SiteBus(src=[], dst=[])
        b = SiteBus(src=[], dst=[])
        assert a == b
        assert hash(a) == hash(b)

    def test_as_set_member(self):
        s = {
            SiteBus(src=[0], dst=[1]),
            SiteBus(src=[0], dst=[1]),
            SiteBus(src=[1], dst=[0]),
        }
        assert len(s) == 2


# ── Zone ──


def _zone(
    name: str = "gate",
    grid: Grid | None = None,
    site_buses: list[SiteBus] | None = None,
    entangling_pairs: list[tuple[int, int]] | None = None,
) -> Zone:
    """Build a small zone; each argument overrides one field of the default."""
    return Zone(
        name=name,
        grid=Grid(0.0, 0.0, [1.0], [1.0]) if grid is None else grid,
        site_buses=[SiteBus(src=[0], dst=[1])] if site_buses is None else site_buses,
        word_buses=[WordBus(src=[0], dst=[1])],
        words_with_site_buses=[0],
        sites_with_word_buses=[0],
        entangling_pairs=[(0, 1)] if entangling_pairs is None else entangling_pairs,
    )


class TestZoneEqHash:
    def test_equal(self):
        assert _zone() == _zone()
        assert hash(_zone()) == hash(_zone())

    def test_different_name(self):
        assert _zone("gate") != _zone("storage")

    def test_different_grid(self):
        assert _zone() != _zone(grid=Grid(0.0, 0.0, [2.0], [1.0]))

    def test_different_buses(self):
        assert _zone() != _zone(site_buses=[SiteBus(src=[1], dst=[0])])

    def test_different_entangling_pairs(self):
        assert _zone() != _zone(entangling_pairs=[])

    def test_not_equal_to_other_type(self):
        assert _zone() != 5

    def test_as_dict_key(self):
        d = {_zone(): "z"}
        assert d[_zone()] == "z"

    def test_as_set_member(self):
        assert len({_zone(), _zone(), _zone("storage")}) == 2

    def test_negative_zero_grid_equals_zero(self):
        """Zone equality/hashing inherits Grid's -0.0 normalization."""
        a = _zone(grid=Grid(0.0, 0.0, [1.0], [1.0]))
        b = _zone(grid=Grid(-0.0, -0.0, [1.0], [1.0]))
        assert a == b
        assert hash(a) == hash(b)


# ── Mode ──


class TestModeEqHash:
    def _order(self):
        return [LocationAddress(0, 0, 0), LocationAddress(0, 1, 0)]

    def test_equal(self):
        a = Mode(name="all", zones=[0], bitstring_order=self._order())
        b = Mode(name="all", zones=[0], bitstring_order=self._order())
        assert a == b
        assert hash(a) == hash(b)

    def test_different_name(self):
        a = Mode(name="all", zones=[0], bitstring_order=self._order())
        b = Mode(name="gate", zones=[0], bitstring_order=self._order())
        assert a != b

    def test_different_zones(self):
        a = Mode(name="all", zones=[0], bitstring_order=self._order())
        b = Mode(name="all", zones=[0, 1], bitstring_order=self._order())
        assert a != b

    def test_bitstring_order_matters(self):
        a = Mode(name="all", zones=[0], bitstring_order=self._order())
        b = Mode(name="all", zones=[0], bitstring_order=self._order()[::-1])
        assert a != b

    def test_not_equal_to_other_type(self):
        assert Mode(name="all", zones=[0], bitstring_order=[]) != 5

    def test_as_dict_key(self):
        d = {Mode(name="all", zones=[0], bitstring_order=self._order()): "m"}
        assert d[Mode(name="all", zones=[0], bitstring_order=self._order())] == "m"

    def test_as_set_member(self):
        s = {
            Mode(name="all", zones=[0], bitstring_order=self._order()),
            Mode(name="all", zones=[0], bitstring_order=self._order()),
            Mode(name="gate", zones=[0], bitstring_order=self._order()),
        }
        assert len(s) == 2


# ── TransportPath ──


class TestTransportPathEqHash:
    def _lane(self):
        return LaneAddress(MoveType.SITE, 0, 0, 0, 0)

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
    entangling_pairs: list[tuple[int, int]] | None = None,
) -> ArchSpec:
    """Build a minimal valid ArchSpec for testing."""
    grid = Grid(0.0, 0.0, [], [])
    word = Word(sites=[(0, 0)])
    words = [word]
    if entangling_pairs:
        # Need a second word for valid entangling pairs
        words = [word, Word(sites=[(0, 0)])]
    zone = Zone(
        name="gate",
        grid=grid,
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[],
        sites_with_word_buses=[],
        entangling_pairs=entangling_pairs,
    )
    bitstring_order = [
        LocationAddress(0, w, s) for w in range(len(words)) for s in range(1)
    ]
    mode = Mode(
        name="all",
        zones=[0],
        bitstring_order=bitstring_order,
    )
    return ArchSpec(
        version=(2, 0),
        words=words,
        zones=[zone],
        zone_buses=[],
        modes=[mode],
    )


class TestArchSpecEqHash:
    def test_equal(self):
        a = _minimal_arch_spec()
        b = _minimal_arch_spec()
        assert a == b

    def test_different_entangling_zones(self):
        a = _minimal_arch_spec(entangling_pairs=[])
        b = _minimal_arch_spec(entangling_pairs=[(0, 1)])
        assert a != b

    def test_not_hashable(self):
        """ArchSpec is not hashable (mutable compound type)."""
        spec = _minimal_arch_spec()
        with pytest.raises(TypeError):
            hash(spec)
