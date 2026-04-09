from bloqade.lanes.bytecode import (
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
)
from bloqade.lanes.bytecode.exceptions import (
    LaneGroupError,
    LocationGroupError,
)


def _make_word():
    return Word(
        sites=[
            (0, 0),
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (0, 1),
            (1, 1),
            (2, 1),
            (3, 1),
            (4, 1),
        ]
    )


def _build_spec_from_python():
    word0 = _make_word()
    word1 = _make_word()
    grid = Grid(
        x_start=1.0,
        y_start=2.5,
        x_spacing=[2.0, 2.0, 2.0, 2.0],
        y_spacing=[2.5],
    )
    site_bus = SiteBus(src=[0, 1, 2, 3, 4], dst=[5, 6, 7, 8, 9])
    word_bus = WordBus(src=[0], dst=[1])
    zone = Zone(
        grid=grid,
        site_buses=[site_bus],
        word_buses=[word_bus],
        words_with_site_buses=[0, 1],
        sites_with_word_buses=[5, 6, 7, 8, 9],
        entangling_pairs=[(0, 1)],
    )
    mode = Mode(
        name="all",
        zones=[0],
        bitstring_order=[LocationAddress(0, w, s) for w in range(2) for s in range(10)],
    )
    return ArchSpec(
        version=(2, 0),
        words=[word0, word1],
        zones=[zone],
        zone_buses=[],
        modes=[mode],
        paths=[
            TransportPath(
                lane=LaneAddress(
                    MoveType.WORD,
                    zone_id=0,
                    word_id=0,
                    site_id=5,
                    bus_id=0,
                    direction=Direction.BACKWARD,
                ),
                waypoints=[(1.0, 15.0), (1.0, 10.0), (1.0, 5.0)],
            )
        ],
    )


class TestConstructFromPython:
    def test_build_and_validate(self):
        spec = _build_spec_from_python()
        spec.validate()  # should not raise

    def test_version(self):
        spec = _build_spec_from_python()
        assert spec.version == (2, 0)

    def test_words(self):
        spec = _build_spec_from_python()
        assert len(spec.words) == 2
        assert spec.sites_per_word == 10

    def test_word_sites(self):
        word = Word(sites=[(0, 0)])
        assert len(word.sites) == 1


class TestCapabilityFlags:
    def test_defaults_to_false(self):
        spec = _build_spec_from_python()
        assert spec.feed_forward is False
        assert spec.atom_reloading is False

    def test_construct_from_python_explicit(self):
        word0 = _make_word()
        word1 = _make_word()
        grid = Grid(
            x_start=1.0,
            y_start=2.5,
            x_spacing=[2.0, 2.0, 2.0, 2.0],
            y_spacing=[2.5],
        )
        site_bus = SiteBus(src=[0, 1, 2, 3, 4], dst=[5, 6, 7, 8, 9])
        word_bus = WordBus(src=[0], dst=[1])
        zone = Zone(
            grid=grid,
            site_buses=[site_bus],
            word_buses=[word_bus],
            words_with_site_buses=[0, 1],
            sites_with_word_buses=[5, 6, 7, 8, 9],
            entangling_pairs=[(0, 1)],
        )
        mode = Mode(
            name="all",
            zones=[0],
            bitstring_order=[LocationAddress(0, 0, 0)],
        )
        spec = ArchSpec(
            version=(2, 0),
            words=[word0, word1],
            zones=[zone],
            zone_buses=[],
            modes=[mode],
            feed_forward=True,
            atom_reloading=True,
        )
        assert spec.feed_forward is True
        assert spec.atom_reloading is True


class TestPropertyAccess:
    def test_zones(self):
        spec = _build_spec_from_python()
        assert len(spec.zones) == 1

    def test_zone_buses(self):
        spec = _build_spec_from_python()
        zone = spec.zones[0]
        assert len(zone.site_buses) == 1
        assert len(zone.word_buses) == 1
        sb = zone.site_buses[0]
        assert sb.src == [0, 1, 2, 3, 4]
        assert sb.dst == [5, 6, 7, 8, 9]

    def test_paths(self):
        spec = _build_spec_from_python()
        assert spec.paths is not None
        assert len(spec.paths) == 1
        lane = spec.paths[0].lane
        assert lane.direction == Direction.BACKWARD
        assert lane.move_type == MoveType.WORD
        assert lane.word_id == 0
        assert lane.site_id == 5
        assert lane.bus_id == 0
        assert len(spec.paths[0].waypoints) == 3


class TestQueryMethods:
    def test_word_by_id(self):
        spec = _build_spec_from_python()
        word = spec.word_by_id(0)
        assert word is not None
        assert len(word.sites) == 10
        assert spec.word_by_id(99) is None

    def test_zone_by_id(self):
        spec = _build_spec_from_python()
        zone = spec.zone_by_id(0)
        assert zone is not None
        assert spec.zone_by_id(99) is None


class TestBusResolution:
    def test_site_bus_resolve_forward(self):
        spec = _build_spec_from_python()
        bus = spec.zones[0].site_buses[0]
        assert bus.resolve_forward(0) == 5
        assert bus.resolve_forward(4) == 9
        assert bus.resolve_forward(99) is None

    def test_site_bus_resolve_backward(self):
        spec = _build_spec_from_python()
        bus = spec.zones[0].site_buses[0]
        assert bus.resolve_backward(5) == 0
        assert bus.resolve_backward(9) == 4
        assert bus.resolve_backward(99) is None

    def test_word_bus_resolve_forward(self):
        spec = _build_spec_from_python()
        bus = spec.zones[0].word_buses[0]
        assert bus.resolve_forward(0) == 1
        assert bus.resolve_forward(99) is None

    def test_word_bus_resolve_backward(self):
        spec = _build_spec_from_python()
        bus = spec.zones[0].word_buses[0]
        assert bus.resolve_backward(1) == 0
        assert bus.resolve_backward(99) is None


class TestLocationPosition:
    def test_valid(self):
        spec = _build_spec_from_python()
        loc = LocationAddress(zone_id=0, word_id=0, site_id=0)
        pos = spec.location_position(loc)
        assert pos is not None

    def test_invalid_word(self):
        spec = _build_spec_from_python()
        loc = LocationAddress(zone_id=0, word_id=99, site_id=0)
        assert spec.location_position(loc) is None


class TestCheckLocations:
    def test_valid(self):
        spec = _build_spec_from_python()
        locs = [LocationAddress(zone_id=0, word_id=0, site_id=0)]
        errors = spec.check_locations(locs)
        assert errors == []

    def test_invalid(self):
        spec = _build_spec_from_python()
        locs = [LocationAddress(zone_id=0, word_id=99, site_id=0)]
        errors = spec.check_locations(locs)
        assert len(errors) > 0
        assert any(isinstance(e, LocationGroupError) for e in errors)

    def test_duplicate(self):
        spec = _build_spec_from_python()
        locs = [
            LocationAddress(zone_id=0, word_id=0, site_id=0),
            LocationAddress(zone_id=0, word_id=0, site_id=1),
            LocationAddress(zone_id=0, word_id=0, site_id=0),
        ]
        errors = spec.check_locations(locs)
        assert len(errors) > 0


class TestCheckLanes:
    def test_valid(self):
        spec = _build_spec_from_python()
        lanes = [
            LaneAddress(MoveType.SITE, zone_id=0, word_id=0, site_id=0, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert errors == []

    def test_invalid_bus(self):
        spec = _build_spec_from_python()
        lanes = [
            LaneAddress(MoveType.SITE, zone_id=0, word_id=0, site_id=0, bus_id=99),
        ]
        errors = spec.check_lanes(lanes)
        assert len(errors) > 0
        assert any(isinstance(e, LaneGroupError) for e in errors)

    def test_consistency_pass(self):
        spec = _build_spec_from_python()
        lanes = [
            LaneAddress(MoveType.SITE, zone_id=0, word_id=0, site_id=0, bus_id=0),
            LaneAddress(MoveType.SITE, zone_id=0, word_id=0, site_id=1, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert errors == []

    def test_consistency_fail_direction(self):
        spec = _build_spec_from_python()
        lanes = [
            LaneAddress(MoveType.SITE, zone_id=0, word_id=0, site_id=0, bus_id=0),
            LaneAddress(
                MoveType.SITE,
                zone_id=0,
                word_id=0,
                site_id=1,
                bus_id=0,
                direction=Direction.BACKWARD,
            ),
        ]
        errors = spec.check_lanes(lanes)
        assert len(errors) > 0


class TestRepr:
    def test_arch_spec_repr(self):
        spec = _build_spec_from_python()
        assert "ArchSpec" in repr(spec)

    def test_word_repr(self):
        spec = _build_spec_from_python()
        word = spec.word_by_id(0)
        assert word is not None
        assert "Word" in repr(word)
