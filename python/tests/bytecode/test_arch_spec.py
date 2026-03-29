import json

import pytest

from bloqade.lanes.bytecode import (
    ArchSpec,
    ArchSpecError,
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
)
from bloqade.lanes.bytecode.exceptions import (
    ArchSpecGeometryError,
    LaneGroupError,
    LocationGroupError,
)

EXAMPLE_JSON = json.dumps(
    {
        "version": 1,
        "geometry": {
            "sites_per_word": 10,
            "words": [
                {
                    "grid": {
                        "x_start": 1.0,
                        "y_start": 2.5,
                        "x_spacing": [2.0, 2.0, 2.0, 2.0],
                        "y_spacing": [2.5],
                    },
                    "sites": [
                        [0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [4, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [3, 1],
                        [4, 1],
                    ],
                    "cz_pairs": [
                        [0, 5],
                        [0, 6],
                        [0, 7],
                        [0, 8],
                        [0, 9],
                        [0, 0],
                        [0, 1],
                        [0, 2],
                        [0, 3],
                        [0, 4],
                    ],
                },
                {
                    "grid": {
                        "x_start": 1.0,
                        "y_start": 12.5,
                        "x_spacing": [2.0, 2.0, 2.0, 2.0],
                        "y_spacing": [2.5],
                    },
                    "sites": [
                        [0, 0],
                        [1, 0],
                        [2, 0],
                        [3, 0],
                        [4, 0],
                        [0, 1],
                        [1, 1],
                        [2, 1],
                        [3, 1],
                        [4, 1],
                    ],
                    "cz_pairs": [
                        [1, 5],
                        [1, 6],
                        [1, 7],
                        [1, 8],
                        [1, 9],
                        [1, 0],
                        [1, 1],
                        [1, 2],
                        [1, 3],
                        [1, 4],
                    ],
                },
            ],
        },
        "buses": {
            "site_buses": [{"src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9]}],
            "word_buses": [{"src": [0], "dst": [1]}],
        },
        "words_with_site_buses": [0, 1],
        "sites_with_word_buses": [5, 6, 7, 8, 9],
        "zones": [{"words": [0, 1]}],
        "entangling_zones": [0],
        "measurement_mode_zones": [0],
        "paths": [
            {
                "lane": "0xC000000000000005",
                "waypoints": [[1.0, 15.0], [1.0, 10.0], [1.0, 5.0]],
            }
        ],
    }
)


def _make_word(word_id, y_start):
    grid = Grid(
        x_start=1.0,
        y_start=y_start,
        x_spacing=[2.0, 2.0, 2.0, 2.0],
        y_spacing=[2.5],
    )
    sites = [
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
    cz_pairs = [
        (word_id, 5),
        (word_id, 6),
        (word_id, 7),
        (word_id, 8),
        (word_id, 9),
        (word_id, 0),
        (word_id, 1),
        (word_id, 2),
        (word_id, 3),
        (word_id, 4),
    ]
    return Word(grid=grid, sites=sites, cz_pairs=cz_pairs)


def _build_spec_from_python():
    word0 = _make_word(0, 2.5)
    word1 = _make_word(1, 12.5)
    geometry = Geometry(sites_per_word=10, words=[word0, word1])

    site_bus = Bus(src=[0, 1, 2, 3, 4], dst=[5, 6, 7, 8, 9])
    word_bus = Bus(src=[0], dst=[1])
    buses = Buses(site_buses=[site_bus], word_buses=[word_bus])

    zone = Zone(words=[0, 1])

    return ArchSpec(
        version=(1, 0),
        geometry=geometry,
        buses=buses,
        words_with_site_buses=[0, 1],
        sites_with_word_buses=[5, 6, 7, 8, 9],
        zones=[zone],
        entangling_zones=[0],
        measurement_mode_zones=[0],
        paths=[
            TransportPath(
                lane=LaneAddress(
                    MoveType.WORD,
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
        assert spec.version == (1, 0)

    def test_geometry(self):
        spec = _build_spec_from_python()
        assert spec.geometry.sites_per_word == 10
        assert len(spec.geometry.words) == 2

    def test_word_without_cz_pairs(self):
        grid = Grid(x_start=1.0, y_start=2.0, x_spacing=[], y_spacing=[])
        word = Word(grid=grid, sites=[(0, 0)])
        assert word.cz_pairs is None


class TestCapabilityFlags:
    def test_defaults_to_false(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        assert spec.feed_forward is False
        assert spec.atom_reloading is False

    def test_explicit_true(self):
        data = json.loads(EXAMPLE_JSON)
        data["feed_forward"] = True
        data["atom_reloading"] = True
        spec = ArchSpec.from_json(json.dumps(data))
        assert spec.feed_forward is True
        assert spec.atom_reloading is True

    def test_construct_from_python_defaults(self):
        spec = _build_spec_from_python()
        assert spec.feed_forward is False
        assert spec.atom_reloading is False

    def test_construct_from_python_explicit(self):
        word0 = _make_word(0, 2.5)
        word1 = _make_word(1, 12.5)
        geometry = Geometry(sites_per_word=10, words=[word0, word1])
        site_bus = Bus(src=[0, 1, 2, 3, 4], dst=[5, 6, 7, 8, 9])
        word_bus = Bus(src=[0], dst=[1])
        buses = Buses(site_buses=[site_bus], word_buses=[word_bus])
        zone = Zone(words=[0, 1])
        spec = ArchSpec(
            version=(1, 0),
            geometry=geometry,
            buses=buses,
            words_with_site_buses=[0, 1],
            sites_with_word_buses=[5, 6, 7, 8, 9],
            zones=[zone],
            entangling_zones=[0],
            measurement_mode_zones=[0],
            feed_forward=True,
            atom_reloading=True,
        )
        assert spec.feed_forward is True
        assert spec.atom_reloading is True


class TestLoadFromJson:
    def test_from_json(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        assert spec.version == (1, 0)

    def test_from_json_validated(self):
        spec = ArchSpec.from_json_validated(EXAMPLE_JSON)
        assert spec.version == (1, 0)

    def test_from_json_invalid(self):
        with pytest.raises(ValueError):
            ArchSpec.from_json('{"version": 1}')

    def test_from_json_validated_bad_schema(self):
        bad = json.dumps(
            {
                "version": 1,
                "geometry": {
                    "sites_per_word": 2,
                    "words": [
                        {
                            "grid": {
                                "x_start": 1.0,
                                "y_start": 2.0,
                                "x_spacing": [],
                                "y_spacing": [],
                            },
                            "sites": [[0, 0]],  # wrong count
                        }
                    ],
                },
                "buses": {"site_buses": [], "word_buses": []},
                "words_with_site_buses": [],
                "sites_with_word_buses": [],
                "zones": [{"words": [0]}],
                "entangling_zones": [],
                "measurement_mode_zones": [0],
            }
        )
        with pytest.raises(ArchSpecError) as exc_info:
            ArchSpec.from_json_validated(bad)
        assert len(exc_info.value.errors) > 0
        assert any(isinstance(e, ArchSpecGeometryError) for e in exc_info.value.errors)


class TestValidation:
    def test_validate_valid(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        spec.validate()

    def test_validate_invalid_raises(self):
        bad = json.dumps(
            {
                "version": 1,
                "geometry": {
                    "sites_per_word": 2,
                    "words": [
                        {
                            "grid": {
                                "x_start": 1.0,
                                "y_start": 2.0,
                                "x_spacing": [],
                                "y_spacing": [],
                            },
                            "sites": [[0, 0]],  # wrong count
                        }
                    ],
                },
                "buses": {"site_buses": [], "word_buses": []},
                "words_with_site_buses": [],
                "sites_with_word_buses": [],
                "zones": [{"words": [0]}],
                "entangling_zones": [],
                "measurement_mode_zones": [0],
            }
        )
        spec = ArchSpec.from_json(bad)
        with pytest.raises(ArchSpecError) as exc_info:
            spec.validate()
        assert len(exc_info.value.errors) > 0


class TestPropertyAccess:
    def test_nested_properties(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)

        geom = spec.geometry
        assert geom.sites_per_word == 10

        word = geom.words[0]
        assert len(word.sites) == 10
        assert word.sites[0] == (0, 0)

        grid = word.grid
        assert grid.x_positions == [1.0, 3.0, 5.0, 7.0, 9.0]
        assert grid.y_positions == [2.5, 5.0]

        assert word.cz_pairs is not None
        assert word.cz_pairs[0] == (0, 5)

    def test_buses(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        buses = spec.buses
        assert len(buses.site_buses) == 1
        assert len(buses.word_buses) == 1

        sb = buses.site_buses[0]
        assert sb.src == [0, 1, 2, 3, 4]
        assert sb.dst == [5, 6, 7, 8, 9]

    def test_zones(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        assert len(spec.zones) == 1
        assert spec.zones[0].words == [0, 1]

    def test_top_level_lists(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        assert spec.words_with_site_buses == [0, 1]
        assert spec.sites_with_word_buses == [5, 6, 7, 8, 9]
        assert spec.entangling_zones == [0]
        assert spec.measurement_mode_zones == [0]

    def test_paths(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        assert spec.paths is not None
        assert len(spec.paths) == 1
        assert spec.paths[0].lane_encoded == 0xC000000000000005
        lane = spec.paths[0].lane
        assert lane.direction == Direction.BACKWARD
        assert lane.move_type == MoveType.WORD
        assert lane.word_id == 0
        assert lane.site_id == 5
        assert lane.bus_id == 0
        assert len(spec.paths[0].waypoints) == 3


class TestQueryMethods:
    def test_word_by_id(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        word = spec.word_by_id(0)
        assert word is not None
        assert len(word.sites) == 10
        assert spec.word_by_id(99) is None

    def test_zone_by_id(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        zone = spec.zone_by_id(0)
        assert zone is not None
        assert zone.words == [0, 1]
        assert spec.zone_by_id(99) is None

    def test_site_bus_by_id(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        bus = spec.site_bus_by_id(0)
        assert bus is not None
        assert bus.src == [0, 1, 2, 3, 4]
        assert spec.site_bus_by_id(99) is None

    def test_word_bus_by_id(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        bus = spec.word_bus_by_id(0)
        assert bus is not None
        assert bus.src == [0]
        assert spec.word_bus_by_id(99) is None


class TestBusResolution:
    def test_site_bus_resolve_forward(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        bus = spec.site_bus_by_id(0)
        assert bus is not None
        assert bus.resolve_forward(0) == 5
        assert bus.resolve_forward(4) == 9
        assert bus.resolve_forward(99) is None

    def test_site_bus_resolve_backward(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        bus = spec.site_bus_by_id(0)
        assert bus is not None
        assert bus.resolve_backward(5) == 0
        assert bus.resolve_backward(9) == 4
        assert bus.resolve_backward(99) is None

    def test_word_bus_resolve_forward(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        bus = spec.word_bus_by_id(0)
        assert bus is not None
        assert bus.resolve_forward(0) == 1
        assert bus.resolve_forward(99) is None

    def test_word_bus_resolve_backward(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        bus = spec.word_bus_by_id(0)
        assert bus is not None
        assert bus.resolve_backward(1) == 0
        assert bus.resolve_backward(99) is None


class TestSitePosition:
    def test_valid_positions(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        word = spec.word_by_id(0)
        assert word is not None
        assert word.site_position(0) == (1.0, 2.5)
        assert word.site_position(5) == (1.0, 5.0)
        assert word.site_position(4) == (9.0, 2.5)

    def test_out_of_range(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        word = spec.word_by_id(0)
        assert word is not None
        assert word.site_position(99) is None


class TestRepr:
    def test_arch_spec_repr(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        assert "ArchSpec" in repr(spec)

    def test_word_repr(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        word = spec.word_by_id(0)
        assert word is not None
        assert "Word" in repr(word)


class TestLocationPosition:
    def test_valid(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        loc = LocationAddress(word_id=0, site_id=0)
        assert spec.location_position(loc) == (1.0, 2.5)

    def test_different_site(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        loc = LocationAddress(word_id=0, site_id=5)
        assert spec.location_position(loc) == (1.0, 5.0)

    def test_invalid_word(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        loc = LocationAddress(word_id=99, site_id=0)
        assert spec.location_position(loc) is None

    def test_invalid_site(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        loc = LocationAddress(word_id=0, site_id=99)
        assert spec.location_position(loc) is None


class TestCheckLocations:
    def test_valid(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        locs = [LocationAddress(word_id=0, site_id=0)]
        errors = spec.check_locations(locs)
        assert errors == []

    def test_invalid(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        locs = [LocationAddress(word_id=99, site_id=0)]
        errors = spec.check_locations(locs)
        assert len(errors) > 0
        assert any(isinstance(e, LocationGroupError) for e in errors)

    def test_duplicate(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        locs = [
            LocationAddress(word_id=0, site_id=0),
            LocationAddress(word_id=0, site_id=1),
            LocationAddress(word_id=0, site_id=0),
        ]
        errors = spec.check_locations(locs)
        assert len(errors) > 0


class TestCheckLanes:
    def test_valid(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert errors == []

    def test_invalid_bus(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=99),
        ]
        errors = spec.check_lanes(lanes)
        assert len(errors) > 0
        assert any(isinstance(e, LaneGroupError) for e in errors)

    def test_consistency_pass(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=0, site_id=1, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert errors == []

    def test_consistency_fail_direction(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
            LaneAddress(
                MoveType.SITE,
                word_id=0,
                site_id=1,
                bus_id=0,
                direction=Direction.BACKWARD,
            ),
        ]
        errors = spec.check_lanes(lanes)
        assert len(errors) > 0

    def test_aod_constraint_rectangle_pass(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        # 2x2 grid using valid forward sources on 2 words:
        # Word 0, Site 0: (1.0, 2.5)   Word 0, Site 1: (3.0, 2.5)
        # Word 1, Site 0: (1.0, 12.5)  Word 1, Site 1: (3.0, 12.5)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=0, site_id=1, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=1, site_id=0, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=1, site_id=1, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert errors == []

    def test_aod_constraint_not_rectangle(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        # 3 corners of a grid (missing word 1, site 1)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=0, site_id=1, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=1, site_id=0, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert len(errors) > 0

    def test_duplicate(self):
        spec = ArchSpec.from_json(EXAMPLE_JSON)
        lanes = [
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=0, site_id=1, bus_id=0),
            LaneAddress(MoveType.SITE, word_id=0, site_id=0, bus_id=0),
        ]
        errors = spec.check_lanes(lanes)
        assert len(errors) > 0
