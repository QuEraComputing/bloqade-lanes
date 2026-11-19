"""Word buses with a ``site_map``: a word move that also changes the site index.

The site map is shared by every word pair of the bus. These tests pin that the
JSON format, the native bindings and the Python arch APIs (``iter_all_lanes``,
``get_endpoints`` / ``get_lane_address``, ``PathFinder`` and
``ArchSpecGeometry``) all resolve through the map instead of assuming the site
index is preserved.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bloqade.lanes.arch.geometry import ArchSpecGeometry
from bloqade.lanes.arch.path import PathFinder
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import ArchSpec as RustArchSpec, SiteBus, WordBus, _native
from bloqade.lanes.bytecode._native import MoveSearch, SearchEngine, SolveStatus
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LocationAddress,
    MoveType,
    WordLaneAddress,
)
from bloqade.lanes.bytecode.exceptions import (
    ArchSpecError,
    LaneGroupAODConstraintViolationError,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_SPEC = REPO_ROOT / "examples" / "arch" / "full.json"
SCHEMA_PATH = REPO_ROOT / "docs" / "src" / "arch" / "archspec-schema.json"

# full.json zone 0: word bus 0 -> 1 over landing pads [5, 6, 7, 8, 9].
SITE_MAP = {"src": [5, 6], "dst": [7, 8]}


def _spec_dict(site_map: dict | None = SITE_MAP) -> dict:
    spec = json.loads(FULL_SPEC.read_text(encoding="utf-8"))
    bus = spec["zones"][0]["word_buses"][0]
    if site_map is not None:
        bus["site_map"] = site_map
    return spec


def _arch(site_map: dict | None = SITE_MAP) -> ArchSpec:
    return ArchSpec(RustArchSpec.from_json_validated(json.dumps(_spec_dict(site_map))))


def test_site_map_matches_schema():
    validator = Draft202012Validator(
        json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    )
    assert not list(validator.iter_errors(_spec_dict()))


def test_native_word_bus_exposes_site_map():
    bus = _arch().zones[0].word_buses[0]
    assert bus.site_map == SiteBus(src=[5, 6], dst=[7, 8])
    assert bus.resolve_site_forward(5) == 7
    assert bus.resolve_site_backward(8) == 6
    assert bus.resolve_site_forward(9) is None

    plain = WordBus(src=[0], dst=[1])
    assert plain.site_map is None
    assert plain.resolve_site_forward(9) == 9


def test_zone_word_bus_site_pairs():
    assert _arch().zones[0].word_bus_site_pairs(0) == [(5, 7), (6, 8)]
    assert _arch(None).zones[0].word_bus_site_pairs(0) == [(s, s) for s in range(5, 10)]
    assert _arch().zones[0].word_bus_site_pairs(3) == []


def test_site_map_round_trips_through_json():
    arch = _arch()
    reloaded = json.loads(arch.to_json())
    assert reloaded["zones"][0]["word_buses"][0]["site_map"] == SITE_MAP
    # No site map -> the key is omitted.
    assert (
        "site_map" not in json.loads(_arch(None).to_json())["zones"][0]["word_buses"][0]
    )


def test_cyclic_site_map_is_rejected():
    with pytest.raises(ArchSpecError):
        RustArchSpec.from_json_validated(
            json.dumps(_spec_dict({"src": [5, 6], "dst": [6, 5]}))
        )


def test_site_map_outside_landing_pads_is_rejected():
    with pytest.raises(ArchSpecError):
        RustArchSpec.from_json_validated(
            json.dumps(_spec_dict({"src": [0], "dst": [5]}))
        )


def test_iter_all_lanes_uses_site_map_sources():
    word_lanes = {
        (lane.word_id, lane.site_id, lane.direction)
        for lane in _arch().iter_all_lanes()
        if lane.move_type == MoveType.WORD
    }
    assert word_lanes == {
        (0, s, d) for s in (5, 6) for d in (Direction.FORWARD, Direction.BACKWARD)
    }


def test_endpoints_follow_site_map():
    arch = _arch()
    lane = WordLaneAddress(0, 5, 0, Direction.FORWARD, 0)
    src, dst = LocationAddress(0, 5, 0), LocationAddress(1, 7, 0)
    assert arch.get_endpoints(lane) == (src, dst)
    assert arch.get_endpoints(lane.reverse()) == (dst, src)
    assert arch.get_lane_address(src, dst) == lane
    assert arch.get_lane_address(dst, src) == lane.reverse()
    # The site-preserving move is not on this bus.
    assert arch.get_lane_address(src, LocationAddress(1, 5, 0)) is None


def test_path_finder_follows_site_map():
    arch = _arch()
    finder = PathFinder(arch)
    src, dst = LocationAddress(0, 6, 0), LocationAddress(1, 8, 0)
    lane = finder.get_lane(src, dst)
    assert lane is not None
    assert lane == WordLaneAddress(0, 6, 0, Direction.FORWARD, 0)
    assert finder.get_endpoints(lane) == (src, dst)
    assert finder.get_endpoints(lane.reverse()) == (dst, src)
    # Every word-bus edge in the graph agrees with the Rust resolver.
    for word_lane, endpoints in finder.end_points_cache.items():
        if word_lane.move_type == MoveType.WORD:
            assert arch.get_endpoints(word_lane) == endpoints


def test_geometry_counts_and_grids_follow_site_map():
    arch = _arch()
    geometry = ArchSpecGeometry(arch)
    word = [b for b in geometry.get_available_buses(0) if b.move_type == MoveType.WORD]
    assert {b.num_lanes for b in word} == {2}

    src_grid, dst_grid = geometry.get_grid_endpoints(
        0, 0, MoveType.WORD, Direction.FORWARD
    )
    src_positions = {
        arch.get_position(LocationAddress(0, s, 0)) for s in SITE_MAP["src"]
    }
    dst_positions = {
        arch.get_position(LocationAddress(1, s, 0)) for s in SITE_MAP["dst"]
    }
    assert set(src_grid.positions) == src_positions
    assert set(dst_grid.positions) == dst_positions


@pytest.mark.parametrize(
    "start, goal",
    [
        (LocationAddress(0, 5, 0), LocationAddress(1, 7, 0)),
        (LocationAddress(1, 8, 0), LocationAddress(0, 6, 0)),
    ],
    ids=["forward", "backward"],
)
def test_rust_solver_routes_through_site_map(start, goal):
    arch = _arch()
    engine = SearchEngine.from_arch_spec(arch._inner)
    solver = _native.TargetSolver(engine, MoveSearch.entropy())
    result = solver.solve({0: start._inner}, {0: goal._inner}, [], None)
    assert result.status == SolveStatus.SOLVED
    assert PathFinder(arch).find_path(start, goal) is not None


# Two 2x2 words side by side; word bus 0 -> 1 over all four sites. Site order:
# 0 = (left, bottom), 1 = (right, bottom), 2 = (left, top), 3 = (right, top).
_TWO_BY_TWO_JSON = """
{
  "version": "2.0",
  "words": [
    { "sites": [[0, 0], [1, 0], [0, 1], [1, 1]] },
    { "sites": [[2, 0], [3, 0], [2, 1], [3, 1]] }
  ],
  "zones": [{
    "grid": { "x_start": 0.0, "y_start": 0.0,
              "x_spacing": [5.0, 5.0, 5.0], "y_spacing": [3.0] },
    "site_buses": [],
    "word_buses": [{ "src": [0], "dst": [1], "site_map": SITE_MAP }],
    "words_with_site_buses": [],
    "sites_with_word_buses": [0, 1, 2, 3]
  }],
  "zone_buses": [],
  "modes": [{ "name": "all", "zones": [0], "bitstring_order": [] }]
}
"""


def _two_by_two(site_map: dict) -> ArchSpec:
    raw = _TWO_BY_TWO_JSON.replace("SITE_MAP", json.dumps(site_map))
    return ArchSpec(RustArchSpec.from_json_validated(raw))


@pytest.mark.parametrize("direction", [Direction.FORWARD, Direction.BACKWARD])
def test_lane_group_landing_off_grid_is_an_aod_violation(direction):
    # The bottom row lands on (15, 0) and (10, 3): a diagonal, not a grid.
    arch = _two_by_two({"src": [0, 1], "dst": [1, 2]})
    lanes = [WordLaneAddress(0, s, 0, direction, 0) for s in (0, 1)]
    errors = arch.check_lane_group(lanes)
    assert [type(e) for e in errors] == [LaneGroupAODConstraintViolationError]
    assert "lane destinations" in str(errors[0])


@pytest.mark.parametrize("direction", [Direction.FORWARD, Direction.BACKWARD])
def test_lane_group_landing_on_grid_is_valid(direction):
    # The bottom row lands on the top row of word 1.
    arch = _two_by_two({"src": [0, 1], "dst": [2, 3]})
    lanes = [WordLaneAddress(0, s, 0, direction, 0) for s in (0, 1)]
    assert list(arch.check_lane_group(lanes)) == []
