"""Guards on the conveyor-capable benchmark archspec (issue #939).

The spec is generated from the bundled Gemini physical one by
``scripts/gen_conveyor_arch.py``. Its value as a benchmark rests on two
properties that are easy to break silently, so both are pinned here:

* every site bus **overlaps** (a destination is also a source), which is what
  makes chain assembly reachable at all; and
* every conveyor bus is a **strict lane superset** of the hypercube bus it
  replaces, and the geometry is untouched, so a conveyor row differs from its
  builtin row only by bus structure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec

REPO_ROOT = Path(__file__).resolve().parents[3]
CONVEYOR_JSON = REPO_ROOT / "examples/arch/gemini-conveyor.json"
BUILTIN_JSON = (
    REPO_ROOT / "python/bloqade/lanes/arch/gemini/physical/_physical_spec.json"
)


@pytest.fixture(scope="module")
def conveyor() -> dict:
    return json.loads(CONVEYOR_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def builtin() -> dict:
    return json.loads(BUILTIN_JSON.read_text(encoding="utf-8"))


def _lanes(bus: dict) -> set[tuple[int, int]]:
    return set(zip(bus["src"], bus["dst"], strict=True))


def test_conveyor_spec_passes_rust_validation():
    """``--arch-spec`` loads via ``from_json``, which skips validation."""
    _RustArchSpec.from_json_validated(CONVEYOR_JSON.read_text(encoding="utf-8"))


def test_every_site_bus_overlaps_so_chains_are_reachable(conveyor):
    """A disjoint bus makes ``vacating_lane`` return ``None`` unconditionally.

    If this regresses, every chain path becomes dead code again and the suite
    silently goes back to measuring nothing.
    """
    site_buses = conveyor["zones"][0]["site_buses"]
    assert site_buses, "the conveyor spec must have site buses"

    overlapping = [
        dimension
        for dimension, bus in enumerate(site_buses)
        if set(bus["src"]) & set(bus["dst"])
    ]
    # Stride 4 on an 8-site word maps 0-3 onto 4-7, which is disjoint however it
    # is written, so the top dimension is a fixed point of the conversion.
    assert overlapping == [0, 1], (
        "expected dimensions 0 and 1 to overlap and the top dimension to be a "
        f"fixed point; overlapping dimensions were {overlapping}"
    )


def test_conveyor_buses_are_supersets_of_the_hypercube_buses(conveyor, builtin):
    """The superset property is what makes plan existence monotone.

    It does *not* guarantee better move counts — every strategy is a bounded
    heuristic search, so a larger space can still yield worse plans. What it does
    guarantee is that no plan available on the builtin spec is taken away, so a
    conveyor regression can never be blamed on lost connectivity.
    """
    conveyor_buses = conveyor["zones"][0]["site_buses"]
    builtin_buses = builtin["zones"][0]["site_buses"]
    assert len(conveyor_buses) == len(builtin_buses)

    for dimension, (new, old) in enumerate(zip(conveyor_buses, builtin_buses)):
        missing = _lanes(old) - _lanes(new)
        assert not missing, (
            f"site bus {dimension} dropped hypercube lanes {sorted(missing)}; "
            "the conveyor spec must be able to do everything builtin can"
        )


def test_geometry_and_non_site_buses_match_builtin_exactly(conveyor, builtin):
    """Only the site buses and their paths may differ from the bundled spec."""
    assert conveyor["words"] == builtin["words"]
    assert conveyor["modes"] == builtin["modes"]
    assert conveyor["zone_buses"] == builtin["zone_buses"]

    new_zone, old_zone = conveyor["zones"][0], builtin["zones"][0]
    for key in (
        "grid",
        "word_buses",
        "words_with_site_buses",
        "sites_with_word_buses",
        "entangling_pairs",
    ):
        assert new_zone[key] == old_zone[key], f"zone key {key!r} must be untouched"


def test_existing_transport_paths_are_preserved_byte_identically(conveyor, builtin):
    """Path entries are keyed by source site id, so conversion only *adds*.

    Rewriting an existing entry would mean the synthesized lanes had displaced
    hardware-derived ones (#753) rather than supplementing them.
    """
    old_paths = {path["lane"]: path["waypoints"] for path in builtin["paths"]}
    new_paths = {path["lane"]: path["waypoints"] for path in conveyor["paths"]}

    assert set(old_paths) <= set(new_paths), "conversion must not drop any path"
    changed = [lane for lane, wp in old_paths.items() if new_paths[lane] != wp]
    assert not changed, f"existing paths were rewritten: {changed[:5]}"


def test_every_site_bus_lane_has_a_transport_path(conveyor):
    """A lane without a path has no derivable duration for move metrics."""
    zone = conveyor["zones"][0]
    known = {int(path["lane"], 16) for path in conveyor["paths"]}

    missing: list[tuple[int, int, int]] = []
    for word_id in zone["words_with_site_buses"]:
        for bus_id, bus in enumerate(zone["site_buses"]):
            for src_site in bus["src"]:
                for direction in (0, 1):
                    data0 = ((word_id & 0xFFFF) << 16) | (src_site & 0xFFFF)
                    data1 = (direction & 1) << 31 | (0 << 29) | (0 << 21) | bus_id
                    if (data0 | (data1 << 32)) not in known:
                        missing.append((word_id, bus_id, src_site))

    assert not missing, f"site-bus lanes without a transport path: {missing[:5]}"
