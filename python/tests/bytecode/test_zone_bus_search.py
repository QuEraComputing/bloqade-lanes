"""Regression tests for inter-zone routing in the Rust move router (#845).

The Rust ``SearchEngine`` / ``TargetSolver`` used to build its search graph
from intra-zone buses only (site + word buses) and omit inter-zone
``zone_buses``. As a result any cross-zone move was reported ``unsolvable``
even when a direct inter-zone bus existed. These tests pin the fix: the search
graph now includes ``zone_buses`` edges, matching Python's ``PathFinder``.
"""

from __future__ import annotations

from bloqade.lanes.arch.path import PathFinder
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import (
    ArchSpec as RustArchSpec,
    EntropyOptions,
    MoveSearch,
    SearchEngine,
)
from bloqade.lanes.bytecode.encoding import LocationAddress

# Minimal two-zone processor: zone 0 ("gate") holds word 0 and zone 1
# ("memory") holds word 1, each a single-site word. A one-to-one zone_bus
# connects (zone 1, word 1) -> (zone 0, word 0). No intra-zone buses exist, so
# the only path between the zones is the inter-zone bus.
_TWO_ZONE_ARCH_JSON = """
{
  "version": "2.0",
  "words": [{ "sites": [[0, 0]] }, { "sites": [[0, 0]] }],
  "zones": [
    {
      "name": "gate",
      "grid": { "x_start": 0.0, "y_start": 0.0, "x_spacing": [], "y_spacing": [] },
      "site_buses": [], "word_buses": [],
      "words_with_site_buses": [], "sites_with_word_buses": [],
      "entangling_pairs": []
    },
    {
      "name": "memory",
      "grid": { "x_start": 0.0, "y_start": 10.0, "x_spacing": [], "y_spacing": [] },
      "site_buses": [], "word_buses": [],
      "words_with_site_buses": [], "sites_with_word_buses": [],
      "entangling_pairs": []
    }
  ],
  "zone_buses": [
    { "src": [{ "zone_id": 1, "word_id": 1 }], "dst": [{ "zone_id": 0, "word_id": 0 }] }
  ],
  "modes": [{ "name": "default", "zones": [0, 1], "bitstring_order": [] }]
}
"""


def _arch() -> ArchSpec:
    return ArchSpec(RustArchSpec.from_json(_TWO_ZONE_ARCH_JSON))


def test_pathfinder_finds_inter_zone_path():
    arch = _arch()
    mem_loc = LocationAddress(1, 0, 1)
    gate_loc = LocationAddress(0, 0, 0)
    path = PathFinder(arch).find_path(mem_loc, gate_loc)
    assert path is not None


def test_rust_solver_routes_across_zone_bus():
    arch = _arch()
    mem_loc = LocationAddress(1, 0, 1)
    gate_loc = LocationAddress(0, 0, 0)

    engine = SearchEngine.from_arch_spec(arch._inner)
    solver = _native.TargetSolver(engine, MoveSearch.entropy())
    result = solver.solve({0: mem_loc._inner}, {0: gate_loc._inner}, [], None)

    assert result.status == "solved"


def test_rust_solver_routes_back_across_zone_bus():
    # The reverse (gate -> memory) direction must also be traversable.
    arch = _arch()
    mem_loc = LocationAddress(1, 0, 1)
    gate_loc = LocationAddress(0, 0, 0)

    engine = SearchEngine.from_arch_spec(arch._inner)
    solver = _native.TargetSolver(engine, MoveSearch.entropy())
    result = solver.solve({0: gate_loc._inner}, {0: mem_loc._inner}, [], None)

    assert result.status == "solved"


def test_solve_result_reports_proven_and_termination():
    """The root certificate reaches Python, and the knob turns it off.

    One atom across the zone bus: the plan costs exactly ``h(root)``, so the
    bound proves it optimal without any complete generator. With
    ``bound_terminates`` off the driver declines to act on that and reports no
    proof, while producing the same plan -- the certificate ends a spin over
    nodes the bound had already cut, so no expansion is skipped.
    """
    engine = SearchEngine.from_arch_spec(RustArchSpec.from_json(_TWO_ZONE_ARCH_JSON))
    mem = LocationAddress(1, 0, 1)
    gate = LocationAddress(0, 0, 0)

    def solve(bound_terminates: bool):
        search = MoveSearch.entropy().with_entropy_options(
            EntropyOptions(
                completion_bound="weighted_distance",
                bound_terminates=bound_terminates,
            )
        )
        return _native.TargetSolver(engine, search).solve(
            {0: mem._inner}, {0: gate._inner}, [], None
        )

    stopped, spun = solve(True), solve(False)

    assert stopped.status == "solved"
    assert stopped.proven is True
    assert stopped.termination == "exhausted_proof"
    assert "proven=true" in repr(stopped)

    assert spun.proven is False
    assert spun.cost == stopped.cost
    assert spun.nodes_expanded == stopped.nodes_expanded


def test_unbounded_solve_is_never_proven():
    """Without a bound there is nothing to certify, whatever the knob says."""
    engine = SearchEngine.from_arch_spec(RustArchSpec.from_json(_TWO_ZONE_ARCH_JSON))
    search = MoveSearch.entropy().with_entropy_options(
        EntropyOptions(bound_terminates=True)
    )
    result = _native.TargetSolver(engine, search).solve(
        {0: LocationAddress(1, 0, 1)._inner},
        {0: LocationAddress(0, 0, 0)._inner},
        [],
        None,
    )
    assert result.status == "solved"
    assert result.proven is False
