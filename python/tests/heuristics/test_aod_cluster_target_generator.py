from __future__ import annotations

import pytest

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.arch.path import PathFinder
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.target_generator import (
    AODClusterTargetGenerator,
    TargetContext,
    _first_hop_sig,
)


@pytest.fixture(scope="module")
def arch() -> ArchSpec:
    return get_arch_spec()


def _pick_cz_pair(
    arch: ArchSpec,
) -> tuple[LocationAddress, LocationAddress]:
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is not None and p != s:
            return s, p
    raise AssertionError("fixture prerequisite: arch has no CZ-partnered home site")


def _ctx(
    arch: ArchSpec,
    layout_tup: tuple[LocationAddress, ...],
    controls: tuple[int, ...],
    targets: tuple[int, ...],
) -> TargetContext:
    state = ConcreteState(
        occupied=frozenset(),
        layout=layout_tup,
        move_count=(0,) * len(layout_tup),
    )
    return TargetContext(
        arch_spec=arch,
        state=state,
        controls=controls,
        targets=targets,
        lookahead_cz_layers=(),
        cz_stage_index=0,
    )


def test_first_hop_sig_none_for_missing_path():
    assert _first_hop_sig(None) is None


def test_first_hop_sig_none_for_empty_path(arch):
    pf = PathFinder(arch)
    loc, _ = _pick_cz_pair(arch)
    path = pf.find_path(loc, loc)
    assert path is not None
    assert _first_hop_sig(path) is None


def test_first_hop_sig_returns_tuple(arch):
    pf = PathFinder(arch)
    src, dst = _pick_cz_pair(arch)
    path = pf.find_path(src, dst)
    assert path is not None and path[0], "fixture: expected non-empty path"
    sig = _first_hop_sig(path)
    assert sig is not None
    mt, zid, bid, direction = sig
    lane0 = path[0][0]
    assert mt == lane0.move_type
    assert zid == lane0.zone_id
    assert bid == lane0.bus_id
    assert direction == lane0.direction


def test_generate_empty_stage_returns_current_placement(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(), targets=())
    out = AODClusterTargetGenerator().generate(ctx)
    assert out == [{0: loc0, 1: loc1}]


def test_generate_already_partnered_pair_is_noop(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(0,), targets=(1,))
    out = AODClusterTargetGenerator().generate(ctx)
    assert out == [{0: loc0, 1: loc1}]


def test_generate_is_deterministic_across_calls(arch):
    loc0, loc1 = _pick_cz_pair(arch)
    ctx = _ctx(arch, (loc0, loc1), controls=(0,), targets=(1,))
    gen = AODClusterTargetGenerator()
    assert gen.generate(ctx) == gen.generate(ctx)


def test_generate_single_pair_plan_is_cz_partnered(gate_arch):
    """For a single non-partnered pair the generator must still produce a
    plan whose qids form a valid CZ partnership."""
    arch = gate_arch(num_cols=8)
    # Control at home word 0, target at home word 2 -- feasible and not
    # already CZ-partnered, so the generator must move one onto the
    # other's blockade partner.
    loc_ctrl = LocationAddress(0, 0)
    loc_tgt = LocationAddress(2, 0)
    ctx = _ctx(arch, (loc_ctrl, loc_tgt), controls=(0,), targets=(1,))
    out = AODClusterTargetGenerator().generate(ctx)
    assert out, "generator returned empty on feasible pair"
    plan = out[0]
    c_loc = plan[0]
    t_loc = plan[1]
    assert (
        arch.get_cz_partner(c_loc) == t_loc or arch.get_cz_partner(t_loc) == c_loc
    ), f"plan is not CZ-partnered: control={c_loc}, target={t_loc}"
