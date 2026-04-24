from __future__ import annotations

import pytest

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.heuristics.physical.target_generator import (
    AODClusterTargetGenerator,
    TargetContext,
    _first_hop_sig,
)


@pytest.fixture(scope="module")
def arch() -> layout.ArchSpec:
    return get_arch_spec()


def _pick_cz_pair(
    arch: layout.ArchSpec,
) -> tuple[layout.LocationAddress, layout.LocationAddress]:
    for s in arch.home_sites:
        p = arch.get_cz_partner(s)
        if p is not None and p != s:
            return s, p
    raise AssertionError("fixture prerequisite: arch has no CZ-partnered home site")


def _ctx(
    arch: layout.ArchSpec,
    layout_tup: tuple[layout.LocationAddress, ...],
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
    pf = layout.PathFinder(arch)
    loc, _ = _pick_cz_pair(arch)
    path = pf.find_path(loc, loc)
    assert path is not None
    assert _first_hop_sig(path) is None


def test_first_hop_sig_returns_tuple(arch):
    pf = layout.PathFinder(arch)
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


def test_generate_single_pair_plan_is_cz_partnered(arch):
    """For a single non-partnered pair the generator must still produce a
    plan whose qids form a valid CZ partnership."""
    # Find a blocker-free non-partnered pair by scanning home_sites.
    pf = layout.PathFinder(arch)
    chosen: (
        tuple[layout.LocationAddress, layout.LocationAddress, layout.LocationAddress]
        | None
    ) = None
    for a in arch.home_sites:
        pa = arch.get_cz_partner(a)
        if pa is None or pa == a:
            continue
        for b in arch.home_sites:
            if b in (a, pa):
                continue
            pb = arch.get_cz_partner(b)
            if pb is None or pb in (a, pa):
                continue
            if pf.find_path(b, pa) and pf.find_path(a, pb):
                chosen = (a, b, pa)
                break
        if chosen is not None:
            break
    if chosen is None:
        pytest.skip("physical arch lacks a non-partnered feasible CZ pair fixture")

    loc_tgt, loc_ctrl, _ = chosen
    ctx = _ctx(arch, (loc_ctrl, loc_tgt), controls=(0,), targets=(1,))
    out = AODClusterTargetGenerator().generate(ctx)
    assert out, "generator returned empty on feasible pair"
    plan = out[0]
    c_loc = plan[0]
    t_loc = plan[1]
    assert (
        arch.get_cz_partner(c_loc) == t_loc or arch.get_cz_partner(t_loc) == c_loc
    ), f"plan is not CZ-partnered: control={c_loc}, target={t_loc}"
