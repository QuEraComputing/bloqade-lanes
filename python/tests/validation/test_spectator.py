import pytest

from bloqade.lanes.arch import (
    ArchBlueprint,
    DeviceLayout,
    HypercubeSiteTopology,
    HypercubeWordTopology,
    MatchingTopology,
    ZoneSpec,
    build_arch,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.dialects.qmove import Effects, Frame, FrameShape
from bloqade.lanes.validation.spectator import SingleZonePolicy, ZonedPolicy

ARCH = get_arch_spec()  # one zone; words (0, 1) are an entangling pair
Z0 = ZoneAddress(0)
A = LocationAddress(0, 0)
A_PARTNER = LocationAddress(1, 0)


def _frame(*binding: LocationAddress, **effects) -> Frame:
    return Frame(FrameShape(((0, len(binding)),)), binding, Effects(**effects))


def test_partner_is_where_the_tests_assume():
    assert ARCH.get_cz_partner(A) == A_PARTNER


def test_zoned_policy_rejects_only_global_pulses():
    policy = ZonedPolicy()
    assert policy.check_frame(_frame(A, cz_zones=frozenset({Z0})), ARCH) == []
    assert policy.check_frame(_frame(A, global_pulses=True), ARCH) == [
        "ZonedPolicy: subroutines may not use global pulses"
    ]


def test_single_zone_policy_accepts_a_pair_closed_footprint():
    frame = _frame(A, A_PARTNER, cz_zones=frozenset({Z0}))
    assert SingleZonePolicy().check_frame(frame, ARCH) == []


def test_single_zone_policy_requires_pair_closure():
    (problem,) = SingleZonePolicy().check_frame(
        _frame(A, cz_zones=frozenset({Z0})), ARCH
    )
    assert "CZ partner (zone 0, word 1, site 0)" in problem


def test_pair_closure_only_matters_in_cz_zones():
    assert SingleZonePolicy().check_frame(_frame(A), ARCH) == []


def test_single_zone_policy_rejects_measurement_and_global_pulses():
    frame = _frame(A, A_PARTNER, measure_zones=frozenset({Z0}), global_pulses=True)
    assert SingleZonePolicy().check_frame(frame, ARCH) == [
        "SingleZonePolicy: subroutines may not use global pulses",
        "SingleZonePolicy: subroutines may not measure",
    ]


def test_check_call_is_declared_only():
    with pytest.raises(NotImplementedError):
        ZonedPolicy().check_call(_frame(A), None, ARCH)  # type: ignore[arg-type]


def _two_zone_arch():
    """A synthetic arch: an entangling "proc" zone and a pair-less "mem" zone."""
    blueprint = ArchBlueprint(
        zones={
            "proc": ZoneSpec(
                num_rows=2,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=HypercubeSiteTopology(),
            ),
            "mem": ZoneSpec(num_rows=2, num_cols=2),
        },
        layout=DeviceLayout(sites_per_word=4),
    )
    return build_arch(blueprint, connections={("proc", "mem"): MatchingTopology()}).arch


TWO_ZONE = _two_zone_arch()
PROC, MEM = ZoneAddress(0), ZoneAddress(1)


def test_two_zone_arch_is_shaped_as_assumed():
    assert len(TWO_ZONE.zones) == 2
    assert TWO_ZONE.get_cz_partner(LocationAddress(0, 0, 0)) == LocationAddress(1, 0, 0)
    assert TWO_ZONE.get_cz_partner(LocationAddress(0, 0, 1)) is None


def test_pair_closure_is_per_zone_on_a_synthetic_arch():
    proc_atom = LocationAddress(0, 0, 0)
    proc_partner = LocationAddress(1, 0, 0)
    mem_atom = LocationAddress(0, 0, 1)
    policy = SingleZonePolicy()
    closed = _frame(proc_atom, proc_partner, mem_atom, cz_zones=frozenset({PROC}))
    assert policy.check_frame(closed, TWO_ZONE) == []
    (problem,) = policy.check_frame(
        _frame(proc_atom, mem_atom, cz_zones=frozenset({PROC})), TWO_ZONE
    )
    assert "CZ partner (zone 0, word 1, site 0)" in problem
    # A pair-less zone needs no closure even when it is a CZ zone.
    assert (
        policy.check_frame(_frame(mem_atom, cz_zones=frozenset({MEM})), TWO_ZONE) == []
    )
