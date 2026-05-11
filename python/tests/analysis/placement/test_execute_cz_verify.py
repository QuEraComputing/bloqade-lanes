"""Tests for ExecuteCZ.verify() — blockade-radius validation of CZ placements."""

from bloqade.lanes.analysis.placement.lattice import ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.bytecode.word import Word


def _make_execute_cz(
    layout_tuple: tuple[LocationAddress, ...],
    active_cz_zones: frozenset[ZoneAddress] | None = None,
) -> ExecuteCZ:
    n = len(layout_tuple)
    return ExecuteCZ(
        occupied=frozenset(),
        layout=layout_tuple,
        move_count=(0,) * n,
        active_cz_zones=active_cz_zones or frozenset([ZoneAddress(0)]),
        move_layers=(),
    )


def test_verify_one_pair():
    """Qubits at (0,0) and (1,0) are CZ pairs on logical arch (word 0 <-> word 1)"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz((LocationAddress(0, 0), LocationAddress(1, 0)))
    assert state.verify(arch_spec, (0,), (1,))
    assert state.verify(arch_spec, (1,), (0,))


def test_verify_one_pair_unblockaded():
    """Qubits at (0,0) and (0,1) are in the same word, not a CZ pair"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz((LocationAddress(0, 0), LocationAddress(0, 1)))
    assert state.verify(arch_spec, (0,), (1,)) is False
    assert state.verify(arch_spec, (1,), (0,)) is False


def test_verify_length_mismatch():
    """Mismatched control/target lengths should fail"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz((LocationAddress(0, 0), LocationAddress(1, 0)))
    assert state.verify(arch_spec, (0, 0), (1,)) is False
    assert state.verify(arch_spec, (1,), (0, 0)) is False


def test_verify_invalid_indices():
    """Test archspec w/invalid indices"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz((LocationAddress(0, 0), LocationAddress(1, 0)))
    assert state.verify(arch_spec, (-1,), (1,)) is False
    assert state.verify(arch_spec, (2,), (0,)) is False
    assert state.verify(arch_spec, (-1,), (0,)) is False
    assert state.verify(arch_spec, (0,), (2,)) is False


def test_verify_multi_word():
    """Test archspec w/multiple words (word 0 <-> word 1 pairing)"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz(
        (
            LocationAddress(0, 0),
            LocationAddress(1, 0),
            LocationAddress(1, 5),
            LocationAddress(0, 5),
        )
    )
    # qubit 0 at (0,0), qubit 1 at (1,0) -> CZ pair (word 0 <-> word 1, site 0)
    # qubit 2 at (1,5), qubit 3 at (0,5) -> CZ pair (word 1 <-> word 0, site 5)
    assert state.verify(arch_spec, (0,), (1,))
    assert state.verify(arch_spec, (2,), (3,))
    assert state.verify(arch_spec, (0, 2), (1, 3))
    assert state.verify(arch_spec, (0, 1), (2, 3)) is False
    assert state.verify(arch_spec, (1, 4), (2, 0)) is False


def test_verify_no_czs():
    """Test archspec w/no cz pairs"""
    from bloqade.lanes.bytecode._native import (
        Grid as RustGrid,
        LocationAddress as RustLocAddr,
        Mode as RustMode,
        Zone as RustZone,
    )

    word = Word(sites=((0, 0), (1, 0), (2, 0)))
    rust_grid = RustGrid.from_positions([0.0, 1.0, 2.0], [0.0])
    rust_zone = RustZone(
        name="test",
        grid=rust_grid,
        site_buses=[],
        word_buses=[],
        words_with_site_buses=[],
        sites_with_word_buses=[],
    )
    rust_mode = RustMode(
        name="all",
        zones=[0],
        bitstring_order=[RustLocAddr(0, 0, s) for s in range(3)],
    )
    arch_spec = ArchSpec.from_components(
        words=(word,),
        zones=(rust_zone,),
        modes=[rust_mode],
    )

    state = _make_execute_cz((LocationAddress(0, 0), LocationAddress(0, 1)))

    assert state.verify(arch_spec, (0,), (1,)) is False
    assert state.verify(arch_spec, (2,), (0,)) is False


def test_verify_custom_large_arch():
    """Verify works with multiword ArchSpec using the builder."""
    from bloqade.lanes.arch.build.blueprint import (
        ArchBlueprint,
        DeviceLayout,
        ZoneSpec,
        build_arch,
    )
    from bloqade.lanes.arch.build.topology import (
        HypercubeSiteTopology,
        HypercubeWordTopology,
    )

    bp = ArchBlueprint(
        zones={
            "gate": ZoneSpec(
                num_rows=2,
                num_cols=2,
                entangling=True,
                word_topology=HypercubeWordTopology(),
                site_topology=HypercubeSiteTopology(),
            )
        },
        layout=DeviceLayout(sites_per_word=4),
    )
    arch_spec = build_arch(bp).arch

    # Place qubits at paired locations
    state = _make_execute_cz(
        (
            LocationAddress(0, 0),
            LocationAddress(1, 0),
            LocationAddress(2, 1),
            LocationAddress(3, 1),
        )
    )
    assert state.verify(arch_spec, (0, 2), (1, 3)) is True
    assert state.verify(arch_spec, (0, 1), (2, 3)) is False
    assert state.verify(arch_spec, (0, 2), (1, 2)) is False
