"""Tests for ExecuteCZ.verify() — blockade-radius validation of CZ placements."""

from bloqade.geometry.dialects import grid

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement.lattice import ExecuteCZ
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout import LocationAddress
from bloqade.lanes.layout.word import Word


def _make_execute_cz(
    layout_tuple: tuple[layout.LocationAddress, ...],
    active_cz_zones: frozenset[layout.ZoneAddress] | None = None,
) -> ExecuteCZ:
    n = len(layout_tuple)
    return ExecuteCZ(
        occupied=frozenset(),
        layout=layout_tuple,
        move_count=(0,) * n,
        active_cz_zones=active_cz_zones or frozenset([layout.ZoneAddress(0)]),
        move_layers=(),
    )


def test_verify_one_pair():
    """Qubits at (0,0) and (0,5) are CZ pairs on logical arch"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz(
        (layout.LocationAddress(0, 0), layout.LocationAddress(0, 5))
    )
    assert state.verify(arch_spec, (0,), (1,))
    assert state.verify(arch_spec, (1,), (0,))


def test_verify_one_pair_unblockaded():
    """Qubits at (0,0) and (0,1) are CZ pairs on logical arch"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz(
        (layout.LocationAddress(0, 0), layout.LocationAddress(0, 1))
    )
    assert state.verify(arch_spec, (0,), (1,)) is False
    assert state.verify(arch_spec, (1,), (0,)) is False


def test_verify_length_mismatch():
    """Qubits at (0,0) and (0,1) are CZ pairs on logical arch"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz(
        (layout.LocationAddress(0, 0), layout.LocationAddress(0, 5))
    )
    assert state.verify(arch_spec, (0, 0), (1,)) is False
    assert state.verify(arch_spec, (1,), (0, 0)) is False


def test_verify_invalid_indices():
    """Test archspec w/invalid indices in one word"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz(
        (layout.LocationAddress(0, 0), layout.LocationAddress(0, 5))
    )
    assert state.verify(arch_spec, (-1,), (1,)) is False
    assert state.verify(arch_spec, (2,), (0,)) is False
    assert state.verify(arch_spec, (-1,), (0,)) is False
    assert state.verify(arch_spec, (0,), (2,)) is False


def test_verify_multi_word():
    """Test archspec w/multiple words"""
    arch_spec = logical.get_arch_spec()
    state = _make_execute_cz(
        (
            layout.LocationAddress(0, 0),
            layout.LocationAddress(1, 5),
            layout.LocationAddress(1, 0),
            layout.LocationAddress(0, 5),
        )
    )
    assert state.verify(arch_spec, (0,), (3,))
    assert state.verify(arch_spec, (1,), (2,))
    assert state.verify(arch_spec, (1, 0), (2, 3))
    assert state.verify(arch_spec, (1, 1), (2, 3)) is False
    assert state.verify(arch_spec, (1, 4), (2, 0)) is False


def test_verify_no_czs():
    """Test archspec w/no cz pairs"""
    word_grid = grid.Grid.from_positions([0.0, 1.0, 2.0], [0.0])
    word = Word(
        word_grid,
        ((0, 0), (1, 0), (2, 0)),
    )

    arch_spec = layout.ArchSpec(
        (word,),
        ((0,),),
        (0,),
        frozenset([0]),
        frozenset(),
        frozenset(),
        (),
        (),
    )

    state = _make_execute_cz(
        (layout.LocationAddress(0, 0), layout.LocationAddress(0, 1))
    )

    assert state.verify(arch_spec, (0,), (1,)) is False
    assert state.verify(arch_spec, (2,), (0,)) is False


def test_verify_custom_large_arch():
    """Verify works with multiword ArchSpec where site 0<->2 and 1<->3."""
    word_grid = grid.Grid.from_positions([0.0, 2.0], [0.0, 10.0])
    words = tuple(
        Word(
            word_grid.shift(10.0 * ix, 0.0),
            ((0, 0), (0, 1), (1, 0), (1, 1)),
            has_cz=(
                LocationAddress(ix, 2),
                LocationAddress(ix, 3),
                LocationAddress(ix, 0),
                LocationAddress(ix, 1),
            ),
        )
        for ix in range(4)
    )
    arch_spec = layout.ArchSpec(
        words,
        (tuple(range(4)),),
        (0,),
        frozenset([0]),
        frozenset(),
        frozenset(),
        (),
        (),
    )

    state = _make_execute_cz(
        (
            layout.LocationAddress(0, 0),
            layout.LocationAddress(0, 2),
            layout.LocationAddress(3, 1),
            layout.LocationAddress(3, 3),
        )
    )
    assert state.verify(arch_spec, (0, 2), (1, 3)) is True
    assert state.verify(arch_spec, (0, 1), (2, 3)) is False
    assert state.verify(arch_spec, (0, 2), (1, 2)) is False
