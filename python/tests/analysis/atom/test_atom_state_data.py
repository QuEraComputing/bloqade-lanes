import pytest
from kirin.interp import InterpreterError

from bloqade.lanes import layout
from bloqade.lanes.analysis.atom import atom_state_data
from bloqade.lanes.arch.gemini import logical


def test_hash():
    data1 = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={layout.LocationAddress(0, 0): 1},
        qubit_to_locations={1: layout.LocationAddress(0, 0)},
    )
    data2 = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={layout.LocationAddress(0, 0): 1},
        qubit_to_locations={1: layout.LocationAddress(0, 0)},
    )
    assert hash(data1) == hash(data2)


def test_add_atoms():
    atom_state = atom_state_data.AtomStateData()

    new_atom_state = atom_state.add_atoms(
        {0: layout.LocationAddress(0, 0), 1: layout.LocationAddress(1, 0)}
    )

    expected_atom_state = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(1, 0),
        },
    )

    assert new_atom_state == expected_atom_state


def test_apply_moves():
    atom_state = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(4, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(4, 0),
        },
    )

    arch_spec = logical.get_arch_spec()

    new_atom_state = atom_state.apply_moves(
        lanes=(layout.WordLaneAddress(0, 0, 0),), arch_spec=arch_spec
    )

    expected_atom_state = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={
            layout.LocationAddress(1, 0): 0,
            layout.LocationAddress(4, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(1, 0),
            1: layout.LocationAddress(4, 0),
        },
        prev_lanes={
            0: layout.WordLaneAddress(0, 0, 0),
        },
        move_count={0: 1},
    )

    assert new_atom_state == expected_atom_state


def test_apply_moves_with_collision():
    atom_state = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(1, 0),
        },
    )

    arch_spec = logical.get_arch_spec()

    new_atom_state = atom_state.apply_moves(
        lanes=(lane_address := layout.WordLaneAddress(0, 0, 0),),
        arch_spec=arch_spec,
    )

    expected_atom_state = atom_state_data.AtomStateData.from_fields(
        collision={0: 1},
        prev_lanes={
            0: lane_address,
        },
        move_count={0: 1},
    )

    assert new_atom_state == expected_atom_state


def test_get_qubit_pairing():
    # CZ partners: word 0 ↔ word 1.
    # Qubit 0 at (w=0,s=0) pairs with qubit 2 at (w=1,s=0).
    # Qubit 1 at (w=2,s=0) has no partner at (w=3,s=0) — unpaired.
    atom_state = atom_state_data.AtomStateData.new(
        [
            layout.LocationAddress(0, 0),
            layout.LocationAddress(2, 0),
            layout.LocationAddress(1, 0),
        ]
    )

    arch_spec = logical.get_arch_spec()

    controls, targets, unpaired = atom_state.get_qubit_pairing(
        zone_address=layout.ZoneAddress(0), arch_spec=arch_spec
    )

    assert set(controls) == {0}
    assert set(targets) == {2}
    assert set(unpaired) == {1}


def test_get_qubit_pairing_with_pairs():
    # CZ partners: word 0 ↔ word 1, word 2 ↔ word 3.
    # Qubit 0 at (w=0,s=0) ↔ qubit 1 at (w=1,s=0) — paired
    # Qubit 2 at (w=2,s=0) ↔ qubit 3 at (w=3,s=0) — paired
    # Qubit 4 at (w=4,s=0) — unpaired (no qubit at (w=5,s=0))
    atom_state = atom_state_data.AtomStateData.new(
        [
            layout.LocationAddress(0, 0),  # qubit 0 — pairs with qubit 1
            layout.LocationAddress(1, 0),  # qubit 1 — pairs with qubit 0
            layout.LocationAddress(2, 0),  # qubit 2 — pairs with qubit 3
            layout.LocationAddress(3, 0),  # qubit 3 — pairs with qubit 2
            layout.LocationAddress(4, 0),  # qubit 4 — unpaired
        ]
    )

    arch_spec = logical.get_arch_spec()

    controls, targets, unpaired = atom_state.get_qubit_pairing(
        zone_address=layout.ZoneAddress(0), arch_spec=arch_spec
    )

    assert set(controls) == {0, 2}
    assert set(targets) == {1, 3}
    assert set(unpaired) == {4}


def test_add_atoms_duplicate_qubit_raises():
    atom_state = atom_state_data.AtomStateData.new([layout.LocationAddress(0, 0)])
    with pytest.raises(InterpreterError, match="already exists"):
        atom_state.add_atoms({0: layout.LocationAddress(1, 0)})


def test_add_atoms_occupied_location_raises():
    atom_state = atom_state_data.AtomStateData.new([layout.LocationAddress(0, 0)])
    with pytest.raises(InterpreterError, match="occupied"):
        atom_state.add_atoms({1: layout.LocationAddress(0, 0)})


def test_apply_moves_invalid_lane_returns_none():
    atom_state = atom_state_data.AtomStateData.new([layout.LocationAddress(0, 0)])
    arch_spec = logical.get_arch_spec()

    # Use a lane with an invalid bus_id
    invalid_lane = layout.LaneAddress(layout.MoveType.SITE, 0, 0, 99)
    result = atom_state.apply_moves(lanes=(invalid_lane,), arch_spec=arch_spec)
    assert result is None


def test_get_qubit_pairing_invalid_zone_raises():
    atom_state = atom_state_data.AtomStateData.new([layout.LocationAddress(0, 0)])
    arch_spec = logical.get_arch_spec()

    with pytest.raises(InterpreterError, match="Invalid zone address"):
        atom_state.get_qubit_pairing(
            zone_address=layout.ZoneAddress(99), arch_spec=arch_spec
        )


def test_get_qubit_empty_location():
    atom_state = atom_state_data.AtomStateData.new([layout.LocationAddress(0, 0)])
    assert atom_state.get_qubit(layout.LocationAddress(1, 0)) is None


def test_empty_state():
    atom_state = atom_state_data.AtomStateData()
    assert len(atom_state.locations_to_qubit) == 0
    assert len(atom_state.qubit_to_locations) == 0
    assert len(atom_state.collision) == 0
    assert len(atom_state.prev_lanes) == 0
    assert len(atom_state.move_count) == 0


def test_properties_return_expected_values():
    atom_state = atom_state_data.AtomStateData.new(
        [layout.LocationAddress(0, 0), layout.LocationAddress(1, 0)]
    )
    assert atom_state.locations_to_qubit == {
        layout.LocationAddress(0, 0): 0,
        layout.LocationAddress(1, 0): 1,
    }
    assert atom_state.qubit_to_locations == {
        0: layout.LocationAddress(0, 0),
        1: layout.LocationAddress(1, 0),
    }
    assert atom_state.collision == {}
    assert atom_state.prev_lanes == {}
    assert atom_state.move_count == {}


def test_equality_with_non_atom_state():
    atom_state = atom_state_data.AtomStateData()
    assert atom_state != "not an atom state"
    assert atom_state != 42


def test_copy():
    atom_state = atom_state_data.AtomStateData.new(
        [layout.LocationAddress(0, 0), layout.LocationAddress(1, 0)]
    )
    copied = atom_state.copy()
    assert atom_state == copied
    assert atom_state is not copied


def test_repr_matches_legacy_dataclass_format():
    """AtomStateData must keep a useful repr (was generated by the old
    ``@dataclass(frozen=True, _inner repr=False)`` declaration). The
    wrapper migration in #466 dropped it; verify it is restored."""
    atom_state = atom_state_data.AtomStateData()
    assert repr(atom_state) == "AtomStateData()"


def test_inner_is_immutable_after_construction():
    """AtomStateData was @dataclass(frozen=True) before the wrapper
    migration. Re-establish that ``_inner`` cannot be rebound after
    construction so cached_property results never go stale relative to
    the underlying Rust state."""
    from bloqade.lanes.bytecode import AtomStateData as _RustAtomStateData

    atom_state = atom_state_data.AtomStateData()
    with pytest.raises(AttributeError, match="immutable"):
        atom_state._inner = _RustAtomStateData()


def test_from_inner_still_works_under_immutability_guard():
    """The from_inner fast path must not be blocked by the immutability
    guard — it constructs a fresh wrapper, so the first ``_inner``
    assignment must succeed."""
    from bloqade.lanes.bytecode import AtomStateData as _RustAtomStateData

    rust = _RustAtomStateData()
    wrapped = atom_state_data.AtomStateData.from_inner(rust)
    assert wrapped._inner is rust
