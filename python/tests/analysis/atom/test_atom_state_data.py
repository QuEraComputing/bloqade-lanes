import pytest
from kirin.interp import InterpreterError

from bloqade.gemini.arch import logical
from bloqade.lanes import layout
from bloqade.lanes.analysis.atom import atom_state_data


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
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(1, 0),
        },
    )

    arch_spec = logical.get_arch_spec()

    new_atom_state = atom_state.apply_moves(
        lanes=(layout.SiteLaneAddress(0, 0, 0),), arch_spec=arch_spec
    )

    expected_atom_state = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={
            layout.LocationAddress(0, 5): 0,
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 5),
            1: layout.LocationAddress(1, 0),
        },
        prev_lanes={
            0: layout.SiteLaneAddress(0, 0, 0),
        },
        move_count={0: 1},
    )

    assert new_atom_state == expected_atom_state


def test_apply_moves_with_collision():
    atom_state = atom_state_data.AtomStateData.from_fields(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(0, 5): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(0, 5),
        },
    )

    arch_spec = logical.get_arch_spec()

    new_atom_state = atom_state.apply_moves(
        lanes=(lane_address := layout.SiteLaneAddress(0, 0, 0),),
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
    atom_state = atom_state_data.AtomStateData.new(
        [
            layout.LocationAddress(0, 0),
            layout.LocationAddress(0, 1),
            layout.LocationAddress(1, 0),
        ]
    )

    arch_spec = logical.get_arch_spec()

    controls, targets, unpaired = atom_state.get_qubit_pairing(
        zone_address=layout.ZoneAddress(0), arch_spec=arch_spec
    )

    assert set(controls) == set()
    assert set(targets) == set()
    assert set(unpaired) == set(range(3))


def test_get_qubit_pairing_with_pairs():
    atom_state = atom_state_data.AtomStateData.new(
        [
            layout.LocationAddress(0, 0),
            layout.LocationAddress(0, 5),
            layout.LocationAddress(1, 0),
            layout.LocationAddress(1, 5),
            layout.LocationAddress(0, 1),
            layout.LocationAddress(3, 5),
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
