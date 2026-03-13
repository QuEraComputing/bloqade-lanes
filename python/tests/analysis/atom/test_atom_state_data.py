from bloqade.lanes import layout
from bloqade.lanes.analysis.atom import atom_state_data
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.layout.path import PathFinder


def test_hash():
    data1 = atom_state_data.AtomStateData(
        locations_to_qubit={layout.LocationAddress(0, 0): 1},
        qubit_to_locations={1: layout.LocationAddress(0, 0)},
        collision={},
        prev_lanes={},
        move_count={},
    )

    assert hash(data1) == hash(
        (
            atom_state_data.AtomStateData,
            frozenset([(layout.LocationAddress(0, 0), 1)]),
            frozenset([(1, layout.LocationAddress(0, 0))]),
            frozenset([]),
            frozenset([]),
            frozenset([]),
        )
    )


def test_add_atoms():
    atom_state = atom_state_data.AtomStateData()

    new_atom_state = atom_state.add_atoms(
        {0: layout.LocationAddress(0, 0), 1: layout.LocationAddress(1, 0)}
    )

    expected_atom_state = atom_state_data.AtomStateData(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(1, 0),
        },
        collision={},
        prev_lanes={},
        move_count={},
    )

    assert new_atom_state == expected_atom_state


def test_apply_moves():
    atom_state = atom_state_data.AtomStateData(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(1, 0),
        },
        collision={},
        prev_lanes={},
        move_count={},
    )

    path_finder = PathFinder(logical.get_arch_spec())

    new_atom_state = atom_state.apply_moves(
        lanes=(layout.SiteLaneAddress(0, 0, 0),), path_finder=path_finder
    )

    expected_atom_state = atom_state_data.AtomStateData(
        locations_to_qubit={
            layout.LocationAddress(0, 5): 0,
            layout.LocationAddress(1, 0): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 5),
            1: layout.LocationAddress(1, 0),
        },
        collision={},
        prev_lanes={
            0: layout.SiteLaneAddress(0, 0, 0),
        },
        move_count={0: 1},
    )

    assert new_atom_state == expected_atom_state


def test_apply_moves_with_collision():
    atom_state = atom_state_data.AtomStateData(
        locations_to_qubit={
            layout.LocationAddress(0, 0): 0,
            layout.LocationAddress(0, 5): 1,
        },
        qubit_to_locations={
            0: layout.LocationAddress(0, 0),
            1: layout.LocationAddress(0, 5),
        },
        collision={},
        prev_lanes={},
        move_count={},
    )

    path_finder = PathFinder(logical.get_arch_spec())

    new_atom_state = atom_state.apply_moves(
        lanes=(lane_address := layout.SiteLaneAddress(0, 0, 0),),
        path_finder=path_finder,
    )

    expected_atom_state = atom_state_data.AtomStateData(
        locations_to_qubit={},
        qubit_to_locations={},
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
