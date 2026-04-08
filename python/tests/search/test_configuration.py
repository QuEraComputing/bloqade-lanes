"""Tests for ConfigurationNode."""

from bloqade.lanes.layout import LocationAddress, SiteLaneAddress
from bloqade.lanes.search.configuration import ConfigurationNode


def test_empty_configuration():
    node = ConfigurationNode(configuration={})
    assert len(node.configuration) == 0
    assert node.depth == 0
    assert node.parent is None
    assert node.parent_moves is None
    assert len(node.children) == 0


def test_configuration_with_atoms():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)
    node = ConfigurationNode(configuration={0: loc0, 1: loc1})

    assert node.configuration[0] == loc0
    assert node.configuration[1] == loc1


def test_config_key_is_hashable():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)
    node = ConfigurationNode(configuration={0: loc0, 1: loc1})

    key = node.config_key
    assert isinstance(key, frozenset)
    # Can be used as a dict key
    d = {key: "test"}
    assert d[key] == "test"


def test_config_key_order_independent():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)

    # Build same configuration in different insertion order
    node1 = ConfigurationNode(configuration={0: loc0, 1: loc1})
    node2 = ConfigurationNode(configuration={1: loc1, 0: loc0})

    assert node1.config_key == node2.config_key


def test_is_occupied():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)
    empty_loc = LocationAddress(0, 1)

    node = ConfigurationNode(configuration={0: loc0, 1: loc1})

    assert node.is_occupied(loc0) is True
    assert node.is_occupied(loc1) is True
    assert node.is_occupied(empty_loc) is False


def test_get_qubit_at():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)
    empty_loc = LocationAddress(0, 1)

    node = ConfigurationNode(configuration={0: loc0, 1: loc1})

    assert node.get_qubit_at(loc0) == 0
    assert node.get_qubit_at(loc1) == 1
    assert node.get_qubit_at(empty_loc) is None


def test_occupied_locations():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)
    node = ConfigurationNode(configuration={0: loc0, 1: loc1})

    assert node.occupied_locations == frozenset({loc0, loc1})


def test_cached_derived_views_and_lookup_methods_are_stable():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(1, 0)
    empty_loc = LocationAddress(0, 1)
    node = ConfigurationNode(configuration={0: loc0, 1: loc1})

    first_key = node.config_key
    first_occupied = node.occupied_locations

    # Repeated access should return stable cached objects.
    assert node.config_key is first_key
    assert node.occupied_locations is first_occupied

    # Lookup helpers should remain stable across repeated calls.
    for _ in range(3):
        assert node.get_qubit_at(loc0) == 0
        assert node.get_qubit_at(loc1) == 1
        assert node.get_qubit_at(empty_loc) is None
        assert node.is_occupied(loc0) is True
        assert node.is_occupied(loc1) is True
        assert node.is_occupied(empty_loc) is False


def test_path_to_root_at_root():
    node = ConfigurationNode(configuration={0: LocationAddress(0, 0)})
    assert node.path_to_root() == []


def test_path_to_root_three_levels():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(0, 1)
    loc2 = LocationAddress(1, 1)

    lane1 = SiteLaneAddress(0, 0, 0)
    lane2 = SiteLaneAddress(0, 1, 0)

    moves1 = frozenset({lane1})
    moves2 = frozenset({lane2})

    root = ConfigurationNode(configuration={0: loc0}, depth=0)
    child = ConfigurationNode(
        configuration={0: loc1},
        parent=root,
        parent_moves=moves1,
        depth=1,
    )
    grandchild = ConfigurationNode(
        configuration={0: loc2},
        parent=child,
        parent_moves=moves2,
        depth=2,
    )

    path = grandchild.path_to_root()
    assert len(path) == 2
    assert path[0] == moves1  # root → child
    assert path[1] == moves2  # child → grandchild


def test_to_move_program():
    loc0 = LocationAddress(0, 0)
    loc1 = LocationAddress(0, 1)

    lane1 = SiteLaneAddress(0, 0, 0)
    moves1 = frozenset({lane1})

    root = ConfigurationNode(configuration={0: loc0}, depth=0)
    child = ConfigurationNode(
        configuration={0: loc1},
        parent=root,
        parent_moves=moves1,
        depth=1,
    )

    program = child.to_move_program()
    assert len(program) == 1
    assert lane1 in program[0]
