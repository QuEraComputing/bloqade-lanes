"""Tests for UserMoved lattice element and ExecuteCZReturn user_move_layers extension."""

from bloqade.lanes.analysis.placement.lattice import (
    ConcreteState,
    ExecuteCZReturn,
    UserMoved,
)
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    ZoneAddress,
)


def _loc(z: int, w: int, s: int) -> LocationAddress:
    return LocationAddress(zone_id=z, word_id=w, site_id=s)


def _lane(z: int, w: int, s: int, d: Direction = Direction.FORWARD) -> SiteLaneAddress:
    return SiteLaneAddress(z, w, s, d)


def _concrete(layout):
    return ConcreteState(
        occupied=frozenset(),
        layout=layout,
        move_count=(0,) * len(layout),
    )


def test_user_moved_is_concrete_state():
    um = UserMoved.from_concrete_state(
        _concrete((_loc(0, 0, 0), _loc(0, 0, 1))),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=(_loc(0, 0, 0), _loc(0, 0, 1)),
    )
    assert isinstance(um, ConcreteState)


def test_user_moved_get_move_layers_returns_own_layers():
    layers = ((_lane(0, 0, 0),),)
    um = UserMoved.from_concrete_state(
        _concrete((_loc(0, 1, 0), _loc(0, 0, 1))),
        move_layers=layers,
        accumulated_move_layers=layers,
        pre_user_layout=(_loc(0, 0, 0), _loc(0, 0, 1)),
    )
    assert um.get_move_layers() == layers


def test_user_moved_get_reverse_moves_empty():
    um = UserMoved.from_concrete_state(
        _concrete((_loc(0, 1, 0),)),
        move_layers=((_lane(0, 0, 0),),),
        accumulated_move_layers=((_lane(0, 0, 0),),),
        pre_user_layout=(_loc(0, 0, 0),),
    )
    assert um.get_reverse_moves() == ()


def test_execute_cz_return_user_move_layers_default_empty():
    ecz = ExecuteCZReturn(
        occupied=frozenset(),
        layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
        move_count=(0, 0),
        active_cz_zones=frozenset([ZoneAddress(0)]),
        move_layers=((_lane(0, 0, 0),),),
        initial_layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
    )
    assert ecz.user_move_layers == ()
    assert len(ecz.return_move_layers) == 1  # only compiler reverse


def test_execute_cz_return_with_user_move_layers():
    compiler_lane = _lane(0, 0, 0)
    user_lane = _lane(0, 0, 1)
    ecz = ExecuteCZReturn(
        occupied=frozenset(),
        layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
        move_count=(0, 0),
        active_cz_zones=frozenset([ZoneAddress(0)]),
        move_layers=((compiler_lane,),),
        user_move_layers=((user_lane,),),
        initial_layout=(_loc(0, 0, 0), _loc(1, 0, 0)),
    )
    assert len(ecz.return_move_layers) == 2  # compiler_reverse + user_reverse
    assert ecz.return_move_layers[0] == (compiler_lane.reverse(),)
    assert ecz.return_move_layers[1] == (user_lane.reverse(),)
