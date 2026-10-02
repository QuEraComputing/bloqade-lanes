from bloqade.lanes.bytecode.encoding import LaneAddress, MoveType
from bloqade.lanes.dialects import move
from bloqade.lanes.metrics import UsedBuses, get_used_buses
from bloqade.lanes.prelude import kernel

_ZONE_FORWARD = LaneAddress(MoveType.ZONE, 0, 0, 1, zone_id=0)
_ZONE_BACKWARD = _ZONE_FORWARD.reverse()
_ZONE_BUS_0 = LaneAddress(MoveType.ZONE, 0, 0, 0, zone_id=0)
_WORD_ZONE_1 = LaneAddress(MoveType.WORD, 0, 0, 2, zone_id=1)
_WORD_ZONE_1_BACKWARD = _WORD_ZONE_1.reverse()
_WORD_ZONE_0 = LaneAddress(MoveType.WORD, 0, 0, 2, zone_id=0)
_SITE_ZONE_1 = LaneAddress(MoveType.SITE, 0, 0, 3, zone_id=1)
_SITE_ZONE_1_BACKWARD = _SITE_ZONE_1.reverse()
_SITE_ZONE_0 = LaneAddress(MoveType.SITE, 0, 0, 3, zone_id=0)


@kernel
def _move_kernel():
    state = move.load()
    state = move.move(state, lanes=(_SITE_ZONE_1, _SITE_ZONE_1_BACKWARD, _SITE_ZONE_0))
    state = move.move(state, lanes=(_WORD_ZONE_1, _WORD_ZONE_1_BACKWARD, _WORD_ZONE_0))
    state = move.move(state, lanes=(_ZONE_FORWARD, _ZONE_BACKWARD, _ZONE_BUS_0))
    state = move.move(state, lanes=(_SITE_ZONE_1, _WORD_ZONE_1, _ZONE_BACKWARD))
    move.store(state)


@kernel
def _kernel_without_moves():
    state = move.load()
    move.store(state)


def test_get_used_buses_counts_buses_per_move():
    result = get_used_buses(_move_kernel)
    assert result == UsedBuses(
        zone={0: 1, 1: 2},
        word={(0, 2): 1, (1, 2): 2},
        site={(0, 3): 1, (1, 3): 2},
    )
    assert list(result.zone) == [0, 1]
    assert list(result.word) == [(0, 2), (1, 2)]
    assert list(result.site) == [(0, 3), (1, 3)]


def test_get_used_buses_returns_empty_dicts_for_a_kernel_without_moves():
    assert get_used_buses(_kernel_without_moves) == UsedBuses(zone={}, word={}, site={})
