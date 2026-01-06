from bloqade.lanes import kernel
from bloqade.lanes.analysis import atom
from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData
from bloqade.lanes.analysis.atom.lattice import AtomState, IListResult, Unknown
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)


@kernel
def main(cond: bool):
    state0 = move.load()
    state1 = move.fill(
        state0, location_addresses=(LocationAddress(0, 0), LocationAddress(1, 0))
    )
    state2 = move.move(state1, lanes=(SiteLaneAddress(0, 0, 0),))
    state3 = move.move(state2, lanes=(WordLaneAddress(0, 5, 0),))

    if cond:
        state4 = move.move(
            state3, lanes=(SiteLaneAddress(1, 0, 0, Direction.BACKWARD),)
        )
    else:
        state4 = move.cz(state3, zone_address=ZoneAddress(0))

    state5 = move.move(state4, lanes=(WordLaneAddress(0, 5, 0, Direction.BACKWARD),))
    state6 = move.move(state5, lanes=(SiteLaneAddress(0, 0, 0, Direction.BACKWARD),))
    move.store(state6)

    return [state0, state1, state2, state3, state4, state5, state6]


frame, result = atom.AtomInterpreter(kernel, arch_spec=get_arch_spec()).run(main)
main.print(analysis=frame.entries)


assert result == IListResult(
    data=(
        AtomState(
            data=AtomStateData(
                locations_to_qubit={},
                qubit_to_locations={},
                collision={},
                prev_lanes={},
                move_count={},
            )
        ),
        AtomState(
            data=AtomStateData(
                locations_to_qubit={
                    LocationAddress(word_id=0, site_id=0): 0,
                    LocationAddress(word_id=1, site_id=0): 1,
                },
                qubit_to_locations={
                    0: LocationAddress(word_id=0, site_id=0),
                    1: LocationAddress(word_id=1, site_id=0),
                },
                collision={},
                prev_lanes={},
                move_count={},
            )
        ),
        AtomState(
            data=AtomStateData(
                locations_to_qubit={
                    LocationAddress(word_id=1, site_id=0): 1,
                    LocationAddress(word_id=0, site_id=5): 0,
                },
                qubit_to_locations={
                    0: LocationAddress(word_id=0, site_id=5),
                    1: LocationAddress(word_id=1, site_id=0),
                },
                collision={},
                prev_lanes={
                    0: SiteLaneAddress(
                        word_id=0, site_id=0, bus_id=0, direction=Direction.FORWARD
                    )
                },
                move_count={0: 1},
            )
        ),
        AtomState(
            data=AtomStateData(
                locations_to_qubit={
                    LocationAddress(word_id=1, site_id=0): 1,
                    LocationAddress(word_id=1, site_id=5): 0,
                },
                qubit_to_locations={
                    0: LocationAddress(word_id=1, site_id=5),
                    1: LocationAddress(word_id=1, site_id=0),
                },
                collision={},
                prev_lanes={
                    0: WordLaneAddress(
                        word_id=0, site_id=5, bus_id=0, direction=Direction.FORWARD
                    )
                },
                move_count={0: 2},
            )
        ),
        Unknown(),
        Unknown(),
        Unknown(),
    )
)
