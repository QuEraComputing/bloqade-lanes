from bloqade.lanes import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import (
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.visualize import debugger

lane1 = SiteLaneAddress(word_id=0, site_id=0, bus_id=0)
lane2 = WordLaneAddress(word_id=0, site_id=5, bus_id=0)


@kernel
def main(cond: bool):
    state = move.load()
    state = move.fill(
        state, location_addresses=(LocationAddress(0, 0), LocationAddress(1, 0))
    )
    state = move.move(state, lanes=(lane1,))
    state = move.move(state, lanes=(lane2,))
    state = move.cz(state, zone_address=ZoneAddress(0))
    state = move.move(state, lanes=(lane2.reverse(),))
    state = move.move(state, lanes=(lane1.reverse(),))
    move.store(state)


arch_spec = get_arch_spec()
debugger(main, arch_spec, interactive=True)

squin_kernel = MoveToSquin(
    arch_spec=get_arch_spec(),
    noise_model=generate_simple_noise_model(),
).emit(main)

squin_kernel.print()
