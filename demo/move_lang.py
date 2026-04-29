from bloqade.lanes._prelude import kernel
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.encoding import (
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.dialects import move
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.transform import MoveToSquinPhysical

lane_word = WordLaneAddress(0, 1, 0)
lane_site = SiteLaneAddress(1, 0, 0)


@kernel
def main(cond: bool):
    state = move.load()
    # Fill two qubits in the same word
    state = move.fill(
        state, location_addresses=(LocationAddress(0, 0), LocationAddress(0, 1))
    )
    # Word bus: move atom at site 1 from word 0 → word 1
    state = move.move(state, lanes=(lane_word,))
    # Site bus: move atom in word 1 from site 1 → site 0 (aligns for CZ)
    state = move.move(state, lanes=(lane_site.reverse(),))
    state = move.cz(state, zone_address=ZoneAddress(0))
    # Reverse moves
    state = move.move(state, lanes=(lane_site,))
    state = move.move(state, lanes=(lane_word.reverse(),))
    move.store(state)


arch_spec = get_arch_spec()

squin_kernel = MoveToSquinPhysical(
    arch_spec=get_arch_spec(),
    noise_model=generate_simple_noise_model(),
).emit(main)

squin_kernel.print()
