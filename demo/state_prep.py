import io
import numpy as np
from typing import Any

from kirin.dialects.ilist import IList
from bloqade.types import Qubit
from bloqade.lanes.dialects import move
from bloqade import annotate
from bloqade import lanes
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.layout.encoding import (
    Direction,
    LocationAddress,
    SiteLaneAddress,
    WordLaneAddress,
    ZoneAddress,
)
from bloqade.lanes.visualize import debugger
from bloqade.stim.upstream.from_squin import squin_to_stim
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade import squin

kernel = lanes.kernel.add(annotate)


hypercube_dim = 4
word_size = 2**hypercube_dim
register_size = 9

reg_locations = tuple(LocationAddress(x, 0) for x in range(register_size))

word_single_step = hypercube_dim - 1
word_two_step = hypercube_dim - 2
word_four_step = hypercube_dim - 3


@dataclass
class Move:
    origin: int
    dest: int


Moves = list[Move]
Layer = list[Moves]
Circuit = list[Layer]
@kernel
def sp_one_anc_postselect():
    state = move.load()
    state = move.fill(state, location_addresses=reg_locations)
    state = move.move(
        state,
        lanes=(
            SiteLaneAddress(0, 0, 0),
            SiteLaneAddress(6, 0, 0),
            SiteLaneAddress(1, 0, 0),
            SiteLaneAddress(7, 0, 0),
        ),
    )
    state = move.move(
        state,
        lanes=(
            WordLaneAddress(0, 1, word_two_step),
            WordLaneAddress(6, 1, word_single_step),
            WordLaneAddress(1, 1, word_four_step),
        ),
    )
    # state = move.cz(state, zone_address=ZoneAddress(0))
    # state = move.move(state,
    #                   lanes=(
    #                         WordLaneAddress(0,1,word_two_step, Direction.BACKWARD),
    #                         WordLaneAddress(3,1,word_four_step, Direction.BACKWARD),
    #                         WordLaneAddress(1,1,word_four_step, Direction.BACKWARD),
    #                   ))


arch_spec = generate_arch_hypercube(hypercube_dims=hypercube_dim, word_size_y=2)
# arch_spec = get_arch_spec()

debugger(mt=sp_one_anc_postselect, arch_spec=arch_spec, atom_marker="o")
exit()

# convert to squin with noise model
noise_model = generate_simple_noise_model()
squin_mt = MoveToSquin(
    arch_spec=arch_spec,
    noise_model=noise_model,
).emit(sp_one_anc_postselect)

# squin_mt.print()


# add detectors
@squin.kernel.add(annotate)
def detectors():
    qreg: IList[Qubit, Any] = squin.qalloc(register_size)
    squin_mt()
    m = squin.broadcast.measure(qreg)
    annotate.set_detector(m, coordinates=[0, 0])


noise_mt = squin_to_stim(detectors)

# print and write to file
buf = io.StringIO()
emit = EmitStimMain(dialects=noise_mt.dialects, io=buf)
emit.initialize()
emit.run(node=noise_mt)

out_path = "demo/state_prep.stim"
with open(out_path, "w", encoding="utf-8") as f:
    emit = EmitStimMain(dialects=noise_mt.dialects, io=f)
    emit.initialize()
    emit.run(node=noise_mt)

print(buf.getvalue().strip())
