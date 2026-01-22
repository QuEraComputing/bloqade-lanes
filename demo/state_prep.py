import io
from typing import Any

from kirin.dialects.ilist import IList
from bloqade.types import Qubit
from bloqade.lanes.dialects import move
from bloqade import annotate
from bloqade import lanes
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.arch.gemini.impls import generate_arch
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
word_size = 2 ** hypercube_dim
register_size = 9

reg_locations = tuple(LocationAddress(x, 0) for x in range(register_size))


@kernel
def prepare_state():
    state = move.load()
    state = move.fill(
        state, location_addresses=reg_locations
    )
    state = move.move(state, lanes=(SiteLaneAddress(0, 0, 0),))
    state = move.local_rz(state, rotation_angle=0.5, location_addresses=(LocationAddress(5, 0),))

arch_spec = generate_arch(hypercube_dims=hypercube_dim, word_size_y=1)
# arch_spec = get_arch_spec()
    
# debugger(mt=prepare_state,
#              arch_spec=arch_spec, 
#              atom_marker="o")

noise_model = generate_simple_noise_model()

squin_mt = MoveToSquin(
   arch_spec=arch_spec,
   noise_model=noise_model, 
).emit(prepare_state)

squin_mt.print()


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



