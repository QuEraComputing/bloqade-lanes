import io
import linecache
import uuid
import numpy as np
from typing import Any
from dataclasses import dataclass

from kirin import ir
from kirin.dialects.ilist import IList
from kirin.dialects import py
from kirin.prelude import python_basic, basic_no_opt, basic
from bloqade.rewrite.passes import aggressive_unroll
from bloqade.types import Qubit, MeasurementResult
from bloqade.lanes.dialects import move, place
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

kernel = lanes.kernel.add(annotate).add(place)

# %%

hypercube_dim = 4
word_size = 2**hypercube_dim

word_single_step = hypercube_dim - 1
word_two_step = hypercube_dim - 2
word_four_step = hypercube_dim - 3

# %%


def get_bus_id(start_word: int, end_word: int) -> int:
    """Get bus ID connecting two words."""
    bus_id = hypercube_dim - int(np.log2(abs(start_word - end_word)).astype(int)) - 1
    return bus_id


@dataclass
class Move:
    origin: int
    dest: int

    def get_word_lane_address(self) -> WordLaneAddress:
        word = self.origin if self.origin < self.dest else self.dest
        bus_id = get_bus_id(self.origin, self.dest)
        direction = Direction.FORWARD if self.origin < self.dest else Direction.BACKWARD
        return WordLaneAddress(word, 1, bus_id, direction)


Moves = list[Move]
Layer = list[Moves]
Circuit = list[Layer]

one_anc_postseect = [
    [[Move(0, 2), Move(6, 7), Move(1, 5)]],
    [[Move(2, 0), Move(5, 1), Move(7, 3)], [Move(0, 1), Move(5, 7)]],
    [[Move(1, 0), Move(3, 2), Move(7, 6)], [Move(4, 6), Move(0, 4), Move(1, 3)]],
    [[Move(2, 0), Move(3, 1), Move(6, 2)]],
    [[Move(1, 5), Move(0, 1), Move(2, 3)]],
    [[Move(1, 0), Move(3, 1)]],
]

register_size = (
    max(
        max(max(max(m.origin, m.dest) for m in moves) for moves in layer_move)
        for layer_move in one_anc_postseect
    )
    + 1
)

reg_locations = tuple(LocationAddress(x, 0) for x in range(register_size))


def _freeze_lanes(circuit: Circuit):
    """
    Build a fully-frozen (tuples all the way down) lane table.
    This is Python-side "constant" data that we can reference from generated kernels.
    """
    atom_active = [False] * register_size

    def get_site_lanes_activate(
        moves: Moves,
    ) -> tuple[SiteLaneAddress | WordLaneAddress, ...]:
        out: list[SiteLaneAddress | WordLaneAddress] = []
        for m in moves:
            if not atom_active[m.origin]:
                out.append(SiteLaneAddress(m.origin, 0, 0))
                # atom_active[m.origin] = True
        return tuple(out)

    def get_site_lanes_deactivate(
        moves: Moves,
    ) -> tuple[SiteLaneAddress | WordLaneAddress, ...]:
        out: list[SiteLaneAddress | WordLaneAddress] = []
        origns = set(m.origin for m in moves)
        for m in moves:
            if atom_active[m.dest] and m.dest not in origns:
                out.append(SiteLaneAddress(m.dest, 0, 0, Direction.BACKWARD))
                # atom_active[m.dest] = False
        return tuple(out)

    def get_word_lanes(moves: Moves) -> tuple[WordLaneAddress, ...]:
        for m in moves:
            atom_active[m.origin] = False
            atom_active[m.dest] = True
        return tuple(m.get_word_lane_address() for m in moves)

    def get_final_site_deactivations() -> tuple[SiteLaneAddress, ...]:
        out: list[SiteLaneAddress] = []
        for word_id in range(register_size):
            if atom_active[word_id]:
                out.append(SiteLaneAddress(word_id, 0, 0, Direction.BACKWARD))
                atom_active[word_id] = False
        return tuple(out)

    lanes = tuple(
        tuple(
            (
                get_site_lanes_activate(moves),
                get_site_lanes_deactivate(moves),
                get_word_lanes(moves),
            )
            for moves in layer
        )
        for layer in circuit
    )
    final_lanes = get_final_site_deactivations()
    return (lanes, final_lanes)


# Global frozen lane table (so it's eligible as a compile-time constant source)
LANES, FINAL_LANES = _freeze_lanes(one_anc_postseect)


def _make_sp_kernel_from_lanes(
    lanes_table,
    final_lanes,
    *,
    name: str = "sp_one_anc_postselect",
    cz_zone: int = 0,
):
    """
    Generates a state preparation kernel from a lane table.
    This avoids the use of dynamic construction with for loops inside the kernel.

    """
    lines: list[str] = []
    lines.append(f"def {name}():")
    lines.append("    state = move.load()")
    lines.append("    state = move.fill(state, location_addresses=reg_locations)")

    for layer_i, layer in enumerate(lanes_table):
        for step_i, step in enumerate(layer):
            if len(step[0]) > 0:
                lines.append(
                    f"    state = move.move(state, lanes=LANES[{layer_i}][{step_i}][0])"
                )
            if len(step[1]) > 0:
                lines.append(
                    f"    state = move.move(state, lanes=LANES[{layer_i}][{step_i}][1])"
                )
            if len(step[2]) > 0:
                lines.append(
                    f"    state = move.move(state, lanes=LANES[{layer_i}][{step_i}][2])"
                )
        if layer_i < len(lanes_table) - 1:  # No CZ after last layer
            lines.append(
                f"    state = move.cz(state, zone_address=ZoneAddress({cz_zone}))"
            )
    lines.append("    return state")

    src = "\n".join(lines)

    # Make inspect.getsource() work for exec-generated functions
    filename = f"<generated:{name}:{uuid.uuid4()}>"
    compiled = compile(src, filename, "exec")
    linecache.cache[filename] = (
        len(src),
        None,
        [line + "\n" for line in src.splitlines()],
        filename,
    )

    src = "\n".join(lines)

    ns: dict[str, Any] = {}
    exec(compiled, globals(), ns)
    fn = ns[name]
    return kernel()(fn)


sp_one_anc_postselect_moves = _make_sp_kernel_from_lanes(LANES, FINAL_LANES)

@kernel 
def measurements(state):
    fut = move.end_measure(state, zone_addresses=(ZoneAddress(0),))
    res_0 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[0])
    res_1 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[1])
    res_2 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[2])
    res_3 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[3])
    res_4 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[4])
    res_5 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[5])
    res_6 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[6])
    res_7 = move.get_future_result(fut, zone_address=ZoneAddress(0),location_address=reg_locations[7])
    res = IList([res_0, res_1, res_2, res_3, res_4, res_5, res_6, res_7])
    return state, res

@kernel 
def detectors(res: IList[MeasurementResult, Any]):
    # det = IList([res[a] for a in [0,1,2,3,4,5,6,7]])
    # annotate.set_detector(det, coordinates=[0, 0])
    annotate.set_detector(res, coordinates=[0, 0])


@kernel 
def sp_one_anc_postselect():
    state =sp_one_anc_postselect_moves()
    state = move.move(state, lanes=FINAL_LANES)
    state_and_res = measurements(state)
    state = state_and_res[0]
    res = state_and_res[1]
    detectors(res)

    
    move.store(state)

#unroll
aggressive_unroll.AggressiveUnroll(sp_one_anc_postselect.dialects)(sp_one_anc_postselect)


sp_one_anc_postselect.print()

arch_spec = generate_arch_hypercube(hypercube_dims=hypercube_dim, word_size_y=1)
# arch_spec = get_arch_spec()

# debugger(mt=sp_one_anc_postselect, arch_spec=arch_spec, atom_marker="o")

# exit()

# convert to squin with noise model
noise_model = generate_simple_noise_model()
squin_mt = MoveToSquin(
    arch_spec=arch_spec,
    noise_model=noise_model,
).emit(sp_one_anc_postselect)

squin_mt.print()


noise_mt = squin_to_stim(squin_mt)

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
