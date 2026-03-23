import io
import math
from collections import Counter
from typing import Any

import pytest
from bloqade.decoders.dialects import annotate
from bloqade.squin.rewrite.non_clifford_to_U3 import RewriteNonCliffordToU3
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import rewrite
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.arch import logical
from bloqade.gemini.arch.impls import generate_arch_hypercube
from bloqade.gemini.arch.logical import get_arch_spec
from bloqade.gemini.compile import (
    compile_squin_to_move,
    transversal_rewrites,
)
from bloqade.gemini.logical.rewrite.initialize import (
    __RewriteU3ToInitialize as _RewriteU3ToInitialize,
)
from bloqade.gemini.noise_model import generate_simple_noise_model
from bloqade.gemini.rewrite.circuit2place import (
    GeminiRewritePlaceOperations,
    RewriteInitializeToLogicalInitialize,
)
from bloqade.lanes.heuristics import logical_layout
from bloqade.lanes.heuristics.logical_placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import (
    always_merge_heuristic,
    default_merge_heuristic,
    squin_to_move,
)
from bloqade.lanes.utils import check_circuit


@gemini_logical.kernel(verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0])
    annotate.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1])
    annotate.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2])


@gemini_logical.kernel(verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_observable([meas[0], meas[1], meas[5]])


@gemini_logical.kernel(aggressive_unroll=True)
def main():
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

    squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
    squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
    squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
    squin.sqrt_x_adj(reg[0])
    squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
    squin.broadcast.sqrt_y_adj(reg)

    measurements = gemini_logical.terminal_measure(reg)

    for i in range(len(reg)):
        set_detector(measurements[i])
        set_observable(measurements[i])


def _compile_to_stim_with_merge_heuristic(mt, merge_heuristic):
    noise_model = generate_simple_noise_model()
    move_mt = squin_to_move(
        mt,
        layout_heuristic=logical_layout.LogicalLayoutHeuristic(
            arch_spec=get_arch_spec()
        ),
        placement_strategy=LogicalPlacementStrategyNoHome(arch_spec=get_arch_spec()),
        insert_return_moves=True,
        merge_heuristic=merge_heuristic,
        pre_rewrites=(
            rewrite.Walk(RewriteNonCliffordToU3()),
            rewrite.Walk(_RewriteU3ToInitialize()),
        ),
        pre_place_rewrites=(RewriteInitializeToLogicalInitialize(),),
        place_rewrite=GeminiRewritePlaceOperations(),
    )
    move_mt = transversal_rewrites(move_mt)
    transformer = MoveToSquin(
        arch_spec=generate_arch_hypercube(4),
        logical_initialization=logical.steane7_initialize,
        noise_model=noise_model,
        aggressive_unroll=False,
    )
    physical_squin = transformer.emit(move_mt)
    stim_kernel = squin_to_stim(physical_squin)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=stim_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=stim_kernel)
    return buf.getvalue().strip()


def _normalized_gate_ops(stim_program: str) -> Counter[str]:
    """Remove all comments and noise operations (PAULI_CHANNEL_* and I_ERROR[...]) from the stim program."""
    ops: list[str] = []
    for raw_line in stim_program.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            line = line.split(" #", 1)[0].rstrip()
        normalized = " ".join(line.split())
        if normalized:
            if normalized.startswith("PAULI_CHANNEL_") or normalized.startswith(
                "I_ERROR["
            ):
                continue
            ops.append(normalized)
    return Counter(ops)


@pytest.mark.slow
def test_default_and_always_merge_have_same_operations():
    default_program = _compile_to_stim_with_merge_heuristic(
        main, default_merge_heuristic
    )
    always_program = _compile_to_stim_with_merge_heuristic(main, always_merge_heuristic)

    default_ops = _normalized_gate_ops(default_program)
    always_ops = _normalized_gate_ops(always_program)
    print(default_program)
    print("--------------------------------")
    print(always_program)

    assert default_ops == always_ops


@pytest.mark.slow
def test_logical_compilation():
    from bloqade.rewrite.passes import AggressiveUnroll

    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        reg = qubit.qalloc(5)
        squin.broadcast.u3(0.3041 * math.pi, 0.25 * math.pi, 0.0, reg)

        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_y_adj(reg)

    logical_move = compile_squin_to_move(main, no_raise=False)

    decompiled_squin = MoveToSquin(get_arch_spec()).emit(logical_move)

    AggressiveUnroll(main.dialects).fixpoint(main)

    assert check_circuit(main, decompiled_squin)
