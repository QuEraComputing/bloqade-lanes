from typing import Any

from bloqade.gemini import logical as gemini_logical
from kirin.dialects import ilist

from bloqade import annotate, qubit, squin, types
from bloqade.lanes.logical_mvp import (
    compile_to_physical_stim_program,
)

kernel = squin.kernel.add(gemini_logical.dialect).add(annotate)
kernel.run_pass = squin.kernel.run_pass


@kernel
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    annotate.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0])
    annotate.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1])
    annotate.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2])


@kernel
def set_observable(meas: ilist.IList[types.MeasurementResult, Any], index: int):
    annotate.set_observable([meas[0], meas[1], meas[5]])


@kernel
def main():
    # see arXiv: 2412.15165v1, Figure 3a
    reg = qubit.qalloc(5)
    squin.broadcast.t(reg)

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
        set_observable(measurements[i], i)


result = compile_to_physical_stim_program(main)
print(result)
