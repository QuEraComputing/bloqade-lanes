import math
from typing import Any

import numpy as np
from bloqade.decoders import BpLsdDecoder
from bloqade.gemini import logical
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.lanes.device import GeminiLogicalSimulator


@logical.kernel(aggressive_unroll=True, verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]):
    return [
        squin.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0]),
        squin.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1]),
        squin.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2]),
    ]


@logical.kernel(aggressive_unroll=True, verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any], index: int):
    return squin.set_observable([meas[0], meas[1], meas[5]], index)


@logical.kernel(aggressive_unroll=True, verify=False)
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

    measurements = logical.terminal_measure(reg)
    detectors = []
    observables = []
    for i in range(len(reg)):
        detectors = detectors + set_detector(measurements[i])
        observables = observables + [set_observable(measurements[i], i)]

    return detectors, observables


task = GeminiLogicalSimulator().task(main)

result = task.run(1000, with_noise=True)
result_wo_noise = task.run(1000, with_noise=False)
return_values = result.return_values
detectors = np.asarray(result.detectors)
observables = np.asarray(result.observables)

flips = BpLsdDecoder(result.detector_error_model).decode(detectors)

print(
    "average error without decoding:",
    np.mean(observables ^ result_wo_noise.observables, axis=0),
)
print(
    "average error with decoding:",
    np.mean((observables ^ flips) ^ result_wo_noise.observables, axis=0),
)
