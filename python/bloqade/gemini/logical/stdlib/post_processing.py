from typing import Any, Literal

from bloqade.decoders.dialects.annotate.types import Detector, Observable
from kirin.dialects import ilist

from bloqade import squin, types

from ..dialects import operations
from ..group import kernel


@kernel(aggressive_unroll=True, verify=False)
def set_detector(
    meas: ilist.IList[types.MeasurementResult, Any],
) -> ilist.IList[Detector, Literal[3]]:
    return ilist.IList(
        [
            squin.set_detector(
                [meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0]
            ),
            squin.set_detector(
                [meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1]
            ),
            squin.set_detector(
                [meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2]
            ),
        ]
    )


@kernel(aggressive_unroll=True, verify=False)
def set_observable(meas: ilist.IList[types.MeasurementResult, Any]) -> Observable:
    """Helper kernel to add observable for the Steane code."""
    return squin.set_observable([meas[0], meas[1], meas[5]])


@kernel(aggressive_unroll=True, verify=False)
def default_post_processing(
    register: ilist.IList[types.Qubit, Any],
) -> tuple[ilist.IList[Detector, Any], ilist.IList[Observable, Any]]:
    """
    Helper kernel to measure the input qubits, and additionally adds statements annotating the kernel with
    detector and observables for the Steane code.

    Args:
        register (ilist.IList[types.Qubit, Any]): The list of qubits to measure. These should be logical qubits
        in your program.

    Returns:
        tuple[ilist.IList[Detector, Any], ilist.IList[Observable, Any]]: A tuple of two IList's, where the first
        IList is a flattened list of detectors and the second is a flattened list of observables.
    """
    measurements = operations.terminal_measure(register)

    detectors = set_detector(measurements[0])
    observables = ilist.IList([set_observable(measurements[0])])
    for i in range(1, len(register)):
        detectors = detectors + set_detector(measurements[i])
        observables = observables + [set_observable(measurements[i])]

    return detectors, observables
