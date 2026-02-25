import math
from collections import Counter
from typing import Any

import numpy as np
from bloqade.decoders import BpLsdDecoder
from bloqade.decoders.dialects.annotate.types import Detector, Observable
from bloqade.gemini import logical
from kirin.dialects import ilist

from bloqade import qubit, squin, types
from bloqade.lanes.device import GeminiLogicalSimulator


# helper functions to analyze statical distribution of logical measurements
def get_hist(obs_array: np.ndarray):
    return Counter(map(lambda x: tuple(map(int, x)), obs_array[:]))


def kl_divergence(p_hist: Counter, q_hist: Counter) -> float:
    """Compute the KL divergence D_KL(P || Q) between two histograms."""
    total_p = sum(p_hist.values())
    total_q = sum(q_hist.values())
    if total_p == 0 or total_q == 0:
        return float("inf")  # Infinite divergence if one distribution is empty
    divergence = 0.0
    for key in p_hist:
        p_prob = p_hist[key] / total_p
        q_prob = q_hist.get(key, 0) / total_q
        if q_prob > 0:
            divergence += p_prob * math.log(p_prob / q_prob)
        else:
            divergence += p_prob * math.log(p_prob / (1e-10))  # Avoid log(0)
    return divergence


@logical.kernel(aggressive_unroll=True, verify=False)
def set_detector(meas: ilist.IList[types.MeasurementResult, Any]) -> list[Detector]:
    """
    Define default detectors for the Steane code.
    """
    return [
        squin.set_detector([meas[0], meas[1], meas[2], meas[3]], coordinates=[0, 0]),
        squin.set_detector([meas[1], meas[2], meas[4], meas[5]], coordinates=[0, 1]),
        squin.set_detector([meas[2], meas[3], meas[4], meas[6]], coordinates=[0, 2]),
    ]


@logical.kernel(aggressive_unroll=True, verify=False)
def set_observable(
    meas: ilist.IList[types.MeasurementResult, Any], index: int
) -> Observable:
    """
    Define default observables for the Steane code.
    """
    return squin.set_observable([meas[0], meas[1], meas[5]], index)


@logical.kernel(aggressive_unroll=True, verify=False)
def default_observe(
    reg: ilist.IList[qubit.Qubit, Any],
) -> tuple[list[Detector], list[Observable]]:
    """
    A default observe utility function that sets a default set of detectors and observables
    """
    measurements = logical.terminal_measure(reg)
    detectors = []
    observables = []
    for i in range(len(reg)):
        detectors = detectors + set_detector(measurements[i])
        observables = observables + [set_observable(measurements[i], i)]
    return detectors, observables


@logical.kernel(aggressive_unroll=True, verify=False)
def main() -> tuple[list[Detector], list[Observable]]:
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

    return default_observe(reg)


task = GeminiLogicalSimulator().task(main)

# run simulation with and without noise
print("Running simulation with noise...")
future = task.run_async(100000)
print("Running simulation without noise...")
future_wo_noise = task.run_async(100000, with_noise=False)

result_wo_noise = future_wo_noise.result()
result = future.result()

# extract detectors and observables
detectors = np.asarray(result.detectors)
observables = np.asarray(result.observables)
observables_without_noise = np.asarray(result_wo_noise.observables)

# Decode the detectors to get the flips
flips = BpLsdDecoder(task.detector_error_model).decode(detectors)

# post-select on no detection events
post_selection = np.all(detectors == 0, axis=1)
observables_postselected = observables[post_selection, :]

# get the histograms of the observables, decoded observables, observables without noise, and post-selected observables
observables_hist = get_hist(observables)
observables_decoded_hist = get_hist(observables ^ flips)
observables_postselected_hist = get_hist(observables_postselected)
observables_wo_noise_hist = get_hist(observables_without_noise)

# compute and print the KL divergence between the histograms
print(
    "KL divergence between noisy and raw observables:",
    kl_divergence(observables_wo_noise_hist, observables_hist),
)
print(
    "KL divergence between noisy and decoded observables:",
    kl_divergence(observables_wo_noise_hist, observables_decoded_hist),
)
print(
    "KL divergence between noisy and post-selected observables:",
    kl_divergence(observables_wo_noise_hist, observables_postselected_hist),
)
