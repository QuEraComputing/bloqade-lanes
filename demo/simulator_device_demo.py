import math
from collections import Counter

import numpy as np
from bloqade.decoders import BpLsdDecoder
from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.gemini import logical
from bloqade.gemini.logical.stdlib import default_post_processing
from bloqade.lanes import GeminiLogicalSimulator


# helper functions to analyze statistical distribution of logical measurements
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


@logical.kernel(aggressive_unroll=True)
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

    return default_post_processing(reg)


task = GeminiLogicalSimulator().task(main)

# run simulation with and without noise
print("Running simulation with noise...")
future = task.run_async(10000)
print("Running simulation without noise...")
future_wo_noise = task.run_async(10000, with_noise=False)

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
    "KL divergence between noiseless and raw observables:",
    kl_divergence(observables_wo_noise_hist, observables_hist),
)
print(
    "KL divergence between noiseless and decoded observables:",
    kl_divergence(observables_wo_noise_hist, observables_decoded_hist),
)
print(
    "KL divergence between noiseless and post-selected observables:",
    kl_divergence(observables_wo_noise_hist, observables_postselected_hist),
)
