# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: bloqade-lanes (3.12.13.final.0)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # STAR Rz Gate Demonstration

# %% [markdown]
# In this notebook, we demonstrate the STAR Rz gate, with k = 1 (to implement the STAR rotation, only single-qubit gates are applied).

# For defining constants and analysis
import math

import numpy as np

from bloqade import squin

# %%
# For defining language and compilation
from bloqade.gemini import GeminiLogicalSimulator, logical as gemini_logical
from bloqade.gemini.logical.stdlib import default_post_processing

# %% [markdown]
# In this demo, we use the injected angle as THETA = pi / 16. Note that this is the logical injected angle. Our compiler will convert the logical injected angle into the physical rotation angle using the following formula:

# %% [markdown]
# $$\theta^{*} = -\,\mathrm{sign}(\theta)\; 2\arctan\!\left(\left|\tan\!\left(\tfrac{\theta}{2}\right)\right|^{1/3}\right)$$

# %%
THETA = math.pi / 16


# %%
@gemini_logical.kernel(aggressive_unroll=True, verify=False)
def star_on_plus_kernel():
    reg = squin.qalloc(1)
    squin.h(reg[0])
    gemini_logical.star_rz(THETA, reg[0])


# %%
task = GeminiLogicalSimulator().task(logical_kernel=star_on_plus_kernel)

# %% [markdown]
# We can visualize the circuit and see that U3 gates have been applied to the last 3 qubits, representing the target support

# %%
task.noiseless_tsim_circuit.diagram()

# %% [markdown]
# We then perform a simple analysis to do tomography in noiseless simulation using tsim to verify the fidelity of our output state. We introduce an ancilla qubit to do X-Steane syndrome extraction.

# %%
SHOTS = 10_000


# Define tomography kernels with Steane syndrome extraction to do postselection
@gemini_logical.kernel(aggressive_unroll=True, verify=False)
def star_on_plus_kernel_x():
    reg = squin.qalloc(2)
    squin.h(reg[0])
    gemini_logical.star_rz(THETA, reg[0])

    squin.cx(reg[1], reg[0])
    squin.h(reg[1])

    squin.h(reg[0])
    return default_post_processing(reg)


@gemini_logical.kernel(aggressive_unroll=True, verify=False)
def star_on_plus_kernel_y():
    reg = squin.qalloc(2)
    squin.h(reg[0])
    gemini_logical.star_rz(THETA, reg[0])

    squin.cx(reg[1], reg[0])
    squin.h(reg[1])

    squin.sqrt_z_adj(reg[0])
    squin.h(reg[0])
    return default_post_processing(reg)


@gemini_logical.kernel(aggressive_unroll=True, verify=False)
def star_on_plus_kernel_z():
    reg = squin.qalloc(2)
    squin.h(reg[0])
    gemini_logical.star_rz(THETA, reg[0])

    squin.cx(reg[1], reg[0])
    squin.h(reg[1])

    return default_post_processing(reg)


# %%
def postselected_observable_bits(result):
    detectors = np.asarray(result.detectors, dtype=bool)
    observables = np.asarray(result.observables, dtype=bool)

    # print(f"detectors.shape: {detectors.shape}")
    # print(f"detectors[:, -3:].shape: {detectors[:, -3:].shape}")
    # print(f"observables.shape: {observables.shape}")
    accepted = ~np.any(detectors[:, -3:], axis=1)
    if not np.any(accepted):
        raise ValueError("no accepted shots after detector postselection")

    return observables[accepted, 0].astype(np.uint8), accepted


# Compile kernels down to tsim circuits to do simulation
x_task = GeminiLogicalSimulator().task(logical_kernel=star_on_plus_kernel_x)
y_task = GeminiLogicalSimulator().task(logical_kernel=star_on_plus_kernel_y)
z_task = GeminiLogicalSimulator().task(logical_kernel=star_on_plus_kernel_z)

x_shots = x_task.run(shots=SHOTS, with_noise=False)
y_shots = y_task.run(shots=SHOTS, with_noise=False)
z_shots = z_task.run(shots=SHOTS, with_noise=False)

# Postselect on ancilla detectors being all 0
x_shots_arr, x_accepted = postselected_observable_bits(x_shots)
y_shots_arr, y_accepted = postselected_observable_bits(y_shots)
z_shots_arr, z_accepted = postselected_observable_bits(z_shots)

print(f"accepted X shots: {len(x_shots_arr)}/{SHOTS} ({np.mean(x_accepted):.6f})")
print(f"accepted Y shots: {len(y_shots_arr)}/{SHOTS} ({np.mean(y_accepted):.6f})")
print(f"accepted Z shots: {len(z_shots_arr)}/{SHOTS} ({np.mean(z_accepted):.6f})")


# %%
x_task.noiseless_tsim_circuit.diagram()


# %%
def fidelity_from_counts(
    x_shots_arr,
    y_shots_arr,
    z_shots_arr,
    binary_precision=None,  # kept for signature compatibility; unused here
    target_bloch=None,
):
    """Estimate state fidelity from X/Y/Z-basis measurement bits.

    Each ``*_shots_arr`` is a 1D array of 0/1 outcomes measured in that basis,
    so the expectation value is <P> = P(0) - P(1) = 1 - 2*mean(bits).
    For a pure target state with Bloch vector t and measured Bloch vector r,
    F = (1 + r . t) / 2.
    """
    ex = 1.0 - 2.0 * np.mean(x_shots_arr)
    ey = 1.0 - 2.0 * np.mean(y_shots_arr)
    ez = 1.0 - 2.0 * np.mean(z_shots_arr)
    bloch = np.array([ex, ey, ez], dtype=float)

    target = np.asarray(target_bloch, dtype=float)
    fidelity = 0.5 * (1.0 + float(bloch @ target))

    return {"point": fidelity, "bloch": (ex, ey, ez)}


# %%
# We then compute the fidelity to the expected bloch vector.
star_bloch_vector = np.array((math.cos(THETA), math.sin(THETA), 0.0))
fidelity_estimate = fidelity_from_counts(
    x_shots_arr,
    y_shots_arr,
    z_shots_arr,
    target_bloch=star_bloch_vector,
)

# %%
star_bloch_vector

# %%
fidelity_estimate
