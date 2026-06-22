#!/usr/bin/env python
# coding: utf-8
# %% [markdown]
# # MSD Postselection Experiment
#
# This notebook walks through the workflow of using a `PostSelectionExperiment` object in order run a logical magic state distillation experiment in simulation of our Gemini-QEC machine.

# %%
# Import utilities for running MSD Experiment
import numpy as np

from bloqade.gemini.decoding import (
    GurobiDecoderWithConfidence,
    PostSelectionExperiment,
    TableDecoderWithConfidence,
    empty_logical_circuit,
    magic_state_dist_steane,
    single_qubit_state_tomography,
)
from bloqade.gemini.decoding.workflow import _plot_decoder_curves
from bloqade.lanes import GeminiLogicalSimulator

# %% [markdown]
# ## Define Circuits to Run
#
# Here, you can define the circuits that you'd like to execute on the hardware. On the first capabilities for Gemini Logical, we will allow a single layer of nonclifford gates used in the state preparation circuit, followed by a Clifford-only circuit.

# %% [markdown]
# <img src="./star_demo_imgs/gemini_mvp_capabilities.png" width=500>

# %%
# Define the nonclifford prefix and the clifford circuit to apply on the logical qubits
nonclifford_prefix, clifford_circuit = magic_state_dist_steane(theta_offset=0.30)
# Define the tomography circuits to use. These must consist of purely Clifford gates for
# the first release of Gemini.
tomography_circuits = single_qubit_state_tomography()

# %% [markdown]
# ## Define Experiments
#
# We provide a "wizard" class that orchestrates the steps of running an experiment that obtains samples from the hardware and runs decoding with postselection on your ancilla qubits.
#
# This class, named `PostSelectionExperiment`, takes in `nonclifford_prefix`, `clifford_circuit`, and `tomography_circuits` to construct the set of circuits for your experiment in each basis. It also takes in the `decoder` class you'd like to use as well as `decoder_init_args`, which are optional arguments used to initialize your decoder, if desired.

# %% [markdown]
# Here, we define three experiments: one for the lookup table decoder on the distillation circuit, one for a Gurobi Maximum-Likelihood Error decoder on the distillation circuit, and finally, one for the lookup table decoder on the experiment to obtain the injected fidelity of the encoded magic state.

# %%
msd_mld_exp = PostSelectionExperiment(
    nonclifford_prefix,
    clifford_circuit,
    tomography_circuits,
    TableDecoderWithConfidence,
    {
        "seed": 10,
    },
)
msd_mle_exp = PostSelectionExperiment(
    nonclifford_prefix,
    clifford_circuit,
    tomography_circuits,
    GurobiDecoderWithConfidence,
)
injected_mld_exp = PostSelectionExperiment(
    nonclifford_prefix,
    empty_logical_circuit(),
    tomography_circuits,
    TableDecoderWithConfidence,
    {
        "seed": 10,
    },
)

# %% [markdown]
# ## Run the Experiments
# On these "wizard" classes, we then call methods to initialize our decoders, obtain samples from our hardware device (for this demo, we are using a simulator in place of our hardware device), and performing decoding and postselection to analyze our results.

# %%
msd_mld_exp.kernels(num_logical_qubits=5)
msd_mld_exp.dem_circuits()
msd_mld_exp.dems()
msd_mld_exp.initialize_decoders()
msd_mld_exp.make_tasks(device=GeminiLogicalSimulator(backend="clifft", seed=10))
msd_mld_exp.get_samples(num_shots=1_000_000)
msd_mld_exp.decode_and_postselect(
    np.array([[1, 0, 1, 1]], dtype=np.uint8),
    decoder_name="MLD",
)

# %%
# Run the same set of methods for the MLE decoder.
msd_mle_exp.kernels(num_logical_qubits=5)
msd_mle_exp.dem_circuits()
msd_mle_exp.dems()
msd_mle_exp.initialize_decoders()
msd_mle_exp.make_tasks(device=GeminiLogicalSimulator(backend="clifft", seed=10))
msd_mle_exp.get_samples(num_shots=1_000_000)
msd_mle_exp.decode_and_postselect(
    np.array([[1, 0, 1, 1]], dtype=np.uint8),
    decoder_name="MLE",
)

# %%
# Run the same set of methods for the injected experiment with no distillation circuit and
# the lookup table decoder.
injected_mld_exp.kernels(num_logical_qubits=1)
injected_mld_exp.dem_circuits()
injected_mld_exp.dems()
injected_mld_exp.initialize_decoders()
injected_mld_exp.make_tasks(device=GeminiLogicalSimulator(backend="clifft", seed=10))
injected_mld_exp.get_samples(num_shots=1_000_000)
injected_mld_exp.decode_and_postselect(
    np.array([[]], dtype=np.uint8),
    decoder_name="Injected MLD",
)

# %% [markdown]
# ## Perform Tomography
# After we have performed decoding, we expose an additional API in `PostSelectionExperiment` to obtain the fidelity to a target magic state, with the ability to vary the fraction of accepted shots and see the impact on the fidelity.
# > `accepted_fraction` is the fraction of shots you want to retain *after* the postselection criteria on the distillation circuit has been applied.
#
# > For example, say that you ran 1,000,000 shots for each basis. After decoding and postselection, we may only have 100,000 shots remaining in each basis. If you set `accepted_fraction` = 0.5, then you would accept around 150,000 shots across all three bases (note that the number of shots accepted in each basis may differ).

# %%
# The bloch vector to compute fidelity to (for MSD, we want to compute fidelity to the (1, 1, 1) state.)
TARGET_BLOCH = np.ones(3, dtype=np.float64) / np.sqrt(3.0)

tomo_result = msd_mld_exp.tomography_result(
    0.50,
)
tomo_result.fidelity_bloch(TARGET_BLOCH)

# %% [markdown]
# ## Perform Sliding-Scale Postselection
# After decoding, because each shot has an associated confidence score, we can additionally threshold on those confidence scores to obtain a continuous scale of selected shots.

# %%
msd_mld_curve = msd_mld_exp.analysis_f_vs_fraction(
    target_bloch=TARGET_BLOCH,
)

msd_mle_curve = None
if msd_mle_exp is not None:
    msd_mle_curve = msd_mle_exp.analysis_f_vs_fraction(
        target_bloch=TARGET_BLOCH,
    )

injected_curve = injected_mld_exp.analysis_f_vs_fraction(
    target_bloch=TARGET_BLOCH,
)

injected_summary = injected_mld_exp.tomography_result(
    1.0,
).fidelity_bloch(TARGET_BLOCH)

# %% [markdown]
# ## Visualize Accepted Fraction vs. Fidelity
# We can subsequently visualize the magic state fidelity as a function of the fraction of accepted shots.
# > The fraction of accepted shots here is the fraction of shots out of the total number of shots used in the entire experiment.
#
# > Example: say that we ran 1,000,000 shots in each basis on the hardware device, totalling 3,000,000 shots. In the below plots, an accepted fraction of 0.05 corresponds to accepting a total of 0.05 * 3,000,000 = 150,000 shots.

# %%

fig_mld, ax_mld = msd_mld_exp.analysis_visualization(
    min_accepted_fraction=0.04,
    title="Distilled MSD with MLD postselection",
)

if msd_mle_exp is not None:
    fig_mle, ax_mle = msd_mle_exp.analysis_visualization(
        min_accepted_fraction=0.04,
        title="Distilled MSD with MLE postselection",
    )

fig_injected, ax_injected = injected_mld_exp.analysis_visualization(
    min_accepted_fraction=0.04,
    title="Injected-state baseline",
)

# %% [markdown]
# You can additionally combine the figures into one for easier comparison of the performance across different decoders.

# %%
curves = {"Distilled (MLD)": msd_mld_curve}
if msd_mle_curve is not None:
    curves = {"Distilled (MLE)": msd_mle_curve, **curves}

fig, ax = _plot_decoder_curves(
    curves,
    injected_summary=injected_summary,
    min_accepted_fraction=0.04,
    title="PostselectionExperiment MSD workflow",
)
ax.set_xscale("linear")
ax.set_xlabel("Total accepted fraction")
ax.set_ylabel("Magic state fidelity")
ax.legend()
