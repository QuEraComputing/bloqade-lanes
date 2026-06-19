#!/usr/bin/env python
# coding: utf-8

# # MSD Postselection Experiment
#
# This notebook wires together the experimental `PostSelectionExperiment`
# scaffold for three cases:
#
# - distilled MSD with an MLD table decoder,
# - distilled MSD with an MLE decoder,
# - injected-state tomography with a degenerate no-ancilla MLD table decoder.
#
# The default shot counts are intentionally small so the notebook can run as a
# smoke test. Increase the constants below to approach the paper-scale curves.
#
#

# In[1]:


from __future__ import annotations

import numpy as np

from bloqade.gemini.decoding import (
    DEFAULT_TARGET_BLOCH,
    GurobiDecoderWithConfidence,
    PostSelectionExperiment,
    TableDecoderWithConfidence,
    empty_logical_circuit,
    magic_state_dist_steane,
    plot_decoder_curves,
    single_qubit_state_tomography,
)
from bloqade.lanes import GeminiLogicalSimulator

# ## Configuration
#
#

# In[2]:


EVAL_SHOTS = 1_000_000
MLD_TRAIN_SHOTS = 10_000_000
MLD_BATCH_SIZE = None
SIM_TYPE = "clifft"
RANDOM_SEED = 10

MSD_VALID_FACTORY_TARGETS = np.array([[1, 0, 1, 1]], dtype=np.uint8)
INJECTED_VALID_FACTORY_TARGETS = np.zeros((1, 0), dtype=np.uint8)


# ## Shared kernels
#
#

# In[3]:


primitive_set = magic_state_dist_steane(theta_offset=0.30)
noncliff_prefix = primitive_set.state_injection_circuit
main_cliff_circ = primitive_set.logical_circuit
tomo_circs = single_qubit_state_tomography()


# ## Experiment construction
#
#

# In[4]:


msd_mld_exp = PostSelectionExperiment(
    noncliff_prefix,
    main_cliff_circ,
    MSD_VALID_FACTORY_TARGETS,
    TableDecoderWithConfidence,
    tomo_circs,
    {
        "num_shots": MLD_TRAIN_SHOTS,
        "step_size": MLD_BATCH_SIZE,
        "seed": RANDOM_SEED,
    },
)
msd_mle_exp = PostSelectionExperiment(
    noncliff_prefix,
    main_cliff_circ,
    MSD_VALID_FACTORY_TARGETS,
    GurobiDecoderWithConfidence,
    tomo_circs,
)
injected_mld_exp = PostSelectionExperiment(
    noncliff_prefix,
    empty_logical_circuit(),
    INJECTED_VALID_FACTORY_TARGETS,
    TableDecoderWithConfidence,
    tomo_circs,
    {
        "num_shots": MLD_TRAIN_SHOTS,
        "step_size": MLD_BATCH_SIZE,
        "seed": RANDOM_SEED,
    },
)


# ## End-to-end runner
#
#

# In[ ]:


msd_mld_exp.kernels(num_logical_qubits=5)
msd_mld_exp.dem_circuits()
msd_mld_exp.dems()
msd_mld_exp.initialize_decoders()
msd_mld_exp.make_tasks(
    device=GeminiLogicalSimulator(backend=SIM_TYPE, seed=RANDOM_SEED)
)
msd_mld_exp.get_samples(num_shots=EVAL_SHOTS)
msd_mld_exp.decode_and_postselect(decoder_name="MLD")

try:
    msd_mle_exp.kernels(num_logical_qubits=5)
    msd_mle_exp.dem_circuits()
    msd_mle_exp.dems()
    msd_mle_exp.initialize_decoders()
    msd_mle_exp.make_tasks(
        device=GeminiLogicalSimulator(backend=SIM_TYPE, seed=RANDOM_SEED)
    )
    msd_mle_exp.get_samples(num_shots=EVAL_SHOTS)
    msd_mle_exp.decode_and_postselect(decoder_name="MLE")
except Exception as exc:
    print(f"Skipping MLE experiment because decoder construction failed: {exc!r}")
    msd_mle_exp = None

injected_mld_exp.kernels(num_logical_qubits=1)
injected_mld_exp.dem_circuits()
injected_mld_exp.dems()
injected_mld_exp.initialize_decoders()
injected_mld_exp.make_tasks(
    device=GeminiLogicalSimulator(backend=SIM_TYPE, seed=RANDOM_SEED)
)
injected_mld_exp.get_samples(num_shots=EVAL_SHOTS)
injected_mld_exp.decode_and_postselect(decoder_name="Injected MLD")


# ## Tomography result API
#
#

# In[ ]:


tomo_result = msd_mld_exp.tomography_result(
    0.05,
)
tomo_result.fidelity_bloch(DEFAULT_TARGET_BLOCH)


# ## Curves
#
#

# In[ ]:


msd_mld_curve = msd_mld_exp.analysis_f_vs_fraction(
    target_bloch=DEFAULT_TARGET_BLOCH,
)

msd_mle_curve = None
if msd_mle_exp is not None:
    msd_mle_curve = msd_mle_exp.analysis_f_vs_fraction(
        target_bloch=DEFAULT_TARGET_BLOCH,
    )

injected_curve = injected_mld_exp.analysis_f_vs_fraction(
    target_bloch=DEFAULT_TARGET_BLOCH,
)

injected_summary = injected_mld_exp.tomography_result(
    1.0,
).fidelity_bloch(DEFAULT_TARGET_BLOCH)


# ## Individual visualizations
#
#

# In[ ]:


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


# ## Combined figure
#
#

# In[ ]:


curves = {"Distilled (MLD)": msd_mld_curve}
if msd_mle_curve is not None:
    curves = {"Distilled (MLE)": msd_mle_curve, **curves}

fig, ax = plot_decoder_curves(
    curves,
    injected_summary=injected_summary,
    min_accepted_fraction=0.04,
    title="PostselectionExperiment MSD workflow",
)
ax.set_xscale("linear")
ax.set_xlabel("Total accepted fraction")
ax.set_ylabel("Magic state fidelity")
ax.legend()
