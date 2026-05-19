# ---
# ruff: noqa: E402
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: kirin-workspace (3.12.13)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # MSD Decoder Workflow API
#
# This notebook shows the compact workflow API for building MSD kernels, compiling
# tasks, training MLD/MLE decoders, sampling evaluation data, evaluating curves,
# and plotting results.

# %%
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT_CANDIDATES = [Path.cwd(), Path.cwd().parent]
for candidate in PROJECT_ROOT_CANDIDATES:
    candidate = candidate.resolve()
    if (candidate / "demo" / "msd_utils").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    raise FileNotFoundError("Could not locate repo root containing demo/msd_utils.")

LOCAL_DECODER_SRC_CANDIDATES = [
    Path.cwd() / ".." / "bloqade-decoders" / "src",
    Path.cwd() / "bloqade-decoders" / "src",
    Path.cwd().parent / "bloqade-decoders" / "src",
    Path.cwd().parent.parent / "bloqade-decoders" / "src",
]
for candidate in LOCAL_DECODER_SRC_CANDIDATES:
    candidate = candidate.resolve()
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

from bloqade.decoders import GurobiDecoder, TableDecoder  # noqa: E402
from demo.msd_utils import (  # noqa: E402
    DecoderCurveOptions,
    MSDDecoderWorkflowConfig,
    build_injected_decoder_bundle,
    build_injected_task_bundle,
    build_mle_decoder_suite,
    build_msd_decoder_bundle,
    build_msd_primitives,
    build_msd_task_bundle,
    evaluate_decoder_curves,
    evaluate_injected_baseline,
    plot_decoder_curves,
    sample_actual_data,
    train_mld_decoder_suite,
)

from bloqade.lanes import GeminiLogicalSimulator  # noqa: E402

# %% [markdown]
# ## User Configuration

# %%
mld_train_shots = 20_000
eval_shots = 20_000

ideal_theta = 0.3041 * math.pi
ideal_phi = 0.25 * math.pi
ideal_lam = 0.0

theta_offset = 0.30
phi_offset = 0.0
lam_offset = 0.0

theta = ideal_theta + theta_offset
phi = ideal_phi + phi_offset
lam = ideal_lam + lam_offset

target_bloch_vector = np.ones(3, dtype=np.float64) / np.sqrt(3.0)
decoder_primitive_set = build_msd_primitives(theta, phi, lam)
valid_factory_targets = np.array([[0, 0, 0, 0]], dtype=np.uint8)

config = MSDDecoderWorkflowConfig(
    mld_train_shots=mld_train_shots,
    eval_shots=eval_shots,
    target_bloch_vector=target_bloch_vector,
    theta=theta,
    phi=phi,
    lam=lam,
    decoder_primitive_set=decoder_primitive_set,
    valid_factory_targets=valid_factory_targets,
    num_logical_qubits=5,
    output_qubit=0,
    # Use prefix-prepare special tasks so MLD table-training data stays on a
    # smaller special path. Actual/ranking/evaluation data uses CliffT below.
    special_kernel_strategy="prefix_prepare",
    # Use CliffT for the noisy non-Clifford actual/evaluation sampling path.
    # The tsim detector sampler can hit a normalization-underflow assertion on
    # these circuits.
    sim_type="clifft",
    binary_precision=4,
    log=True,
)

# %% [markdown]
# ## Kernels And Tasks

# %%
simulator = GeminiLogicalSimulator()

msd_decoder_bundle = build_msd_decoder_bundle(config)
injected_decoder_bundle = build_injected_decoder_bundle(config)

msd_tasks = build_msd_task_bundle(simulator, config, msd_decoder_bundle)
injected_tasks = build_injected_task_bundle(simulator, config, injected_decoder_bundle)

# %% [markdown]
# ## Decoder Training

# %%
mld_decoders = train_mld_decoder_suite(
    msd_tasks,
    config,
    table_decoder_cls=TableDecoder,
)

mle_decoders = build_mle_decoder_suite(
    msd_tasks,
    gurobi_decoder_cls=GurobiDecoder,
    log=config.log,
)

# %% [markdown]
# ## Sampling, Curves, And Plot

# %%
actual_data = sample_actual_data(msd_tasks, config)

curves = evaluate_decoder_curves(
    actual_data,
    {
        "MLD": mld_decoders,
        "MLE": mle_decoders,
    },
    config,
    curve_options=DecoderCurveOptions(
        threshold_points=24,
        threshold_policy="quantile",
        selection_mode="threshold",
    ),
)

injected_summary = evaluate_injected_baseline(
    injected_tasks,
    config,
    table_decoder_cls=TableDecoder,
    raw=False,
)

fig, ax = plot_decoder_curves(
    curves,
    injected_summary=injected_summary,
    title="MSD Decoder Postselection Curves",
)
fig
