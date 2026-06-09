# %% [markdown]
# # Compare Injected Special-Circuit Paths
#
# This notebook compares the injected decoder-reference path used by
# `build_injected_kernel_bundle(...)` against the generic
# `PostSelectionExperiment(..., empty_logical_circuit(), ...)` path with
# `prefix_prepare`.
#
# The goal is to debug whether the two paths train/evaluate the same injected
# decoder-reference circuits.

# %%
from __future__ import annotations

import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from bloqade.lanes import GeminiLogicalSimulator

try:
    REPO_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    REPO_ROOT = Path.cwd()
if REPO_ROOT.name == "demo":
    REPO_ROOT = REPO_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.msd_utils.application.experiments import (
    empty_logical_circuit,
    magic_state_dist_steane,
)

from bloqade.gemini.decoding.kernels import DecoderPrimitiveSet
from bloqade.gemini.decoding.measurement_maps import build_measurement_maps
from bloqade.gemini.decoding.msd import (
    build_decoder_kernel_bundle,
    build_injected_kernel_bundle,
)
from bloqade.gemini.decoding.sampling import _sample_task_raw, run_task
from bloqade.gemini.decoding.special_tasks import (
    apply_special_tsim_circuit_strategy,
    build_task_map,
)

# %% [markdown]
# ## Build Both Paths

# %%
IDEAL_THETA = 0.3041 * math.pi
IDEAL_PHI = 0.25 * math.pi
IDEAL_LAM = 0.0

THETA = IDEAL_THETA + 0.30
PHI = IDEAL_PHI
LAM = IDEAL_LAM

BASIS_LABELS = ("X", "Y", "Z")
SHOTS = 256
SIM_TYPE = "tsim"

simulator = GeminiLogicalSimulator()
m2dets, m2obs = build_measurement_maps(1)

dedicated_kernels = build_injected_kernel_bundle(THETA, PHI, LAM)
dedicated_actual_tasks = build_task_map(
    simulator,
    dedicated_kernels.actual,
    m2dets=m2dets,
    m2obs=m2obs,
    append_measurements=False,
)
dedicated_special_tasks = build_task_map(
    simulator,
    dedicated_kernels._special,
    m2dets=m2dets,
    m2obs=m2obs,
    append_measurements=False,
)

primitive_set = magic_state_dist_steane()
generic_primitive_set = DecoderPrimitiveSet(
    state_injection_circuit=primitive_set.state_injection_circuit,
    logical_circuit=empty_logical_circuit(),
)
generic_kernels = build_decoder_kernel_bundle(
    generic_primitive_set,
    num_logical_qubits=1,
    output_qubit=0,
    special_kernel_strategy="prefix_prepare",
)
generic_actual_tasks = build_task_map(
    simulator,
    generic_kernels.actual,
    m2dets=m2dets,
    m2obs=m2obs,
    append_measurements=False,
)
generic_special_tasks_unmodified = build_task_map(
    simulator,
    generic_kernels._special,
    m2dets=m2dets,
    m2obs=m2obs,
    append_measurements=False,
)
generic_special_tasks = apply_special_tsim_circuit_strategy(
    generic_special_tasks_unmodified,
    "prefix_prepare",
    normalize_observable_reference=True,
)

# %% [markdown]
# ## Helpers


# %%
def circuit_fingerprint(circuit) -> str:
    text = str(circuit)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def dem_fingerprint(dem) -> str:
    text = str(dem)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def unique_rows(array: np.ndarray) -> list[tuple[int, ...]]:
    rows = np.unique(np.asarray(array, dtype=np.uint8), axis=0)
    return [tuple(int(x) for x in row) for row in rows]


def task_summary(task, *, shots: int = SHOTS) -> dict[str, object]:
    raw = _sample_task_raw(
        task,
        shots,
        with_noise=False,
        sim_type=SIM_TYPE,
    )
    normalized = run_task(
        task,
        shots,
        with_noise=False,
        sim_type=SIM_TYPE,
    )
    dem = task.detector_error_model
    circuit_text = str(task.tsim_circuit)
    return {
        "observable_frame": str(getattr(task, "observable_frame", "raw-task")),
        "raw_detectors": unique_rows(raw.detectors),
        "raw_observables": unique_rows(raw.observables),
        "normalized_detectors": unique_rows(normalized.detectors),
        "normalized_observables": unique_rows(normalized.observables),
        "num_detectors": dem.num_detectors,
        "num_observables": dem.num_observables,
        "dem_lines": len(str(dem).splitlines()),
        "dem_hash": dem_fingerprint(dem),
        "circuit_hash": circuit_fingerprint(task.tsim_circuit),
        "circuit_lines": len(circuit_text.splitlines()),
    }


def print_pair_summary(
    label: str,
    left_name: str,
    left_task,
    right_name: str,
    right_task,
) -> None:
    left = task_summary(left_task)
    right = task_summary(right_task)
    print(f"\n=== {label} ===")
    print(left_name, left)
    print(right_name, right)
    print(
        "same circuit text:",
        str(left_task.tsim_circuit) == str(right_task.tsim_circuit),
    )
    print(
        "same DEM text:",
        str(left_task.detector_error_model) == str(right_task.detector_error_model),
    )


# %% [markdown]
# ## Actual Injected Tasks
#
# These should be very close: dedicated injected actual kernels use
# `squin.u3(..., reg[0])`, while the generic path uses the one-qubit
# broadcasted MSD state-injection primitive plus an empty logical circuit.

# %%
for basis in BASIS_LABELS:
    print_pair_summary(
        basis,
        "dedicated actual",
        dedicated_actual_tasks[basis],
        "generic actual",
        generic_actual_tasks[basis],
    )

# %% [markdown]
# ## Special / Decoder-Reference Tasks
#
# These are the important comparison:
#
# - dedicated injected special tasks come from `build_injected_decoder_kernel_map()`
# - generic special tasks come from `empty_logical_circuit() + tomography` with
#   `prefix_prepare`

# %%
for basis in BASIS_LABELS:
    print_pair_summary(
        basis,
        "dedicated special",
        dedicated_special_tasks[basis],
        "generic prefix special",
        generic_special_tasks[basis],
    )

# %% [markdown]
# ## Show Circuit Text For One Basis
#
# Change `BASIS_TO_SHOW` if needed.

# %%
BASIS_TO_SHOW = "Y"

print("Dedicated special circuit")
print(dedicated_special_tasks[BASIS_TO_SHOW].tsim_circuit)

print("\nGeneric prefix special circuit")
print(generic_special_tasks[BASIS_TO_SHOW].tsim_circuit)

# %% [markdown]
# ## Show DEM Text For One Basis

# %%
print("Dedicated special DEM")
print(dedicated_special_tasks[BASIS_TO_SHOW].detector_error_model)

print("\nGeneric prefix special DEM")
print(generic_special_tasks[BASIS_TO_SHOW].detector_error_model)
