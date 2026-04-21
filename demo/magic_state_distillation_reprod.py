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
# # Gemini-Logical Reproduction of Magic-State Distillation
#
# This notebook reproduces the **encoded distillation** part of [`magic_state_distillation.py`](/Users/jasonhan/Downloads/magic_state_distillation.py), but uses `@gemini.logical.kernel` so that each logical qubit is realized as a `[[7,1,3]]` Steane block by the Gemini logical lowering pipeline.
#
# The goal here is intentionally modest:
# - mirror the logical 5-to-1 MSD circuit and tomography branches,
# - avoid decoders entirely,
# - test whether simple postselection alone can improve over the injected logical magic-state fidelity.
#
# Because Gemini's raw logical-observable convention is not identical to the reference script's convention, the notebook empirically calibrates:
# - the raw factory-acceptance branch in the current observable convention,
# - the sign frame of the distilled output Bloch vector.
#
# This keeps the notebook focused on the Steane-encoded Gemini analogue of the reference implementation.

# %%
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import stim
from bloqade.squin import kernel
from bloqade.tsim import Circuit
from IPython.display import HTML, display

from bloqade import squin

# NOTE: this is kind of a hack because the msd_utils code isn't incorporated into the actual lib code yet.
PROJECT_ROOT_CANDIDATES = [Path.cwd(), Path.cwd().parent]
for candidate in PROJECT_ROOT_CANDIDATES:
    candidate = candidate.resolve()
    if (candidate / "demo" / "msd_utils").exists():
        sys.path.insert(0, str(candidate))
        break
else:
    raise FileNotFoundError("Could not locate repo root containing demo/msd_utils.")

from demo.msd_utils.circuits import (  # noqa: E402
    build_naive_kernel_bundle,
    build_task_map,
    make_noisy_steane7_initializer,
)
from demo.msd_utils.core import (  # noqa: E402
    DEFAULT_BASIS_LABELS,
    DEFAULT_IDEAL_FACTORY_ACCEPTANCE,
    DEFAULT_TARGET_BLOCH,
    infer_distilled_sign_vector,
    infer_factory_target,
    logical_expectation,
    naive_distilled_summary,
    naive_injected_summary,
    run_task,
)

from bloqade.lanes import GeminiLogicalSimulator  # noqa: E402
from bloqade.lanes.steane_defaults import steane7_m2dets, steane7_m2obs  # noqa: E402

# %%
DEFAULT_BASIS_LABELS

# %%
DEFAULT_IDEAL_FACTORY_ACCEPTANCE

# %%
DEFAULT_TARGET_BLOCH

# %%
np.arccos(np.sqrt(1 / 3))

# %%
# This should give the state (1, 1, 1)
THETA = np.arccos(np.sqrt(1 / 3))
PHI = 0.25 * np.pi
LAM = 0.0

# P_PREP = 0.05
FAST_SHOTS = 200_000
POSTERIOR_SAMPLES = 20_000

BASIS_LABELS = DEFAULT_BASIS_LABELS
OUTPUT_QUBIT = 0
ANCILLA_QUBITS = (1, 2, 3, 4)
IDEAL_FACTORY_ACCEPTANCE = DEFAULT_IDEAL_FACTORY_ACCEPTANCE
TARGET_BLOCH = DEFAULT_TARGET_BLOCH

M2DETS_5 = steane7_m2dets(5)
M2OBS_5 = steane7_m2obs(5)
M2DETS_1 = steane7_m2dets(1)
M2OBS_1 = steane7_m2obs(1)


# %%
# M2DETS_5

# %%
# M2DETS_1

# %%
# M2OBS_5

# %%
# M2OBS_1

# %%
OUTPUT_QUBIT

# %%
kernel_bundle = build_naive_kernel_bundle(THETA, PHI, LAM, output_qubit=OUTPUT_QUBIT)
DISTILLED_KERNELS = kernel_bundle.distilled
INJECTED_KERNELS = kernel_bundle.injected


# %%
kernel_bundle

# %%
kernel_bundle.distilled["Z"].print()

# %%
kernel_bundle.injected["Z"].print()

# %%

# %%
# DISTILLED_KERNELS["X"].print()

# %% [markdown]
# ### Clifford Flow Check
#
# Stim flow generators only work for Clifford circuits, so this diagnostic strips off the non-Clifford magic-state preparation and checks the stabilizer flow of the logical distillation block itself. This is a good sanity check that the distilled logical circuit skeleton matches the expected 5-to-1 MSD Clifford.
#


# %%
def make_reprod_clifford_flow_kernel(basis: str = "Z"):
    @kernel
    def flow_kernel():
        reg = squin.qalloc(5)
        squin.broadcast.reset(reg)
        squin.broadcast.sqrt_x([reg[0], reg[1], reg[4]])
        squin.broadcast.cz([reg[0], reg[2]], [reg[1], reg[3]])
        squin.broadcast.sqrt_y([reg[0], reg[3]])
        squin.broadcast.cz([reg[0], reg[3]], [reg[2], reg[4]])
        squin.broadcast.sqrt_x_adj([reg[0]])
        squin.broadcast.cz([reg[0], reg[1]], [reg[4], reg[3]])
        squin.broadcast.sqrt_x_adj(reg)

        if basis == "X":
            squin.h(reg[0])
        elif basis == "Y":
            squin.sqrt_z_adj(reg[0])
            squin.h(reg[0])

        squin.broadcast.measure(reg)

    return flow_kernel


REPROD_LOGICAL_DISTILLATION_FLOWS = {
    basis: stim.Circuit(
        str(Circuit(make_reprod_clifford_flow_kernel(basis)))
    ).flow_generators()
    for basis in BASIS_LABELS
}

for basis in BASIS_LABELS:
    print(f"{basis} basis: {len(REPROD_LOGICAL_DISTILLATION_FLOWS[basis])} flows")
    for flow in REPROD_LOGICAL_DISTILLATION_FLOWS[basis]:
        print(flow)
    print()


# %% [markdown]
# ### adding noise to state init circ

# %%
# from bloqade.lanes.transform import MoveToSquin
# from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube

# %%
sim = GeminiLogicalSimulator()

# %%
sim = GeminiLogicalSimulator()
noisy_steane7_initialize = make_noisy_steane7_initializer(sim)


# %%
noisy_steane7_initialize.print()

# %%
distilled_tasks = build_task_map(
    sim,
    DISTILLED_KERNELS,
    m2dets=M2DETS_5,
    m2obs=M2OBS_5,
    noisy_initializer=noisy_steane7_initialize,
    append_measurements=True,
)
injected_tasks = build_task_map(
    sim,
    INJECTED_KERNELS,
    m2dets=M2DETS_1,
    m2obs=M2OBS_1,
    noisy_initializer=noisy_steane7_initialize,
    append_measurements=True,
)

print("Built tasks:", list(distilled_tasks), list(injected_tasks))


# %%

# %%
distilled_tasks

# %%
# # %matplotlib qt

# %%
distilled_tasks["X"].physical_move_kernel.print()


# %%
def display_circ(diagram):

    # diagram = distilled_tasks["X"].tsim_circuit.diagram(height=500)
    display(HTML(f"""
    <div style="background: white; padding: 12px;">
    {diagram}
    </div>
    """))


# %%

display_circ(distilled_tasks["Y"].noiseless_tsim_circuit.diagram(height=500))

# %%
distilled_tasks["Y"].noiseless_tsim_circuit

# %%
# import tsim

# %%
# noiseless_circ_injnoise = tsim.Circuit("DEPOLARIZE1(0) 0 1 2 3 4 5 6") + distilled_tasks["X"].noiseless_tsim_circuit

# %%
# noiseless_circ_injnoise.append_from_stim_program_text()

# %%
# # HACK: for now, to add noise gate at the beginning, for tsim bug temporary hack.
# distilled_tasks["X"].__dict__["noiseless_tsim_circuit"] = noiseless_circ_injnoise

# %%
distilled_tasks["X"].noiseless_tsim_circuit

# %%
shots = 100000

x_data = run_task(distilled_tasks["X"], shots, with_noise=False, chunk_size=10000)
y_data = run_task(distilled_tasks["Y"], shots, with_noise=False, chunk_size=10000)
z_data = run_task(distilled_tasks["Z"], shots, with_noise=False, chunk_size=10000)

ex = logical_expectation(x_data.observables[:, 0])
ey = logical_expectation(y_data.observables[:, 0])
ez = logical_expectation(z_data.observables[:, 0])

print("Noiseless <X>, <Y>, <Z> =", (ex, ey, ez))

# %%
FACTORY_TARGET = np.array([0, 0, 0, 0])


# %%
def accepted_output_bits(data, factory_target):
    perfect_stabilizers = np.all(data.detectors == 0, axis=1)
    correct_syndrome = np.all(data.observables[:, 1:] == factory_target, axis=1)
    mask = perfect_stabilizers & correct_syndrome
    return {
        "bits": data.observables[mask, 0].astype(np.uint8),
        "rate_total": float(mask.mean()),
        "rate_stabilizers": float(perfect_stabilizers.mean()),
        "rate_syndrome_given_stabilizers": (
            float(mask.sum() / perfect_stabilizers.sum())
            if perfect_stabilizers.sum() > 0
            else float("nan")
        ),
    }


x = accepted_output_bits(x_data, FACTORY_TARGET)
y = accepted_output_bits(y_data, FACTORY_TARGET)
z = accepted_output_bits(z_data, FACTORY_TARGET)

print(
    "Accepted noiseless <X>, <Y>, <Z> =",
    (
        logical_expectation(x["bits"]),
        logical_expectation(y["bits"]),
        logical_expectation(z["bits"]),
    ),
)

print(
    "Total postselection rates =", (x["rate_total"], y["rate_total"], z["rate_total"])
)
print(
    "Perfect stabilizer rates =",
    (x["rate_stabilizers"], y["rate_stabilizers"], z["rate_stabilizers"]),
)
print(
    "Syndrome rates given perfect stabilizers =",
    (
        x["rate_syndrome_given_stabilizers"],
        y["rate_syndrome_given_stabilizers"],
        z["rate_syndrome_given_stabilizers"],
    ),
)

# %% [markdown]
# ### Noiseless Factory-Branch Debugging
#
# This section helps compare the **reported Steane ancilla observable bits** in this notebook against the bare logical MSD branch you may expect from the unencoded 5-qubit circuit. It prints the dominant noiseless ancilla branches for each tomography basis and gives a helper for checking candidate targets such as `1011` and `0000`.
#


# %%
def branch_histogram(data):
    counts = Counter(tuple(int(x) for x in row) for row in data.observables[:, 1:])
    total = len(data.observables)
    return [(pattern, count / total) for pattern, count in counts.most_common()]


def accepted_output_bits_and_fraction(data, factory_target):
    target = np.asarray(factory_target, dtype=np.uint8)
    mask = np.all(data.observables[:, 1:] == target, axis=1)
    return data.observables[mask, 0].astype(np.uint8), float(mask.mean())


def summarize_factory_target(factory_target):
    print(f"\nFactory target {tuple(int(x) for x in factory_target)}")
    for basis, data in [("X", x_data), ("Y", y_data), ("Z", z_data)]:
        bits, frac = accepted_output_bits_and_fraction(data, factory_target)
        exp = logical_expectation(bits) if len(bits) else float("nan")
        print(
            f"  {basis}: accepted_fraction={frac:.5f}, raw_expectation={exp:.6f}, shots={len(bits)}"
        )


for basis, data in [("X", x_data), ("Y", y_data), ("Z", z_data)]:
    print(f"\nTop noiseless ancilla branches for {basis}:")
    for pattern, frac in branch_histogram(data)[:8]:
        print(f"  {pattern}: {frac:.5f}")

for candidate in [
    np.array([1, 0, 1, 1], dtype=np.uint8),
    np.array([0, 0, 0, 0], dtype=np.uint8),
]:
    summarize_factory_target(candidate)


# %%
def all_syndrome_patterns(num_bits=4):
    return [
        tuple((value >> shift) & 1 for shift in range(num_bits - 1, -1, -1))
        for value in range(2**num_bits)
    ]


def syndrome_fractions(data, patterns):
    counts = Counter(
        tuple(int(x) for x in row)
        for row in np.asarray(data.observables[:, 1:], dtype=np.uint8)
    )
    total = len(data.observables)
    return np.array([counts[pattern] / total for pattern in patterns], dtype=np.float64)


patterns = all_syndrome_patterns(4)
labels = ["".join(str(bit) for bit in pattern) for pattern in patterns]
target_label = "".join(str(int(bit)) for bit in FACTORY_TARGET.tolist())

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, constrained_layout=True)

for ax, (basis, data) in zip(axes, [("X", x_data), ("Y", y_data), ("Z", z_data)]):
    fractions = syndrome_fractions(data, patterns)
    colors = ["#D55E00" if label == target_label else "#4C78A8" for label in labels]
    ax.bar(labels, fractions, color=colors, alpha=0.9)
    ax.set_ylabel("Shot fraction")
    ax.set_title(f"Noiseless 4-bit distillation syndrome distribution ({basis} basis)")
    ax.grid(True, axis="y", alpha=0.25)

axes[-1].set_xlabel("Distillation syndrome")
fig.suptitle("Noiseless distillation syndrome histogram", fontsize=14)
plt.show()

print("Highlighted factory target:", target_label)


# %%
injected_tasks

# %%
# injected_tasks["X"].physical_squin_kernel.print()

# %%
diagram = distilled_tasks["X"].tsim_circuit.diagram(height=500)
display(HTML(f"""
<div style="background: white; padding: 12px;">
  {diagram}
</div>
"""))

# %%
# NOTE: infer.. is also kind of a hack.. you can also presupply the target of the distillation factory; this "brute-forces search" it over all possible distillation syndromes
# and tries to find the syndrome that appears "close" to the ideal factory acceptance. Could be useful, but is not tractable for larger distillation circuits
# (should be OK for up to 10 logical qubit circuits, like on Gemini Logical)
FACTORY_TARGET = infer_factory_target(
    distilled_tasks,
    shots=12_000,
    basis_labels=BASIS_LABELS,
    ideal_factory_acceptance=IDEAL_FACTORY_ACCEPTANCE,
)
DISTILLED_SIGN_VECTOR = infer_distilled_sign_vector(
    distilled_tasks,
    FACTORY_TARGET,
    shots=12_000,
    basis_labels=BASIS_LABELS,
    target_bloch=TARGET_BLOCH,
)
INJECTED_SIGN_VECTOR = np.array([1.0, -1.0, 1.0], dtype=np.float64)

print("Using factory target:", tuple(int(x) for x in FACTORY_TARGET.tolist()))


# %%
# FACTORY_TARGET

# %%

# %%
# QUESTION: why is injected perfect stabilizer fid so high?
inj_raw = naive_injected_summary(
    injected_tasks,
    sign_vector=INJECTED_SIGN_VECTOR,
    posterior_samples=POSTERIOR_SAMPLES,
    shots=FAST_SHOTS,
    require_zero_detectors=False,
    basis_labels=BASIS_LABELS,
    target_bloch=TARGET_BLOCH,
)
inj_ps = naive_injected_summary(
    injected_tasks,
    sign_vector=INJECTED_SIGN_VECTOR,
    posterior_samples=POSTERIOR_SAMPLES,
    shots=FAST_SHOTS,
    require_zero_detectors=True,
    basis_labels=BASIS_LABELS,
    target_bloch=TARGET_BLOCH,
)
dist_raw = naive_distilled_summary(
    distilled_tasks,
    FACTORY_TARGET,
    sign_vector=DISTILLED_SIGN_VECTOR,
    posterior_samples=POSTERIOR_SAMPLES,
    shots=FAST_SHOTS,
    require_zero_ancilla_detectors=False,
    basis_labels=BASIS_LABELS,
    target_bloch=TARGET_BLOCH,
)
dist_flagged = naive_distilled_summary(
    distilled_tasks,
    FACTORY_TARGET,
    sign_vector=DISTILLED_SIGN_VECTOR,
    posterior_samples=POSTERIOR_SAMPLES,
    shots=FAST_SHOTS,
    require_zero_ancilla_detectors=True,
    basis_labels=BASIS_LABELS,
    target_bloch=TARGET_BLOCH,
)

for name, summary in [
    ("Injected raw", inj_raw),
    ("Injected perfect-stabilizer", inj_ps),
    ("Distilled raw branch", dist_raw),
    ("Distilled + zero ancilla detectors", dist_flagged),
]:
    print(name)
    print(summary)
    print()


# %%
labels = [
    "Injected",
    "Injected + zero dets",
    "Distilled",
    "Distilled + zero ancilla dets",
]
summaries = [inj_raw, inj_ps, dist_raw, dist_flagged]
medians = [summary["median"] for summary in summaries]
lows = [summary["median"] - summary["low"] for summary in summaries]
highs = [summary["high"] - summary["median"] for summary in summaries]
accepted = [summary["accepted_fraction"] for summary in summaries]

fig, ax = plt.subplots(figsize=(8, 4.5))
x = np.arange(len(labels))
ax.bar(x, medians, color=["#2A9D8F", "#52B788", "#457B9D", "#1D3557"], alpha=0.8)
ax.errorbar(
    x, medians, yerr=[lows, highs], fmt="none", ecolor="black", capsize=4, linewidth=1.2
)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Magic-state fidelity")
ax.set_title("Naive postselection on Gemini logical Steane blocks")
ax.set_ylim(0.5, 1.01)
ax.grid(True, axis="y", alpha=0.25)

for xi, fidelity, frac in zip(x, medians, accepted):
    ax.text(
        xi, fidelity + 0.004, f"acc={frac:.3f}", ha="center", va="bottom", fontsize=9
    )

plt.tight_layout()
plt.show()

# %%
