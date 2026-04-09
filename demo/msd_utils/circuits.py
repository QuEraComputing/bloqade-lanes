from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from kirin.dialects import ilist

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical
from bloqade.gemini.device import GeminiLogicalSimulatorTask
from bloqade.gemini.logical.stdlib import default_post_processing
from bloqade.lanes import GeminiLogicalSimulator
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.logical_mvp import (
    append_measurements_and_annotations,
    compile_squin_to_move,
)
from bloqade.lanes.transform import MoveToSquin


@dataclass(frozen=True)
class NaiveKernelBundle:
    distilled: dict[str, Any]
    injected: dict[str, Any]


@dataclass(frozen=True)
class DecoderKernelBundle:
    actual: dict[str, Any]
    special: dict[str, Any]
    injected: dict[str, Any]


def _build_primitives(theta: float, phi: float, lam: float, *, output_qubit: int):
    @squin.kernel
    def msd_magic_prep(reg):
        squin.broadcast.u3(theta, phi, lam, reg)

    @squin.kernel
    def msd_forward(reg):
        squin.broadcast.sqrt_x(ilist.IList([reg[0], reg[1], reg[4]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_y(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.sqrt_x_adj(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.broadcast.sqrt_x_adj(reg)

    @squin.kernel
    def msd_inverse(reg):
        squin.broadcast.sqrt_x(reg)
        squin.broadcast.cz(ilist.IList([reg[0], reg[1]]), ilist.IList([reg[4], reg[3]]))
        squin.sqrt_x(reg[0])
        squin.broadcast.cz(ilist.IList([reg[0], reg[3]]), ilist.IList([reg[2], reg[4]]))
        squin.broadcast.sqrt_y_adj(ilist.IList([reg[0], reg[3]]))
        squin.broadcast.cz(ilist.IList([reg[0], reg[2]]), ilist.IList([reg[1], reg[3]]))
        squin.broadcast.sqrt_x_adj(ilist.IList([reg[0], reg[1], reg[4]]))

    @squin.kernel
    def tomography_x(reg):
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_y(reg):
        squin.sqrt_z_adj(reg[output_qubit])
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_z(reg):
        return

    @squin.kernel
    def tomography_x_inv(reg):
        squin.h(reg[output_qubit])

    @squin.kernel
    def tomography_y_inv(reg):
        squin.h(reg[output_qubit])
        squin.sqrt_z(reg[output_qubit])

    @squin.kernel
    def tomography_z_inv(reg):
        return

    return {
        "msd_magic_prep": msd_magic_prep,
        "msd_forward": msd_forward,
        "msd_inverse": msd_inverse,
        "tomography_x": tomography_x,
        "tomography_y": tomography_y,
        "tomography_z": tomography_z,
        "tomography_x_inv": tomography_x_inv,
        "tomography_y_inv": tomography_y_inv,
        "tomography_z_inv": tomography_z_inv,
    }


def build_naive_kernel_bundle(
    theta: float,
    phi: float,
    lam: float,
    *,
    output_qubit: int = 0,
) -> NaiveKernelBundle:
    primitives = _build_primitives(theta, phi, lam, output_qubit=output_qubit)
    msd_magic_prep = primitives["msd_magic_prep"]
    msd_forward = primitives["msd_forward"]
    tomography_x = primitives["tomography_x"]
    tomography_y = primitives["tomography_y"]
    tomography_z = primitives["tomography_z"]

    @gemini_logical.kernel(aggressive_unroll=True)
    def distilled_x():
        reg = qubit.qalloc(5)
        msd_magic_prep(reg)
        msd_forward(reg)
        tomography_x(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def distilled_y():
        reg = qubit.qalloc(5)
        msd_magic_prep(reg)
        msd_forward(reg)
        tomography_y(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def distilled_z():
        reg = qubit.qalloc(5)
        msd_magic_prep(reg)
        msd_forward(reg)
        tomography_z(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_x():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_x(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_y():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_y(reg)
        return

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_z():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_z(reg)
        return

    return NaiveKernelBundle(
        distilled={"X": distilled_x, "Y": distilled_y, "Z": distilled_z},
        injected={"X": injected_x, "Y": injected_y, "Z": injected_z},
    )


def build_decoder_kernel_bundle(
    theta: float,
    phi: float,
    lam: float,
    *,
    output_qubit: int = 0,
) -> DecoderKernelBundle:
    primitives = _build_primitives(theta, phi, lam, output_qubit=output_qubit)
    msd_magic_prep = primitives["msd_magic_prep"]
    msd_forward = primitives["msd_forward"]
    msd_inverse = primitives["msd_inverse"]
    tomography_x = primitives["tomography_x"]
    tomography_y = primitives["tomography_y"]
    tomography_z = primitives["tomography_z"]
    tomography_x_inv = primitives["tomography_x_inv"]
    tomography_y_inv = primitives["tomography_y_inv"]
    tomography_z_inv = primitives["tomography_z_inv"]

    @squin.kernel
    def prepare_special_x(reg):
        tomography_x_inv(reg)
        msd_inverse(reg)

    @squin.kernel
    def prepare_special_y(reg):
        tomography_y_inv(reg)
        msd_inverse(reg)

    @squin.kernel
    def prepare_special_z(reg):
        tomography_z_inv(reg)
        msd_inverse(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_actual_x():
        reg = qubit.qalloc(5)
        msd_magic_prep(reg)
        msd_forward(reg)
        tomography_x(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_actual_y():
        reg = qubit.qalloc(5)
        msd_magic_prep(reg)
        msd_forward(reg)
        tomography_y(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_actual_z():
        reg = qubit.qalloc(5)
        msd_magic_prep(reg)
        msd_forward(reg)
        tomography_z(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_special_x():
        reg = qubit.qalloc(5)
        prepare_special_x(reg)
        msd_forward(reg)
        tomography_x(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_special_y():
        reg = qubit.qalloc(5)
        prepare_special_y(reg)
        msd_forward(reg)
        tomography_y(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def msd_special_z():
        reg = qubit.qalloc(5)
        prepare_special_z(reg)
        msd_forward(reg)
        tomography_z(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_x():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_x(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_y():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_y(reg)
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_z():
        reg = qubit.qalloc(1)
        squin.u3(theta, phi, lam, reg[0])
        tomography_z(reg)
        return default_post_processing(reg)

    return DecoderKernelBundle(
        actual={"X": msd_actual_x, "Y": msd_actual_y, "Z": msd_actual_z},
        special={"X": msd_special_x, "Y": msd_special_y, "Z": msd_special_z},
        injected={"X": injected_x, "Y": injected_y, "Z": injected_z},
    )


def build_injected_decoder_kernel_map() -> dict[str, Any]:
    h_theta = 0.5 * 3.141592653589793
    h_phi = 0.0
    hs_theta = 0.5 * 3.141592653589793
    hs_phi = -0.5 * 3.141592653589793
    lam = 0.0

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_decoder_x():
        reg = qubit.qalloc(1)
        squin.u3(h_theta, h_phi, lam, reg[0])
        squin.h(reg[0])
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_decoder_y():
        reg = qubit.qalloc(1)
        squin.u3(hs_theta, hs_phi, lam, reg[0])
        squin.sqrt_z_adj(reg[0])
        squin.h(reg[0])
        return default_post_processing(reg)

    @gemini_logical.kernel(aggressive_unroll=True)
    def injected_decoder_z():
        reg = qubit.qalloc(1)
        squin.u3(0.0, 0.0, 0.0, reg[0])
        return default_post_processing(reg)

    return {
        "X": injected_decoder_x,
        "Y": injected_decoder_y,
        "Z": injected_decoder_z,
    }


def make_noisy_steane7_initializer(simulator: GeminiLogicalSimulator):
    local_r_noise = simulator.noise_model.local_r_noise
    local_rz_noise = simulator.noise_model.local_rz_noise
    cz_paired_noise = simulator.noise_model.cz_paired_noise
    cz_unpaired_noise = simulator.noise_model.cz_unpaired_noise

    x_axis = 0.0
    y_axis = 0.25
    half_turn = 0.5
    quarter_turn = 0.25

    @squin.kernel
    def noisy_steane7_initialize(theta, phi, lam, qubits):
        evens = qubits[::2]
        odds = qubits[1::2]

        squin.u3(theta, phi, lam, qubits[6])
        local_r_noise(ilist.IList([qubits[6]]), x_axis, half_turn)

        first6 = qubits[:6]
        squin.broadcast.sqrt_y_adj(first6)
        local_r_noise(first6, y_axis, quarter_turn)

        targets1 = evens[1:]
        squin.broadcast.cz(odds, targets1)
        cz_paired_noise(odds, targets1)
        cz_unpaired_noise(ilist.IList([qubits[0]]))

        squin.sqrt_y(qubits[6])
        local_r_noise(ilist.IList([qubits[6]]), y_axis, quarter_turn)

        controls2 = evens[:-1]
        targets2 = ilist.IList([qubits[3], qubits[5], qubits[6]])
        squin.broadcast.cz(controls2, targets2)
        cz_paired_noise(controls2, targets2)
        cz_unpaired_noise(ilist.IList([qubits[1]]))

        tail = qubits[2:]
        squin.broadcast.sqrt_y(tail)
        local_r_noise(tail, y_axis, quarter_turn)

        controls3 = evens[:-1]
        targets3 = odds
        squin.broadcast.cz(controls3, targets3)
        cz_paired_noise(controls3, targets3)
        cz_unpaired_noise(ilist.IList([qubits[6]]))

        subset = ilist.IList([qubits[1], qubits[2], qubits[4]])
        squin.broadcast.sqrt_y(subset)
        local_r_noise(subset, y_axis, quarter_turn)

        squin.x(qubits[3])
        local_r_noise(ilist.IList([qubits[3]]), x_axis, half_turn)

        z_subset = ilist.IList([qubits[1], qubits[5]])
        squin.broadcast.z(z_subset)
        local_rz_noise(z_subset, half_turn)

    return noisy_steane7_initialize


def build_task(
    simulator: GeminiLogicalSimulator,
    kernel: Any,
    *,
    m2dets: Any,
    m2obs: Any,
    noisy_initializer: Any | None = None,
    append_measurements: bool = True,
    physical_hypercube_dims: int = 4,
    transversal_rewrite: bool = True,
) -> GeminiLogicalSimulatorTask:
    logical_kernel = kernel.similar()
    if append_measurements:
        append_measurements_and_annotations(logical_kernel, m2dets, m2obs)

    physical_arch_spec = generate_arch_hypercube(physical_hypercube_dims)
    physical_move_kernel = compile_squin_to_move(
        logical_kernel,
        transversal_rewrite=transversal_rewrite,
    )
    post_processing = atom.AtomInterpreter(
        physical_move_kernel.dialects,
        arch_spec=physical_arch_spec,
    ).get_post_processing(physical_move_kernel)

    task = GeminiLogicalSimulatorTask(
        logical_kernel,
        simulator.noise_model,
        physical_arch_spec,
        physical_move_kernel,
        post_processing,
    )

    if noisy_initializer is not None:
        task.__dict__["physical_squin_kernel"] = MoveToSquin(
            physical_arch_spec,
            logical_initialization=noisy_initializer,
            noise_model=simulator.noise_model,
        ).emit(physical_move_kernel)

    return task


def build_task_map(
    simulator: GeminiLogicalSimulator,
    kernel_map: Mapping[str, Any],
    *,
    m2dets: Any,
    m2obs: Any,
    noisy_initializer: Any | None = None,
    append_measurements: bool = True,
    physical_hypercube_dims: int = 4,
    transversal_rewrite: bool = True,
) -> dict[str, GeminiLogicalSimulatorTask]:
    return {
        basis: build_task(
            simulator,
            kernel,
            m2dets=m2dets,
            m2obs=m2obs,
            noisy_initializer=noisy_initializer,
            append_measurements=append_measurements,
            physical_hypercube_dims=physical_hypercube_dims,
            transversal_rewrite=transversal_rewrite,
        )
        for basis, kernel in kernel_map.items()
    }
