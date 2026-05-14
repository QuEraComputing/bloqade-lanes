from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from kirin.dialects import ilist

from bloqade import squin
from bloqade.gemini.logical.stdlib import default_post_processing

from ..domain.kernels import (
    DecoderPrimitiveSet,
    _build_tomography_primitives,
    _kernels_by_tomography_basis,
    _squin_return_none,
    produce_tomography_kernels,
)
from ..domain.special_tasks import _attach_special_circuit_kernel
from ..standard.types import KirinKernel


@dataclass(frozen=True)
class DecoderKernelBundle:
    """Actual and special tomography kernels for decoder workflows.

    Attributes:
        actual: Basis-labeled kernels for the full noisy/input-prepared logical
            circuit.
        special: Basis-labeled kernels used for special/reference task
            construction.
    """

    actual: dict[str, KirinKernel]
    special: dict[str, KirinKernel]


# NOTE: this is basically what the user would "instantiate" for this specific
# MSD experiment
def build_msd_primitives(
    theta: float,
    phi: float,
    lam: float,
) -> DecoderPrimitiveSet:
    """Build the state-preparation and logical-circuit primitives for MSD.

    Args:
        theta: U3 ``theta`` angle for input magic-state preparation.
        phi: U3 ``phi`` angle for input magic-state preparation.
        lam: U3 ``lambda`` angle for input magic-state preparation.

    Returns:
        Primitive Squin kernels for the MSD state injection and logical circuit.
    """

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

    return DecoderPrimitiveSet(
        state_injection_circuit=msd_magic_prep,
        logical_circuit=msd_forward,
    )


def build_decoder_kernel_bundle(
    primitive_set: DecoderPrimitiveSet,
    # TODO: get rid of logical qubits argument here?
    num_logical_qubits: int = 5,
    output_qubit: int = 0,
    # TODO: have to pass down special_kernel_strategy here?
    special_kernel_strategy: Literal[
        "prefix_prepare", "compiled_inverse_prefix"
    ] = "prefix_prepare",
) -> DecoderKernelBundle:
    """Build tomography kernels for actual and special MSD tasks.

    Args:
        primitive_set: Primitive state-injection and logical-circuit kernels.
        num_logical_qubits: Number of logical qubits in the MSD logical circuit.
        output_qubit: Logical qubit measured as the output state.
        special_kernel_strategy: Strategy expected for the special task path.

    Returns:
        A ``DecoderKernelBundle`` containing basis-labeled kernel maps.

    Raises:
        ValueError: If ``special_kernel_strategy`` is unsupported.
    """

    if special_kernel_strategy not in {"prefix_prepare", "compiled_inverse_prefix"}:
        raise ValueError(
            "special_kernel_strategy must be 'prefix_prepare' or "
            "'compiled_inverse_prefix'."
        )

    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)
    state_injection_circuit = primitive_set.state_injection_circuit
    logical_circuit = primitive_set.logical_circuit

    @squin.kernel
    def actual_logical_kernel(reg):
        state_injection_circuit(reg)
        logical_circuit(reg)

    @squin.kernel
    def special_logical_kernel(reg):
        logical_circuit(reg)

    actual_kernels = produce_tomography_kernels(
        num_logical_qubits,
        actual_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "msd_actual",
    )
    special_task_kernels = produce_tomography_kernels(
        num_logical_qubits,
        special_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "msd_special",
    )
    special_circuit_sources = _kernels_by_tomography_basis(
        produce_tomography_kernels(
            num_logical_qubits,
            special_logical_kernel,
            tomography_primitives,
            _squin_return_none,
            "msd_special_circuit",
            supply_reg=False,
        )
    )

    actual = _kernels_by_tomography_basis(actual_kernels)
    if special_kernel_strategy == "prefix_prepare":
        # TODO: think about a cleaner way to pass down this information? Do I have to pass down this "special_kernel_{x, y, z}"?
        special = {
            basis: _attach_special_circuit_kernel(
                kernel,
                special_circuit_sources[basis],
                num_qubits=num_logical_qubits,
            )
            for basis, kernel in _kernels_by_tomography_basis(
                special_task_kernels
            ).items()
        }
    else:
        special = dict(actual)

    return DecoderKernelBundle(
        actual=actual,
        special=special,
    )


def build_injected_kernel_bundle(
    theta: float,
    phi: float,
    lam: float,
    *,
    output_qubit: int = 0,
) -> DecoderKernelBundle:
    """Build injected-state evaluation and decoder-reference kernels.

    Args:
        theta: U3 ``theta`` angle for the injected input state.
        phi: U3 ``phi`` angle for the injected input state.
        lam: U3 ``lambda`` angle for the injected input state.
        output_qubit: Logical qubit measured by the tomography kernels.

    Returns:
        A bundle whose ``actual`` kernels prepare the requested injected state
        and whose ``special`` kernels prepare ideal X/Y/Z decoder references.
    """

    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)

    @squin.kernel
    def injected_logical_kernel(reg):
        squin.u3(theta, phi, lam, reg[0])

    injected_kernels = produce_tomography_kernels(
        1,
        injected_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "injected",
    )

    return DecoderKernelBundle(
        actual=_kernels_by_tomography_basis(injected_kernels),
        special=build_injected_decoder_kernel_map(output_qubit=output_qubit),
    )


def build_injected_decoder_kernel_map(
    *,
    output_qubit: int = 0,
) -> dict[str, KirinKernel]:
    """Build ideal injected-state tomography kernels for decoder calibration.

    Args:
        output_qubit: Logical qubit measured by the tomography kernels.

    Returns:
        Basis-labeled kernels for ideal X/Y/Z injected reference states.
    """

    h_theta = 0.5 * math.pi
    h_phi = 0.0
    hs_theta = 0.5 * math.pi
    hs_phi = -0.5 * math.pi
    lam = 0.0
    tomography_primitives = _build_tomography_primitives(output_qubit=output_qubit)

    @squin.kernel
    def injected_decoder_x_logical_kernel(reg):
        squin.u3(h_theta, h_phi, lam, reg[0])

    @squin.kernel
    def injected_decoder_y_logical_kernel(reg):
        squin.u3(hs_theta, hs_phi, lam, reg[0])

    @squin.kernel
    def injected_decoder_z_logical_kernel(reg):
        squin.u3(0.0, 0.0, 0.0, reg[0])

    return {
        **_kernels_by_tomography_basis(
            produce_tomography_kernels(
                1,
                injected_decoder_x_logical_kernel,
                {"tomography_x": tomography_primitives["tomography_x"]},
                default_post_processing,
                "injected_decoder",
            )
        ),
        **_kernels_by_tomography_basis(
            produce_tomography_kernels(
                1,
                injected_decoder_y_logical_kernel,
                {"tomography_y": tomography_primitives["tomography_y"]},
                default_post_processing,
                "injected_decoder",
            )
        ),
        **_kernels_by_tomography_basis(
            produce_tomography_kernels(
                1,
                injected_decoder_z_logical_kernel,
                {"tomography_z": tomography_primitives["tomography_z"]},
                default_post_processing,
                "injected_decoder",
            )
        ),
    }
