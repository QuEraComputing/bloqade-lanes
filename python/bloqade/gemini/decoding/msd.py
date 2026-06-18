from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kirin.dialects import ilist

from bloqade import squin

from .kernels import (
    DecoderPrimitiveSet,
    _build_tomography_primitives,
    _kernels_by_tomography_basis,
    produce_tomography_kernels,
)
from .types import KirinKernel


def _default_post_processing():
    from bloqade.gemini.logical.stdlib import default_post_processing

    return default_post_processing


@dataclass(frozen=True)
class TomographyKernels:
    """Basis-labeled tomography kernels.

    Attributes:
        actual: Basis-labeled kernels for the full noisy/input-prepared logical
            circuit.
    """

    actual: dict[str, KirinKernel]


# NOTE: this is basically what the user would "instantiate" for this specific
# MSD experiment
def build_msd_primitives(
    theta: float,
    phi: float,
    lam: float,
    *,
    log: bool = True,
) -> DecoderPrimitiveSet:
    """Build the state-preparation and logical-circuit primitives for MSD.

    Args:
        theta: U3 ``theta`` angle for input magic-state preparation.
        phi: U3 ``phi`` angle for input magic-state preparation.
        lam: U3 ``lambda`` angle for input magic-state preparation.
        log: If true, print a progress message.

    Returns:
        Primitive Squin kernels for the MSD state injection and logical circuit.
    """

    if log:
        print("Building MSD primitives...")

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
    num_logical_qubits: int = 5,
    tomography_kernels: Mapping[str, KirinKernel] | None = None,
) -> TomographyKernels:
    """Build basis-labeled MSD tomography kernels.

    Args:
        primitive_set: Primitive state-injection and logical-circuit kernels.
        num_logical_qubits: Number of logical qubits in the MSD logical circuit.
        tomography_kernels: Optional tomography operations keyed by basis.

    Returns:
        A ``TomographyKernels`` containing basis-labeled kernel maps.
    """

    tomography_primitives = (
        dict(tomography_kernels)
        if tomography_kernels is not None
        else _build_tomography_primitives(output_qubit=0)
    )
    state_injection_circuit = primitive_set.state_injection_circuit
    logical_circuit = primitive_set.logical_circuit

    @squin.kernel
    def actual_logical_kernel(reg):
        state_injection_circuit(reg)
        logical_circuit(reg)

    default_post_processing = _default_post_processing()
    actual_kernels = produce_tomography_kernels(
        num_logical_qubits,
        actual_logical_kernel,
        tomography_primitives,
        default_post_processing,
        "msd_actual",
    )

    return TomographyKernels(actual=_kernels_by_tomography_basis(actual_kernels))
