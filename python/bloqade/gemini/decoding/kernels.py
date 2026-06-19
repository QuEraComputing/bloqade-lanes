from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from bloqade import qubit, squin

from .types import KirinKernel, SquinKernel


# TODO: in principle, these tomography kernels aren't gemini-specific.
# BUT, right now, the kernels are written in the gemini logical dialect.
# Can maybe move this code to bloqade-core.
@dataclass(frozen=True)
class _DecoderPrimitiveSet:
    """Primitive Squin kernels needed to build MSD decoder tasks.

    Attributes:
        state_injection_circuit: Squin kernel that prepares the input logical
            magic states.
        logical_circuit: Squin kernel for the logical circuit under test.
    """

    state_injection_circuit: SquinKernel
    logical_circuit: SquinKernel

    # TODO: remove "__getitem__" -- have one standard interface -- can delete this for first PR.
    def __getitem__(self, key: str) -> SquinKernel:
        return getattr(self, key)


def _build_tomography_primitives(*, output_qubit: int) -> dict[str, SquinKernel]:
    """Build X/Y/Z tomography-basis Squin kernels for one output qubit."""

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

    # TODO: stick to X, Y, Z for now. Work with pauli strings by default.
    return {"X": tomography_x, "Y": tomography_y, "Z": tomography_z}


def _produce_tomography_kernels(
    num_qubits: int,
    logical_kernel: KirinKernel,
    tomography_kernels: Mapping[str, SquinKernel],
    return_val_fn: KirinKernel | Callable[[Any], Any],
    kernel_name: str,
    *,
    supply_reg: bool = True,
) -> Mapping[str, KirinKernel]:
    """Compose logical and tomography kernels into labeled tomography kernels.

    Args:
        num_qubits: Number of logical qubits to allocate when ``supply_reg`` is
            true.
        logical_kernel: Kernel applied before the tomography rotation.
        tomography_kernels: Mapping from tomography-kernel name to tomography
            rotation kernel.
        return_val_fn: Callable applied to the logical register to produce the
            returned value for allocated Gemini logical kernels.
        kernel_name: Prefix for generated kernel names.
        supply_reg: If true, create Gemini logical kernels that allocate their
            own register. If false, return Squin kernels that accept ``reg``.

    Returns:
        Mapping from generated kernel names to generated Kirin kernels.
    """

    # TODO: remove the current customization.
    def make_kernel(tomog_kernel: SquinKernel, generated_name: str) -> KirinKernel:
        def inner_tomog_kernel(reg):
            logical_kernel(reg)
            tomog_kernel(reg)

        inner_tomog_kernel.__name__ = generated_name
        inner_tomog_kernel.__qualname__ = generated_name
        inner_kernel = squin.kernel(inner_tomog_kernel)

        if not supply_reg:
            return inner_kernel

        def alloc_kernel():
            reg = qubit.qalloc(num_qubits)
            inner_kernel(reg)
            return return_val_fn(reg)

        alloc_kernel.__name__ = generated_name
        alloc_kernel.__qualname__ = generated_name
        from bloqade.gemini import logical as gemini_logical

        return gemini_logical.kernel(aggressive_unroll=True)(alloc_kernel)

    return {
        tomog_kernel_key: make_kernel(
            tomog_kernel,
            f"{kernel_name}_{tomog_kernel_key.lower()}",
        )
        for tomog_kernel_key, tomog_kernel in tomography_kernels.items()
    }


# This is to give us a dictionary of form {"X": ..., "Y": ..., "Z": ...} for
# downstream consumption.
def _kernels_by_tomography_basis(
    kernels: Mapping[str, KirinKernel],
) -> dict[str, KirinKernel]:
    """Rekey generated tomography kernels by basis label."""

    return {
        kernel_name.split("_")[-1].upper(): kernel
        for kernel_name, kernel in kernels.items()
    }
