from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical

from ..standard.types import KirinKernel, SquinKernel


@dataclass(frozen=True)
class DecoderPrimitiveSet:
    state_injection_circuit: SquinKernel
    logical_circuit: SquinKernel

    def __getitem__(self, key: str) -> SquinKernel:
        return getattr(self, key)


def _build_tomography_primitives(*, output_qubit: int) -> dict[str, SquinKernel]:
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

    return {
        "tomography_x": tomography_x,
        "tomography_y": tomography_y,
        "tomography_z": tomography_z,
    }


def _squin_return_none(reg):
    return


def produce_tomography_kernels(
    num_qubits: int,
    logical_kernel: KirinKernel,
    tomography_kernels: Mapping[str, SquinKernel],
    return_val_fn: KirinKernel | Callable[[Any], Any],
    kernel_name: str,
    *,
    supply_reg: bool = True,
) -> Mapping[str, KirinKernel]:
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
        return gemini_logical.kernel(aggressive_unroll=True)(alloc_kernel)

    return {
        f"{kernel_name}_{tomog_kernel_key.split('_')[-1]}": make_kernel(
            tomog_kernel,
            f"{kernel_name}_{tomog_kernel_key.split('_')[-1]}",
        )
        for tomog_kernel_key, tomog_kernel in tomography_kernels.items()
    }


# This is to give us a dictionary of form {"X": ..., "Y": ..., "Z": ...} for
# downstream consumption.
def _kernels_by_tomography_basis(
    kernels: Mapping[str, KirinKernel],
) -> dict[str, KirinKernel]:
    return {
        kernel_name.split("_")[-1].upper(): kernel
        for kernel_name, kernel in kernels.items()
    }
