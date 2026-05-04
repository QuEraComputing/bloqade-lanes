from typing import Any, Literal, TypeVar

import numpy as np
from bloqade.analysis.fidelity import FidelityAnalysis
from bloqade.types import Qubit
from kirin.dialects import ilist
from kirin.dialects.ilist import IList

from bloqade import squin
from bloqade.gemini.common.dialects import qubit
from bloqade.lanes.arch.gemini.logical import steane7_initialize
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.pipeline import PhysicalPipeline
from bloqade.lanes.transform import MoveToSquinPhysical

kernel = squin.kernel.add(qubit)
kernel.run_pass = squin.kernel.run_pass


LogicalQubit = ilist.IList[Qubit, Literal[7]]


def steane_slot_allocator():
    """Generates a qubit allocator for logical qubits.
    Tries to allocate logical qubits into the architecture in an efficient way to
    make parallelism in the logical gagdets as easily as possible in the move compiler.

    """
    # canonical slot order
    slot_words = ilist.IList([0, 4, 8, 12, 16, 2, 6, 10, 14, 20])

    slots = IList(
        [
            IList([(0, word_id, site_id) for site_id in range(7)])
            for word_id in slot_words
        ]
    )

    @kernel
    def qalloc_slot(
        slot_index: int, theta: float, phi: float, lam: float
    ) -> LogicalQubit:
        def allocate_at(address: tuple[int, int, int]):
            return qubit.new_at(address[0], address[1], address[2])

        addresses = slots[slot_index]

        reg = ilist.map(allocate_at, addresses)

        steane7_initialize(theta, phi, lam, reg)

        return reg

    @kernel
    def qalloc(
        slot_indices: list[int] | IList[int, Any],
        theta: float = 0.0,
        phi: float = 0.0,
        lam: float = 0.0,
    ) -> IList[LogicalQubit, Any]:

        def _inner(slot_index: int):
            return qalloc_slot(slot_index, theta, phi, lam)

        return ilist.map(_inner, slot_indices)

    return qalloc, qalloc_slot


qalloc, qalloc_slot = steane_slot_allocator()

N = TypeVar("N")


@kernel
def flat(
    reg: ilist.IList[LogicalQubit, Any],
) -> ilist.IList[Qubit, Any]:
    """Flatten a logical register into a single list of physical qubits"""

    def _inner(cumulant, ele):
        return cumulant + ele

    return ilist.foldl(_inner, reg, ilist.IList([]))


@kernel
def cx(controls: ilist.IList[LogicalQubit, N], targets: ilist.IList[LogicalQubit, N]):
    """Efficient broadcasted cx gate over steane logical qubits"""
    squin.broadcast.cx(flat(controls), flat(targets))


@kernel
def measure_logical_reg(logical_reg: ilist.IList[LogicalQubit, Any]):
    """Helper function to get around single measurement restriction of kernel.
    first flatten the logical register into physical qubits, then reconstruct
    the groups of physical measurements into groups related to logical qubits.

    """
    # measurements must be flattened, only one measurement is allowed!
    measurements = squin.broadcast.measure(flat(logical_reg))
    logical_groups = []
    for i in range(len(logical_reg)):
        logical_groups = logical_groups + [measurements[7 * i : 7 * i + 7]]

    return logical_groups


@kernel(typeinfer=True)
def main():
    reg = qalloc([0, 1, 2, 3], 0.0, 0.0, 0.0)

    squin.broadcast.h(reg[0])
    cx(reg[:1], reg[1:2])
    cx(reg[:2], reg[2:])

    return measure_logical_reg(reg)


move_mt = PhysicalPipeline(place_opt_type=ASAPPlacePass).emit(main)

noise_model = MoveToSquinPhysical(
    get_arch_spec(),
    noise_model=generate_simple_noise_model(loss=False),
    aggressive_unroll=True,
).emit(move_mt)

fid = FidelityAnalysis(noise_model.dialects)
fid.run(noise_model)
# note min/max is only used when there is control flow
print("Log Fidelity max: ", -sum(np.log(frange.max) for frange in fid.gate_fidelities))
print("Log Fidelity min: ", -sum(np.log(frange.min) for frange in fid.gate_fidelities))
