"""Regression test: physical compilation must preserve terminal-measurement order.

Under ``always_merge`` the CZ and measurement ``StaticPlacement`` blocks merge and
qubit indices are renumbered to block-local positions. The lowered readout must
still pair each returned measurement result with the qubit measured at that
position — that coupling is carried by ``place.EndMeasure.qubits`` and applied in
``place2move.InsertMeasure`` against the canonical (qubit-id-indexed) layout.

This guards the bug where a non-identity measurement order produced a permuted
readout (each logical qubit read the wrong physical patch).
"""

import pytest
from bloqade.pyqrack.device import StackMemorySimulator
from kirin.dialects import ilist

from bloqade import squin
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.heuristics.physical import make_physical_placement_strategy
from bloqade.lanes.pipeline import PhysicalPipeline
from bloqade.lanes.transform import MoveToSquinPhysical

# Distinct per-qubit bits so any misordered readout changes the result vector.
_BITS = (0, 1, 1, 0)


def _build_kernel(perm: tuple[int, int, int, int]):
    """Deterministic |b0 b1 b2 b3> prep, three overlapping CZ layers, then a
    terminal measurement of all four qubits in ``perm`` order."""
    one_bits = [i for i, b in enumerate(_BITS) if b]

    @squin.kernel
    def circuit():
        q = squin.qalloc(4)
        for i in one_bits:
            squin.x(q[i])
        # Overlapping CZ layers force the place pass to merge the CZ and measure
        # StaticPlacement blocks and renumber qubit indices block-locally. CZ on
        # a computational-basis state only adds phase, so outcomes stay
        # deterministic.
        squin.cz(q[0], q[1])
        squin.cz(q[1], q[2])
        squin.cz(q[3], q[2])
        return squin.broadcast.measure(
            ilist.IList([q[perm[0]], q[perm[1]], q[perm[2]], q[perm[3]]])
        )

    return circuit


@pytest.mark.parametrize("return_moves", [False, True])
@pytest.mark.parametrize(
    "perm",
    [(0, 1, 2, 3), (3, 2, 1, 0), (0, 3, 2, 1), (1, 3, 0, 2)],
    ids=["identity", "reversed", "counter_example", "arbitrary"],
)
def test_physical_roundtrip_preserves_measurement_order(
    perm: tuple[int, int, int, int], return_moves: bool
):
    arch = get_arch_spec()
    strategy = make_physical_placement_strategy(
        return_moves=return_moves, search_budget=None, move_solutions_per_layer=100
    )
    move_kernel = PhysicalPipeline(placement_strategy=strategy).emit(
        _build_kernel(perm)
    )
    compiled = MoveToSquinPhysical(arch).emit(move_kernel)

    sim = StackMemorySimulator()
    # Results are MeasurementResultValue (an IntEnum), so they compare equal to
    # the plain-int expected bits element-wise.
    expected = [_BITS[p] for p in perm]
    # Ground truth: the uncompiled kernel yields results in measurement order.
    assert list(sim.run(_build_kernel(perm))) == expected
    # The physical round-trip (squin -> move -> squin) must preserve that order.
    assert list(sim.run(compiled)) == expected
