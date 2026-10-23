"""End-to-end: MergeSingleQubitRuns in the logical pipeline (#1117).

The count checks pin the gain. The exactness check compiles each kernel with the
rule on and off and requires the encoded Clifford tableaux to be equal. Both
sides run ``EliminateRz`` with its residual flushed, for the reason given in
``test_logical_clifford_lowering.py``: a discarded residual is a Z-type
difference, which is exactly what a sign error in the merge would also look
like.
"""

import pytest
import stim
from kirin.rewrite.abc import RewriteRule

from bloqade import qubit, squin
from bloqade.gemini import GeminiLogicalSimulator, logical as gemini_logical
from bloqade.gemini.logical.stdlib import default_post_processing
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import place
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz
from bloqade.lanes.transform import native_to_place
from bloqade.lanes.transform.native_to_place import LogicalNativeToPlace


class _FlushingEliminateRz(EliminateRz):
    def __init__(self, no_raise: bool = False) -> None:
        super().__init__(no_raise=no_raise, flush_residual=True)


class _NoOpMerge(RewriteRule):
    """Stand-in for MergeSingleQubitRuns that leaves every statement alone."""


@gemini_logical.kernel(aggressive_unroll=True)
def _cx_twice():
    """The issue's repro: CX = H . CZ . H twice, so H . H cancels between."""
    q = qubit.qalloc(2)
    squin.h(q[1])
    squin.cz(q[0], q[1])
    squin.h(q[1])
    squin.h(q[1])
    squin.cz(q[0], q[1])
    squin.h(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _basis_change_gadget():
    """S . H ... H . S_adj around a CZ, on a qubit prepared in |+>."""
    q = qubit.qalloc(2)
    squin.h(q[0])
    squin.h(q[1])
    squin.s(q[1])
    squin.h(q[1])
    squin.cz(q[0], q[1])
    squin.h(q[1])
    squin.s_adj(q[1])
    return default_post_processing(q)


@gemini_logical.kernel(aggressive_unroll=True)
def _mixed_runs():
    """Runs of every shape on three wires, including Paulis and sqrt gates."""
    q = qubit.qalloc(3)
    squin.h(q[0])
    squin.sqrt_x(q[1])
    squin.y(q[1])
    squin.s(q[2])
    squin.sqrt_y(q[2])
    squin.cx(q[0], q[1])
    squin.x(q[0])
    squin.sqrt_x_adj(q[0])
    squin.h(q[2])
    squin.cz(q[1], q[2])
    squin.sqrt_y_adj(q[1])
    squin.z(q[1])
    squin.h(q[1])
    return default_post_processing(q)


def _tableau(kernel) -> stim.Tableau:
    circuit = GeminiLogicalSimulator().task(kernel).noiseless_tsim_circuit.stim_circuit
    prefix = stim.Circuit()
    for instruction in stim.Circuit(str(circuit)).flattened():
        if instruction.name not in ("M", "MZ", "DETECTOR", "OBSERVABLE_INCLUDE"):
            prefix.append(instruction)
    return prefix.to_tableau()


@pytest.mark.parametrize(
    "kernel",
    [_cx_twice, _basis_change_gadget, _mixed_runs],
    ids=["cx_twice", "basis_change_gadget", "mixed_runs"],
)
def test_merging_preserves_the_encoded_clifford(kernel, monkeypatch):
    monkeypatch.setattr(native_to_place, "EliminateRz", _FlushingEliminateRz)
    on = _tableau(kernel)

    monkeypatch.setattr(native_to_place, "MergeSingleQubitRuns", _NoOpMerge)
    off = _tableau(kernel)

    assert on == off


def _count_r(kernel) -> int:
    placed = LogicalNativeToPlace(arch_spec=get_arch_spec()).emit(
        kernel, no_raise=False
    )
    return sum(isinstance(s, place.R) for s in placed.callable_region.walk())


def test_issue_repro_emits_two_r():
    assert _count_r(_cx_twice) == 2


def test_merging_never_adds_r(monkeypatch):
    on = {k: _count_r(k) for k in (_cx_twice, _basis_change_gadget, _mixed_runs)}
    monkeypatch.setattr(native_to_place, "MergeSingleQubitRuns", _NoOpMerge)
    off = {k: _count_r(k) for k in (_cx_twice, _basis_change_gadget, _mixed_runs)}
    assert all(on[k] <= off[k] for k in on)
    assert on[_cx_twice] < off[_cx_twice]
