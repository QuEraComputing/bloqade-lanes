"""Compilation barriers fence only the addressed wires through the pipeline."""

from typing import Any

import pytest
from bloqade.native.dialects.gate import stmts as gate
from bloqade.native.upstream import SquinToNative
from bloqade.rewrite.passes import AggressiveUnroll
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade import qubit, squin, types as bloqade_types
from bloqade.gemini import logical as gemini_logical
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.dialects import move, place
from bloqade.lanes.passes import ALAPPlacePass, ASAPPlacePass, SequentialPlacePass
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz
from bloqade.lanes.rewrite.merge_single_qubit_runs import MergeSingleQubitRuns
from bloqade.lanes.rewrite.reorder_static_placement import (
    ReorderStaticPlacement,
    alap_reorder_policy,
    asap_reorder_policy,
)
from bloqade.lanes.transform import LogicalPipeline
from bloqade.lanes.transform.native_to_place import NativeToPlace
from bloqade.lanes.utils import constant_float

# Resolve the optional API dynamically for the supported Circuit 0.15 baseline.
barrier: Any = getattr(squin, "barrier", None)
broadcast_barrier: Any = getattr(squin.broadcast, "barrier", None)
Barrier: Any = getattr(qubit.stmts, "Barrier", None)

pytestmark = pytest.mark.skipif(
    not hasattr(qubit.stmts, "Barrier"), reason="Requires bloqade-circuit >= 0.16"
)


def _register_qubits(stmt: ir.Statement):
    register = stmt.args[-1].owner
    assert isinstance(register, ilist.New)
    return register.values


def _native(method):
    out = SquinToNative().emit(method, no_raise=False)
    AggressiveUnroll(out.dialects, no_raise=False).fixpoint(out)
    return out


def test_merge_stops_on_addressed_wire_only():
    @squin.kernel
    def main():
        q = squin.qalloc(2)
        squin.x(q[0])
        squin.x(q[1])
        barrier(q[0])
        squin.x(q[0])
        squin.x(q[1])

    out = _native(main)
    rewrite.Walk(MergeSingleQubitRuns()).rewrite(out.code)
    ops = [s for s in out.code.walk() if isinstance(s, (gate.R, Barrier))]
    assert [type(s) for s in ops] == [gate.R, Barrier, gate.R]
    assert _register_qubits(ops[0]) == _register_qubits(ops[1])
    assert _register_qubits(ops[2]) == _register_qubits(ops[1])


def test_optimizations_still_run_on_each_side():
    @squin.kernel
    def main():
        q = squin.qalloc(1)
        squin.x(q[0])
        squin.x(q[0])
        barrier(q[0])
        squin.x(q[0])
        squin.x(q[0])

    out = _native(main)
    rewrite.Walk(MergeSingleQubitRuns()).rewrite(out.code)
    assert not any(isinstance(s, (gate.R, gate.Rz)) for s in out.code.walk())
    assert sum(isinstance(s, Barrier) for s in out.code.walk()) == 1


def test_phase_commutation_flushes_only_barrier_qubits():
    q0, q1 = qubit.stmts.New(), qubit.stmts.New()
    reg = ilist.New(values=(q0.result, q1.result), elem_type=bloqade_types.QubitType)
    single = ilist.New(values=(q0.result,), elem_type=bloqade_types.QubitType)
    angle, zero = py.Constant(0.25), py.Constant(0.0)
    boundary = Barrier(single.result)
    block = ir.Block(
        [
            q0,
            q1,
            reg,
            single,
            angle,
            zero,
            gate.Rz(angle.result, reg.result),
            boundary,
            gate.R(zero.result, angle.result, reg.result),
        ]
    )
    rewrite.Walk(EliminateRz()).rewrite(block)
    ops = [s for s in block.stmts if isinstance(s, (gate.R, gate.Rz, Barrier))]
    assert [type(s) for s in ops] == [gate.Rz, Barrier, gate.R, gate.R]
    assert isinstance(ops[0], gate.Rz)
    assert isinstance(ops[2], gate.R)
    assert isinstance(ops[3], gate.R)
    assert _register_qubits(ops[0]) == (q0.result,)
    assert constant_float(ops[0].rotation_angle) == 0.25
    assert _register_qubits(ops[2]) == (q0.result,)
    assert constant_float(ops[2].axis_angle) == 0.0
    assert _register_qubits(ops[3]) == (q1.result,)
    assert constant_float(ops[3].axis_angle) == -0.25


@pytest.mark.parametrize(
    "pass_type", [SequentialPlacePass, ASAPPlacePass, ALAPPlacePass]
)
def test_barrier_survives_placement_merging_and_scheduling(pass_type):
    @squin.kernel
    def main():
        q = squin.qalloc(2)
        squin.x(q[0])
        broadcast_barrier(q)
        squin.x(q[1])

    out = NativeToPlace().emit(main, no_raise=False)
    pass_type(out.dialects)(out)
    out.verify()
    out.verify_type()
    placements = [s for s in out.code.walk() if isinstance(s, place.StaticPlacement)]
    assert len(placements) == 1
    ops = list(placements[0].body.blocks[0].stmts)
    assert [type(s) for s in ops] == [place.R, place.Barrier, place.R, place.Yield]
    assert isinstance(ops[1], place.Barrier)
    assert ops[1].qubits == (0, 1)


@pytest.mark.parametrize("policy", [asap_reorder_policy, alap_reorder_policy])
def test_scheduler_keeps_scoped_barrier_dependencies(policy):
    body = ir.Block()
    state = body.args.append_from(move.StateType)
    axis, angle = py.Constant(0.0), py.Constant(0.25)
    first = place.R(
        state, axis_angle=axis.result, rotation_angle=angle.result, qubits=(0,)
    )
    boundary = place.Barrier(first.state_after, qubits=(0,))
    independent = place.R(
        boundary.state_after,
        axis_angle=axis.result,
        rotation_angle=angle.result,
        qubits=(1,),
    )
    last = place.R(
        independent.state_after,
        axis_angle=axis.result,
        rotation_angle=angle.result,
        qubits=(0,),
    )
    body.stmts.extend(
        [first, boundary, independent, last, place.Yield(last.state_after)]
    )
    q0, q1 = qubit.stmts.New(), qubit.stmts.New()
    outer = ir.Block(
        [
            q0,
            q1,
            axis,
            angle,
            place.StaticPlacement((q0.result, q1.result), ir.Region(body)),
        ]
    )
    rewrite.Walk(ReorderStaticPlacement(policy)).rewrite(outer)
    placement = outer.last_stmt
    assert isinstance(placement, place.StaticPlacement)
    ops = [
        s
        for s in placement.body.blocks[0].stmts
        if isinstance(s, (place.R, place.Barrier))
    ]
    fenced = [s for s in ops if s.qubits == (0,)]
    assert [type(s) for s in fenced] == [place.R, place.Barrier, place.R]
    # A wire outside the barrier can move across it during scheduling.
    independent_index = ops.index(next(s for s in ops if s.qubits == (1,)))
    if policy is asap_reorder_policy:
        assert independent_index < ops.index(fenced[1])
    else:
        assert independent_index > ops.index(fenced[1])


@pytest.mark.parametrize(
    "pass_type", [SequentialPlacePass, ASAPPlacePass, ALAPPlacePass]
)
def test_logical_pipeline_preserves_fenced_x_gates_and_erases_directive(pass_type):
    @gemini_logical.kernel(aggressive_unroll=True)
    def main():
        q = qubit.qalloc(1)
        squin.x(q[0])
        barrier(q[0])
        squin.x(q[0])
        gemini_logical.terminal_measure(q)

    out = LogicalPipeline(get_arch_spec(), place_opt_type=pass_type).emit(
        main, no_raise=False
    )
    assert sum(isinstance(s, (move.LocalR, move.GlobalR)) for s in out.code.walk()) == 2
    assert not any(isinstance(s, (Barrier, place.Barrier)) for s in out.code.walk())
    out.verify()
    out.verify_type()


def test_barrier_preserves_encoded_clifford_with_pending_phase(monkeypatch):
    import stim

    from bloqade.gemini import GeminiLogicalSimulator
    from bloqade.gemini.logical.stdlib import default_post_processing
    from bloqade.lanes.transform import native_to_place

    class FlushingEliminateRz(EliminateRz):
        def __init__(self, no_raise=False):
            super().__init__(no_raise=no_raise, flush_residual=True)

    # Both circuits must retain their terminal Z frames for an exact unitary
    # comparison, rather than only preserving Z-basis measurement outcomes.
    monkeypatch.setattr(native_to_place, "EliminateRz", FlushingEliminateRz)

    @gemini_logical.kernel(aggressive_unroll=True)
    def reference():
        q = qubit.qalloc(2)
        squin.h(q[0])
        squin.s(q[0])
        squin.cx(q[0], q[1])
        return default_post_processing(q)

    @gemini_logical.kernel(aggressive_unroll=True)
    def fenced():
        q = qubit.qalloc(2)
        squin.h(q[0])
        squin.s(q[0])
        broadcast_barrier(q)
        squin.cx(q[0], q[1])
        return default_post_processing(q)

    def tableau(method):
        circuit = (
            GeminiLogicalSimulator().task(method).noiseless_tsim_circuit.stim_circuit
        )
        prefix = stim.Circuit()
        for instruction in stim.Circuit(str(circuit)).flattened():
            if instruction.name not in ("M", "MZ", "DETECTOR", "OBSERVABLE_INCLUDE"):
                prefix.append(instruction)
        return prefix.to_tableau()

    assert tableau(fenced) == tableau(reference)


def test_physical_barrier_after_cz_adds_no_moves_or_pulses():
    from bloqade.lanes.transform import PhysicalPipeline

    @squin.kernel
    def reference():
        q = qubit.qalloc(2)
        squin.h(q[0])
        squin.cz(q[0], q[1])
        squin.broadcast.measure(q)

    @squin.kernel
    def fenced():
        q = qubit.qalloc(2)
        squin.h(q[0])
        squin.cz(q[0], q[1])
        broadcast_barrier(q)
        squin.broadcast.measure(q)

    pipeline = PhysicalPipeline()
    actual = pipeline.emit(fenced, no_raise=False)
    expected = pipeline.emit(reference, no_raise=False)
    for kind in (move.Move, move.CZ, move.LocalR, move.GlobalR):
        assert sum(isinstance(s, kind) for s in actual.code.walk()) == sum(
            isinstance(s, kind) for s in expected.code.walk()
        )
    assert not any(isinstance(s, (Barrier, place.Barrier)) for s in actual.code.walk())
