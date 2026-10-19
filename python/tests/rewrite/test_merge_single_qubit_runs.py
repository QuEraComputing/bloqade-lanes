"""Tests for the MergeSingleQubitRuns rewrite rule.

Exactness is checked by simulation rather than by comparing IR: each case is a
small circuit description, built into a flat native block, rewritten, read back
and simulated as a statevector next to the original. That covers the matrix
algebra, the run boundaries and the placement of the merged gates at once, and
does not pin the rule to one of the many equal ways of writing a product.
"""

import math

import numpy as np
from bloqade.native.dialects import gate as native_gate
from kirin import ir, rewrite
from kirin.dialects import ilist, py

from bloqade import qubit as squin_qubit, types as bloqade_types
from bloqade.lanes.rewrite.merge_single_qubit_runs import (
    MergeSingleQubitRuns,
    decompose,
    gate_matrix,
)
from bloqade.lanes.utils import constant_float

QUBIT = bloqade_types.QubitType

# ── circuit descriptions ─────────────────────────────────────────────────
#
# ("R", phi, theta, qubits) | ("Rz", theta, qubits) | ("CZ", controls, targets)
# | ("M", qubits) for a measurement, | ("Rsym", qubits) for an R whose angle the
# compiler cannot read.


def _build(n: int, ops) -> tuple[ir.Block, list[ir.SSAValue]]:
    qubits = [squin_qubit.stmts.New() for _ in range(n)]
    stmts: list[ir.Statement] = list(qubits)
    results: list[ir.SSAValue] = [q.result for q in qubits]

    def register(indices):
        reg = ilist.New(values=tuple(results[i] for i in indices), elem_type=QUBIT)
        stmts.append(reg)
        return reg.result

    def constant(value):
        stmts.append(const := py.Constant(value))
        return const.result

    for op in ops:
        kind = op[0]
        if kind == "R":
            _, phi, theta, idx = op
            stmts.append(
                native_gate.stmts.R(constant(phi), constant(theta), register(idx))
            )
        elif kind == "Rz":
            _, theta, idx = op
            stmts.append(native_gate.stmts.Rz(constant(theta), register(idx)))
        elif kind == "CZ":
            _, controls, targets = op
            stmts.append(native_gate.stmts.CZ(register(controls), register(targets)))
        elif kind == "M":
            stmts.append(squin_qubit.stmts.Measure(register(op[1])))
        elif kind == "Rsym":
            # An angle produced by a statement: constant_float cannot read it.
            symbolic = py.Add(constant(0.125), constant(0.125))
            stmts.append(symbolic)
            stmts.append(
                native_gate.stmts.R(constant(0.0), symbolic.result, register(op[1]))
            )
        else:
            raise AssertionError(kind)
    return ir.Block(stmts), results


def _read_back(block: ir.Block, qubits: list[ir.SSAValue]):
    index = {q: i for i, q in enumerate(qubits)}

    def idx(register):
        return tuple(index[q] for q in register.owner.values)

    ops = []
    for stmt in block.stmts:
        if isinstance(stmt, native_gate.stmts.R):
            phi, theta = constant_float(stmt.axis_angle), constant_float(
                stmt.rotation_angle
            )
            if theta is None:
                ops.append(("Rsym", idx(stmt.qubits)))
            else:
                ops.append(("R", phi, theta, idx(stmt.qubits)))
        elif isinstance(stmt, native_gate.stmts.Rz):
            ops.append(("Rz", constant_float(stmt.rotation_angle), idx(stmt.qubits)))
        elif isinstance(stmt, native_gate.stmts.CZ):
            ops.append(("CZ", idx(stmt.controls), idx(stmt.targets)))
        elif isinstance(stmt, squin_qubit.stmts.Measure):
            ops.append(("M", idx(stmt.qubits)))
    return ops


def _r(phi, theta):
    c, s = math.cos(math.pi * theta), math.sin(math.pi * theta)
    axis = np.exp(1j * math.tau * phi)
    return np.array([[c, -1j * s * axis.conjugate()], [-1j * s * axis, c]])


def _rz(theta):
    return np.diag([np.exp(-1j * math.pi * theta), np.exp(1j * math.pi * theta)])


def _apply_1q(state, n, q, u):
    state = state.reshape([2] * n)
    state = np.moveaxis(np.tensordot(u, state, axes=([1], [q])), 0, q)
    return state.reshape(-1)


def _simulate(n: int, ops) -> np.ndarray:
    """Statevector after ``ops`` from a fixed generic product state.

    A generic input rather than ``|0...0>``, so no gate's action is hidden by
    landing on an eigenstate. Measurements are treated as identity: they only
    mark run boundaries here, and the rule must never move a gate across one.
    """
    rng = np.random.default_rng(7)
    state = np.ones(1, dtype=complex)
    for _ in range(n):
        v = rng.normal(size=2) + 1j * rng.normal(size=2)
        state = np.kron(state, v / np.linalg.norm(v))
    for op in ops:
        kind = op[0]
        if kind == "R":
            for q in op[3]:
                state = _apply_1q(state, n, q, _r(op[1], op[2]))
        elif kind == "Rz":
            for q in op[2]:
                state = _apply_1q(state, n, q, _rz(op[1]))
        elif kind == "Rsym":
            for q in op[1]:
                state = _apply_1q(state, n, q, _r(0.0, 0.25))
        elif kind == "CZ":
            state = state.reshape([2] * n).copy()
            for c, t in zip(op[1], op[2]):
                sl: list[slice | int] = [slice(None)] * n
                sl[c], sl[t] = 1, 1
                state[tuple(sl)] *= -1
            state = state.reshape(-1)
    return state


def _equal_up_to_phase(a, b) -> bool:
    return math.isclose(abs(np.vdot(a, b)), 1.0, abs_tol=1e-9)


def _run(n, ops):
    block, qubits = _build(n, ops)
    rewrite.Walk(MergeSingleQubitRuns()).rewrite(block)
    out = _read_back(block, qubits)
    assert _equal_up_to_phase(_simulate(n, ops), _simulate(n, out)), out
    return out


def _count(ops, kind):
    return sum(op[0] == kind for op in ops)


# ── decompose ────────────────────────────────────────────────────────────


def _rebuild(spec):
    u = np.eye(2, dtype=complex)
    for name, angles in spec:
        u = (_r(*angles) if name == "R" else _rz(*angles)) @ u
    return u


def _same_up_to_phase(u, v):
    return math.isclose(abs(np.trace(u.conj().T @ v)) / 2, 1.0, abs_tol=1e-9)


def test_decompose_round_trips_random_unitaries():
    rng = np.random.default_rng(0)
    for _ in range(200):
        z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        u, _ = np.linalg.qr(z)
        spec = decompose(u)
        assert _same_up_to_phase(_rebuild(spec), u)
        assert [name for name, _ in spec] in (["R"], ["R", "Rz"])


def _cliffords():
    gens = [_r(0.0, 0.25), _rz(0.25)]
    found = [np.eye(2, dtype=complex)]
    frontier = list(found)
    while frontier:
        nxt = []
        for u in frontier:
            for g in gens:
                v = g @ u
                if not any(_same_up_to_phase(v, w) for w in found):
                    found.append(v)
                    nxt.append(v)
        frontier = nxt
    return found


def test_decompose_cliffords_to_grid_angles():
    """Every one of the 24 Cliffords comes out on the eighth-turn grid, as
    exact literals -- float noise would defeat CSE and so gate fusion."""
    cliffords = _cliffords()
    assert len(cliffords) == 24
    for u in cliffords:
        spec = decompose(u)
        assert _same_up_to_phase(_rebuild(spec), u)
        for _, angles in spec:
            for angle in angles:
                assert angle * 8 == round(angle * 8)
                assert 0.0 <= angle < 1.0


def test_decompose_identity_and_diagonal():
    assert decompose(np.eye(2)) == ()
    assert decompose(-np.eye(2)) == ()
    assert decompose(_rz(0.25)) == (("Rz", (0.25,)),)


def test_gate_matrix_reads_constants_only():
    block, _ = _build(1, [("Rsym", (0,))])
    (r,) = [s for s in block.stmts if isinstance(s, native_gate.stmts.R)]
    assert gate_matrix(r) is None


# ── the rewrite ──────────────────────────────────────────────────────────

H = [("Rz", 0.25, (1,)), ("R", 0.0, 0.25, (1,)), ("Rz", 0.25, (1,))]


def test_issue_1117_repro():
    """CX twice: the H . H between the two CZ cancels, leaving 2 R from 4."""
    ops = H + [("CZ", (0,), (1,))] + H + H + [("CZ", (0,), (1,))] + H
    out = _run(2, ops)
    assert _count(out, "R") == 2
    assert _count(out, "CZ") == 2


def test_inverse_pair_vanishes():
    out = _run(1, [("R", 0.25, 0.25, (0,)), ("R", 0.75, 0.25, (0,))])
    assert out == []


def test_run_that_cannot_shrink_is_left_alone():
    """R . Rz already is R then Rz: rebuilding it would gain nothing."""
    ops = [("R", 0.0, 0.25, (0,)), ("Rz", 0.25, (0,))]
    assert _run(1, ops) == ops


def test_cz_measure_and_symbolic_angles_are_barriers():
    for barrier in (("CZ", (0,), (1,)), ("M", (0,)), ("Rsym", (0,))):
        ops = [("R", 0.0, 0.25, (0,)), barrier, ("R", 0.0, 0.75, (0,))]
        assert _run(2, ops) == ops, barrier


def test_barrier_only_ends_its_own_qubits_runs():
    """A CZ on q0/q1 does not stop q2's pair from cancelling."""
    ops = [
        ("R", 0.0, 0.25, (2,)),
        ("CZ", (0,), (1,)),
        ("R", 0.0, 0.75, (2,)),
    ]
    assert _run(3, ops) == [("CZ", (0,), (1,))]


def test_broadcast_layer_stays_one_statement():
    """H on a whole register twice is the identity; H then S on a whole
    register merges to one broadcast R rather than one per qubit."""
    layer = [
        ("Rz", 0.25, (0, 1, 2)),
        ("R", 0.0, 0.25, (0, 1, 2)),
        ("Rz", 0.25, (0, 1, 2)),
    ]
    assert _run(3, layer + layer) == []

    out = _run(3, layer + [("R", 0.0, 0.5, (0, 1, 2))])
    rs = [op for op in out if op[0] == "R"]
    assert len(rs) == 1 and rs[0][3] == (0, 1, 2)


def test_broadcast_is_never_split():
    """q0 alone could cancel its pair, but only by splitting the shared gate."""
    ops = [("R", 0.0, 0.25, (0, 1)), ("R", 0.0, 0.75, (0,))]
    assert _run(2, ops) == ops


def test_broadcasts_on_the_same_qubits_merge():
    ops = [("R", 0.0, 0.25, (0, 1)), ("R", 0.25, 0.25, (0, 1)), ("CZ", (0,), (1,))]
    out = _run(2, ops)
    rs = [op for op in out if op[0] == "R"]
    assert len(rs) == 1 and rs[0][3] == (0, 1)


def test_single_qubit_runs_merge_beside_an_untouched_broadcast():
    """q1's run is two lone gates; q0 shares a broadcast with q2 and stays."""
    ops = [
        ("R", 0.0, 0.25, (0, 2)),
        ("R", 0.0, 0.25, (1,)),
        ("R", 0.0, 0.75, (1,)),
        ("R", 0.0, 0.75, (0,)),
    ]
    assert _run(3, ops) == [("R", 0.0, 0.25, (0, 2)), ("R", 0.0, 0.75, (0,))]


def test_random_clifford_and_generic_circuits_are_exact():
    rng = np.random.default_rng(1117)
    n = 3
    grid = [i / 8 for i in range(8)]
    for trial in range(60):
        ops = []
        for _ in range(rng.integers(5, 25)):
            kind = rng.choice(["R", "Rz", "CZ", "M"], p=[0.45, 0.3, 0.2, 0.05])
            qs = tuple(
                sorted(rng.choice(n, size=rng.integers(1, n + 1), replace=False))
            )
            angle = (lambda: rng.choice(grid)) if trial % 2 else rng.random
            if kind == "R":
                ops.append(("R", float(angle()), float(angle()), qs))
            elif kind == "Rz":
                ops.append(("Rz", float(angle()), qs))
            elif kind == "CZ":
                a, b = rng.choice(n, size=2, replace=False)
                ops.append(("CZ", (int(a),), (int(b),)))
            else:
                ops.append(("M", qs[:1]))
        out = _run(n, ops)
        # Never more R pulses on any qubit than it started with.
        for q in range(n):
            before = sum(1 for op in ops if op[0] == "R" and q in op[3])
            after = sum(1 for op in out if op[0] == "R" and q in op[3])
            assert after <= before


def test_nothing_to_merge_reports_no_change():
    ops = [("R", 0.0, 0.25, (0,)), ("CZ", (0,), (1,)), ("R", 0.0, 0.25, (0,))]
    block, _ = _build(2, ops)
    assert not rewrite.Walk(MergeSingleQubitRuns()).rewrite(block).has_done_something
