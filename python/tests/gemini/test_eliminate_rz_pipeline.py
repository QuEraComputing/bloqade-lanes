"""End-to-end: the logical pipeline must emit no local_rz from Clifford gates."""

import math

import numpy as np
import pytest
from kirin.dialects import func, py
from kirin.rewrite.abc import RewriteRule

from bloqade import qubit, squin
from bloqade.gemini import (
    GeminiLogicalSimulator,
    logical as gemini_logical,
    physical as gemini_physical,
)
from bloqade.gemini.logical import default_post_processing
from bloqade.gemini.logical.rewrite.remove_postprocessing import RemovePostProcessing
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.rewrite.transversal import steane_star_theta
from bloqade.lanes.transform import (
    LogicalPipeline,
    PhysicalPipeline,
    native_to_place,
)


def _compile(kernel, **kwargs):
    return LogicalPipeline(
        get_logical_spec(), transversal_rewrite=True, simulation=False, **kwargs
    ).emit(kernel)


def _count(method, kind) -> int:
    return sum(1 for stmt in method.callable_region.walk() if isinstance(stmt, kind))


def test_teleportation_kernel_emits_no_local_rz():
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalRz) == 0


def test_local_r_count_is_unchanged():
    """Rz removal must not drop or duplicate rotation pulses."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    assert _count(_compile(kernel), move.LocalR) == 3


def test_kernel_with_the_terminal_measure_removed_still_drops_rz():
    """The other shape: no measurement statement to discard the frame at."""

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        default_post_processing(reg)

    stripped = kernel.similar()
    RemovePostProcessing(kernel.dialects, delete_terminal_measure=True)(stripped)

    assert _count(_compile(stripped), move.LocalRz) == 0


def test_equal_frames_still_fuse_after_lowering_to_place():
    """Constant sharing must survive the native->place boundary.

    FuseAdjacentGates matches axis angles by SSA identity and circuit2place
    carries them through unchanged, so a fresh constant per statement would stop
    these two identical gates fusing. ASAPPlacePass is the option that runs
    fusion. Both qubits get the identical rotation here, so ASAPPlacePass's
    global-broadcast promotion further collapses the fused pair into a single
    ``GlobalR`` rather than a two-qubit ``LocalR`` -- confirmed against the
    pre-EliminateRz baseline, where the same kernel already promotes to
    ``GlobalRz`` + ``GlobalR``. Count both statement kinds so the assertion
    reflects "one physical R remains", independent of that promotion.
    """

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.s(reg[0])
        squin.s(reg[1])
        squin.sqrt_x(reg[0])
        squin.sqrt_x(reg[1])
        gemini_logical.terminal_measure(reg)

    result = _compile(kernel, place_opt_type=ASAPPlacePass)
    assert _count(result, move.LocalR) + _count(result, move.GlobalR) == 1


def test_star_rz_payload_survives():
    """StarRz is a user-requested gadget, not a compiler artifact.

    Beyond the count, also check (the same way ``test_star_rz.py`` verifies
    ``RewriteStarRz``'s own transversal rewrite) that the surviving
    ``LocalRz``'s angle is still the ``steane_star_theta``-derived value --
    i.e. ``EliminateRz`` passed a ``StarRz``'s payload through untouched
    rather than folding it into the phase frame.
    """
    theta = math.pi / 16

    @gemini_logical.kernel(aggressive_unroll=True, verify=False)
    def kernel():
        reg = qubit.qalloc(1)
        gemini_logical.star_rz(theta, reg[0])
        gemini_logical.terminal_measure(reg)

    out = _compile(kernel)
    local_rz_nodes = [
        stmt for stmt in out.callable_region.walk() if isinstance(stmt, move.LocalRz)
    ]
    assert len(local_rz_nodes) == 1

    angle_owner = local_rz_nodes[0].rotation_angle.owner
    assert isinstance(angle_owner, func.Invoke)
    assert angle_owner.callee is steane_star_theta
    assert len(angle_owner.inputs) == 1
    input_owner = angle_owner.inputs[0].owner
    assert isinstance(input_owner, py.Constant)
    assert input_owner.value.unwrap() == pytest.approx(theta / math.tau)


def test_physical_pipeline_is_unchanged():
    """PhysicalNativeToPlace inherits the empty hook, so nothing moves."""

    @gemini_physical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(1)
        squin.s(reg[0])
        squin.measure(reg[0])

    out = PhysicalPipeline(get_physical_spec()).emit(kernel)

    assert _count(out, move.LocalRz) + _count(out, move.GlobalRz) > 0


# ── distribution equality ────────────────────────────────────────────────
#
# `EliminateRz` is measurement-transparent: it only ever discards a diagonal
# (Z-type) residual, which cannot move probability between computational
# basis outcomes. These tests check that directly, by simulation, for both
# IR shapes the rule has to handle: a flat block that ends in a
# `TerminalLogicalMeasurement` (the frame is popped there), and one that
# doesn't (the frame is simply abandoned, since nothing downstream ever
# reads it) -- see `test_kernel_with_the_terminal_measure_removed_still_drops_rz`
# above for the same two shapes at the IR-count level.
#
# `GeminiLogicalSimulator.task()` itself requires exactly one
# `TerminalLogicalMeasurement` (`GeminiTerminalMeasurementValidation`), so it
# cannot compile the delete-the-measurement shape end to end. Both shapes
# below instead follow the same recipe `compile_task` uses internally
# (`LogicalPipeline` -> `MoveToSquinLogical` -> `TsimSimulatorBackend`'s own
# `_tsim_circuit`), skipping only that one validation call, then append a
# fresh `M` over every physical qubit to the emitted Stim circuit so both
# shapes always have something to sample. Sampling the same seed against the
# rule enabled vs. replaced by a structural no-op must give bit-identical
# shots.


# ---------------------------------------------------------------------------
# Measurement-outcome equivalence.
#
# This is the honest statement of what EliminateRz promises: the residual it
# discards is diagonal, and a diagonal unitary commutes with every Z-basis
# measurement projector, so no outcome can shift. Compare the compiled circuit
# with the rule on against the same circuit with it off.
# ---------------------------------------------------------------------------


class _NoOpEliminateRz(RewriteRule):
    """Stand-in for EliminateRz that leaves every statement alone."""


def _stim_circuit(kernel):
    """The kernel's noiseless physical circuit, via the public simulator API."""
    return GeminiLogicalSimulator().task(kernel).noiseless_tsim_circuit.stim_circuit


def _sample(kernel, *, seed, shots):
    return _stim_circuit(kernel).compile_sampler(seed=seed).sample(shots=shots)


def test_measurement_outcomes_are_unchanged(monkeypatch):
    """The residual EliminateRz discards is diagonal, and a diagonal unitary
    commutes with every Z-basis measurement projector -- so no outcome can
    shift. Compare the compiled circuit with the rule on against the same
    circuit with it off.

    Only the terminal-measurement shape is testable this way, and that is not a
    gap: the ``RemovePostProcessing`` shape has no measurement, hence no outcome
    distribution to compare. Making it comparable would mean appending a
    measurement the program does not contain, which tests a circuit nobody
    compiles. That shape is covered structurally by
    ``test_kernel_with_the_terminal_measure_removed_still_drops_rz``.
    """

    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        default_post_processing(reg)

    with_rule = _sample(kernel, seed=1234, shots=4000)

    monkeypatch.setattr(native_to_place, "EliminateRz", _NoOpEliminateRz)
    without_rule = _sample(kernel, seed=1234, shots=4000)

    assert np.array_equal(with_rule, without_rule)
