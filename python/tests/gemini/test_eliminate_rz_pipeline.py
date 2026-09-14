"""End-to-end: the logical pipeline must emit no local_rz from Clifford gates."""

import math

import numpy as np
import pytest
import stim
from kirin.dialects import func, py
from kirin.rewrite.abc import RewriteRule

from bloqade import qubit, squin
from bloqade.gemini import logical as gemini_logical, physical as gemini_physical
from bloqade.gemini.device.simulator_backend import TsimSimulatorBackend
from bloqade.gemini.logical import default_post_processing
from bloqade.gemini.logical.rewrite.remove_postprocessing import RemovePostProcessing
from bloqade.lanes.arch.gemini import physical as gemini_physical_arch
from bloqade.lanes.arch.gemini.logical import get_arch_spec as get_logical_spec
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_spec
from bloqade.lanes.dialects import move
from bloqade.lanes.noise_model import generate_logical_noise_model
from bloqade.lanes.passes import ASAPPlacePass
from bloqade.lanes.rewrite.eliminate_rz import EliminateRz
from bloqade.lanes.rewrite.transversal import steane_star_theta
from bloqade.lanes.transform import (
    LogicalPipeline,
    MoveToSquinLogical,
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


class _NoOpEliminateRz(RewriteRule):
    """Structural stand-in for a disabled ``EliminateRz``: touches nothing, so
    every native ``Rz`` survives to the physical circuit."""


def _physical_stim_circuit(kernel):
    """Compile a logical kernel to its noiseless physical Stim circuit.

    Mirrors ``bloqade.gemini.compile.task.compile_task``'s
    ``LogicalPipeline`` -> ``MoveToSquinLogical`` steps, but skips
    ``run_squin_kernel_validation`` so a kernel with no terminal measurement
    can still be compiled (see the module note above).
    """
    physical_move_kernel = LogicalPipeline(transversal_rewrite=True).emit(kernel)
    physical_squin_kernel = MoveToSquinLogical(
        arch_spec=gemini_physical_arch.get_arch_spec(),
        noise_model=generate_logical_noise_model(),
        add_noise=False,
    ).emit(physical_move_kernel)
    return TsimSimulatorBackend()._tsim_circuit(physical_squin_kernel).stim_circuit


def _distribution_circuit(kernel):
    """The kernel's physical circuit, with any measurement tail replaced by a
    single fresh ``M`` over every physical qubit."""
    circuit = _physical_stim_circuit(kernel)
    prefix = stim.Circuit()
    for instruction in stim.Circuit(str(circuit)).flattened():
        if instruction.name not in ("M", "MZ", "DETECTOR", "OBSERVABLE_INCLUDE"):
            prefix.append(instruction)
    prefix.append("M", list(range(prefix.num_qubits)))
    return prefix


def _sampled_shots(kernel, *, eliminate_rz_enabled: bool, seed: int, shots: int):
    original = native_to_place.EliminateRz
    native_to_place.EliminateRz = original if eliminate_rz_enabled else _NoOpEliminateRz
    try:
        circuit = _distribution_circuit(kernel)
    finally:
        native_to_place.EliminateRz = original
    assert native_to_place.EliminateRz is EliminateRz
    return circuit.compile_sampler(seed=seed).sample(shots=shots)


def _make_distribution_kernel():
    @gemini_logical.kernel(aggressive_unroll=True)
    def kernel():
        reg = qubit.qalloc(2)
        squin.h(reg[1])
        squin.s(reg[1])
        squin.cx(reg[0], reg[1])
        gemini_logical.terminal_measure(reg)

    return kernel


def test_distribution_equality_with_terminal_measure_present():
    """Shape (a): the frame is popped at a real ``TerminalLogicalMeasurement``."""
    kernel = _make_distribution_kernel()
    with_rule = _sampled_shots(kernel, eliminate_rz_enabled=True, seed=1234, shots=4000)
    without_rule = _sampled_shots(
        kernel, eliminate_rz_enabled=False, seed=1234, shots=4000
    )
    assert np.array_equal(with_rule, without_rule)


def test_distribution_equality_with_terminal_measure_deleted():
    """Shape (b): ``RemovePostProcessing(delete_terminal_measure=True)`` has
    removed the only ``TerminalLogicalMeasurement``, so the frame is never
    explicitly popped -- it is simply left in ``EliminateRz._frame`` when the
    walk ends, with no statement left to have absorbed it into."""
    kernel = _make_distribution_kernel()
    stripped = kernel.similar()
    RemovePostProcessing(kernel.dialects, delete_terminal_measure=True)(stripped)

    with_rule = _sampled_shots(
        stripped, eliminate_rz_enabled=True, seed=5678, shots=4000
    )
    without_rule = _sampled_shots(
        stripped, eliminate_rz_enabled=False, seed=5678, shots=4000
    )
    assert np.array_equal(with_rule, without_rule)
