"""Tests for movement_kernel decorator."""

import bloqade.squin as squin
import pytest

import bloqade.gemini as gemini
from bloqade.gemini.logical import loc, move_to, terminal_measure
from bloqade.gemini.physical import kernel as movement_kernel
from bloqade.lanes.bytecode.encoding import LocationAddress


def test_movement_kernel_exists():
    assert movement_kernel is not None


def test_movement_kernel_compiles_plain_cz():
    """movement_kernel compiles a plain CZ without move_to."""

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        return terminal_measure(q)

    assert k is not None


def test_movement_kernel_compiles_move_to_then_cz():
    """movement_kernel accepts move_to followed by CZ."""

    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc_a, loc_b])
        squin.cz(q[0], q[1])
        return terminal_measure(q)

    assert k is not None


def test_loc_inline_compiles():
    """loc() constructs a LocationAddress inside the kernel body."""

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        move_to([q[0]], [loc(0, 0, 0)])
        return terminal_measure(q)

    assert k is not None


def test_loc_inline_move_to_then_cz():
    """loc() + move_to + CZ compiles end-to-end."""

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc(0, 0, 0), loc(0, 1, 0)])
        squin.cz(q[0], q[1])
        return terminal_measure(q)

    assert k is not None


def test_existing_kernel_unaffected():
    """Plain @kernel usage is unaffected (regression canary)."""

    @gemini.logical.kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])
        return terminal_measure(q)

    assert k is not None


def test_movement_kernel_rejects_move_to_on_plain_kernel():
    """Plain @kernel does not support movement.move_to."""

    from kirin.lowering.exception import BuildError

    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)

    with pytest.raises(BuildError, match="unsupported dialect"):

        @gemini.logical.kernel(aggressive_unroll=True)
        def k():
            q = squin.qalloc(1)
            move_to([q[0]], [loc_a])
            return terminal_measure(q)


def test_movement_kernel_multi_move_to_then_cz():
    """Two consecutive move_to calls followed by CZ accumulate layers correctly."""
    loc0 = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc1 = LocationAddress(zone_id=0, word_id=2, site_id=0)

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(3)
        move_to([q[0]], [loc0])
        move_to([q[1]], [loc1])
        squin.cz(q[0], q[2])
        return terminal_measure(q)

    assert k is not None


def test_movement_kernel_terminal_move_to_valid():
    """move_to immediately before the terminal measure is accepted by the
    movement_kernel validation suite at kernel-definition time (it must not be
    rejected for lacking a following CZ).

    This asserts decoration-time acceptance only. The placement-layer behavior
    it depends on — measure_placements committing a terminal user-move instead
    of returning bottom — is verified directly in
    ``tests/analysis/placement/test_user_moved.py::
    test_measure_placements_user_moved_concretizes``. (Full move-lowering of
    move_to -> terminal_measure is separately gated by terminal-measurement
    validation, which is out of scope here.)"""
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        move_to([q[0]], [loc_a])
        return terminal_measure(q)

    assert k is not None


def test_move_to_before_sq_gate_compiles_successfully():
    """move_to followed by a single-qubit gate compiles without error.

    UserMoved state passes through SQ gates cleanly (no bottom state produced).
    """
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        move_to([q[0]], [loc_a])
        squin.rz(0.0, q[0])
        return terminal_measure(q)

    assert k is not None


def test_movement_kernel_move_to_then_sq_then_cz():
    """move_to -> SQ gate -> CZ compiles correctly (UserMoved passes through SQ gates)."""
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @movement_kernel(aggressive_unroll=True)
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc_a, loc_b])
        squin.rz(0.0, q[0])  # SQ gate -- must not corrupt UserMoved state
        squin.cz(q[0], q[1])
        return terminal_measure(q)

    assert k is not None


@pytest.mark.slow
def test_move_to_cz_full_pipeline_regression():
    """Regression: move_to + CZ compiles through the full logical pipeline.

    Two sub-cases:
    1. Plain CZ baseline (no move_to) — establishes the AOD shot baseline.
    2. Pre-positioned CZ — move_to places q1 at the CZ-partner slot of q0
       before the CZ, so the compiler should need zero additional forward
       moves at the CZ site.

    With the pre-staging fix in LogicalPlacementStrategyNoHome, move_to replaces
    the CZ-forward moves exactly, so n_prepos == n_plain.
    """
    from bloqade.gemini.logical import kernel as plain_kernel
    from bloqade.lanes.arch.gemini.logical import get_arch_spec
    from bloqade.lanes.dialects import move
    from bloqade.lanes.transform import LogicalPipeline

    arch = get_arch_spec()

    # In the logical arch the layout heuristic places 2 qubits at the first
    # two home positions: q0 → (zone=0, word=0, site=0),
    #                     q1 → (zone=0, word=2, site=0).
    # word 0 and word 1 are a CZ pair (word 0 is home, word 1 is staging).
    # Pre-positioning q1 at word 1 puts atoms in blockade range before the CZ.
    assert (
        arch.get_cz_partner(LocationAddress(zone_id=0, word_id=0, site_id=0))
        is not None
    )
    q1_cz_slot = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @plain_kernel(verify=False)
    def k_plain():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])

    @movement_kernel(verify=False)
    def k_prepos():
        q = squin.qalloc(2)
        move_to([q[1]], [q1_cz_slot])
        squin.cz(q[0], q[1])

    move_plain = LogicalPipeline().emit(k_plain, no_raise=False)
    move_prepos = LogicalPipeline().emit(k_prepos, no_raise=False)

    n_plain = sum(
        1 for s in move_plain.callable_region.walk() if isinstance(s, move.Move)
    )
    n_prepos = sum(
        1 for s in move_prepos.callable_region.walk() if isinstance(s, move.Move)
    )

    # Both kernels must produce at least one AOD shot.
    assert n_plain > 0, "plain CZ produced no moves"
    assert n_prepos > 0, "pre-positioned CZ produced no moves"

    # Pre-staged atoms are not returned home before the CZ:
    # move_to replaces the CZ-forward step, so total shot counts are equal.
    assert n_prepos == n_plain, (
        f"pre-positioned CZ generated {n_prepos} shots (plain={n_plain}); "
        "move_to should replace the CZ-forward moves exactly"
    )
