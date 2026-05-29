"""Tests for movement_kernel decorator."""

import bloqade.squin as squin
import pytest

from bloqade.gemini.logical.movement_group import movement_kernel
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.dialects.movement import move_to


def test_movement_kernel_exists():
    assert movement_kernel is not None


def test_movement_kernel_compiles_plain_cz():
    """movement_kernel compiles a plain CZ without move_to."""

    # verify=False: kernel without terminal_measure only compiles with verify off
    @movement_kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])

    assert k is not None


def test_movement_kernel_compiles_move_to_then_cz():
    """movement_kernel accepts move_to followed by CZ."""

    # LocationAddress objects must be captured from outside the kernel
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=1, site_id=0)

    @movement_kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        move_to([q[0], q[1]], [loc_a, loc_b])
        squin.cz(q[0], q[1])

    assert k is not None


def test_movement_kernel_length_mismatch_raises():
    """MoveToValidation surfaces length mismatch at decoration time."""
    from kirin.ir.exception import ValidationErrorGroup

    # LocationAddress objects must be captured from outside the kernel
    # (constructors can't be called inside kernel bodies directly)
    loc_a = LocationAddress(zone_id=0, word_id=0, site_id=0)
    loc_b = LocationAddress(zone_id=0, word_id=1, site_id=0)

    with pytest.raises(ValidationErrorGroup):

        @movement_kernel
        def k():
            q = squin.qalloc(2)
            # 1 qubit but 2 locations — length mismatch
            move_to([q[0]], [loc_a, loc_b])


def test_existing_kernel_unaffected():
    """Plain @kernel usage is unaffected (regression canary)."""
    import bloqade.gemini as gemini

    # verify=False: kernel without terminal_measure only compiles with verify off
    @gemini.logical.kernel(verify=False)
    def k():
        q = squin.qalloc(2)
        squin.cz(q[0], q[1])

    assert k is not None


def test_movement_kernel_rejects_move_to_on_plain_kernel():
    """Plain @kernel does not support movement.move_to."""
    import bloqade.gemini as gemini

    # LocationAddress objects must be captured from outside the kernel
    loc = LocationAddress(zone_id=0, word_id=0, site_id=0)

    with pytest.raises(Exception):

        @gemini.logical.kernel(verify=False)
        def k():
            q = squin.qalloc(1)
            move_to([q[0]], [loc])
