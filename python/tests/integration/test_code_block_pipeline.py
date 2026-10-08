"""Code blocks end to end through PhysicalPipeline."""

import pytest

from bloqade import squin
from bloqade.gemini.physical import kernel
from bloqade.lanes import code_block
from bloqade.lanes.analysis.code_blocks import CodeBlockWarning, block_shape_error
from bloqade.lanes.dialects import move
from bloqade.lanes.transform import PhysicalPipeline


@kernel
def steane_transversal_cx():
    def new_steane_block():
        q = squin.qalloc(7)
        code_block.register(q)
        return q

    a = new_steane_block()
    b = new_steane_block()
    squin.broadcast.cx(a, b)
    squin.broadcast.measure(a + b)


def _fill_layout(out) -> tuple:
    (fill,) = [s for s in out.callable_region.walk() if isinstance(s, move.Fill)]
    return fill.location_addresses


def test_steane_blocks_get_matching_slots_in_two_words():
    out = PhysicalPipeline().emit(steane_transversal_cx, no_raise=False)
    layout = _fill_layout(out)
    a, b = layout[:7], layout[7:]
    assert block_shape_error(a, 8) is None
    assert block_shape_error(b, 8) is None
    assert a[0].word_id != b[0].word_id
    assert a[0].site_id == b[0].site_id


def test_opt_out_compiles_with_one_warning():
    with pytest.warns(CodeBlockWarning, match="use_code_blocks=False"):
        out = PhysicalPipeline(use_code_blocks=False).emit(
            steane_transversal_cx, no_raise=False
        )
    assert len(_fill_layout(out)) == 14
