"""The [[4,2,2]] routing baseline keeps its block-shaped pinned layout."""

from benchmarks.kernels.medium.code422_physical_16 import code422_physical_16

from bloqade.lanes.analysis.code_blocks import block_shape_error
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.dialects import place
from bloqade.lanes.transform.native_to_place import NativeToPlace


def test_pins_form_four_blocks_two_per_word():
    out = NativeToPlace(arch_spec=get_arch_spec()).emit(
        code422_physical_16, no_raise=False
    )
    pins = [
        stmt.location_address
        for stmt in out.callable_region.walk()
        if isinstance(stmt, place.NewLogicalQubit) and stmt.location_address is not None
    ]
    assert len(pins) == 16
    blocks = [pins[4 * i : 4 * i + 4] for i in range(4)]
    for block in blocks:
        assert block_shape_error(block, 8) is None
    assert [(b[0].word_id, b[0].site_id) for b in blocks] == [
        (0, 0),
        (0, 4),
        (2, 0),
        (2, 4),
    ]
