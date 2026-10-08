"""Registration, validation and lowering of ``code_block.register``."""

import warnings

import pytest
from kirin.dialects import ilist
from kirin.ir.exception import ValidationErrorGroup

from bloqade import squin
from bloqade.gemini.common.dialects.qubit import new_at
from bloqade.gemini.physical import kernel
from bloqade.lanes import code_block
from bloqade.lanes.analysis.code_blocks import CodeBlockTag, CodeBlockWarning
from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.dialects import place
from bloqade.lanes.dialects.code_block import Register
from bloqade.lanes.transform.native_to_place import (
    NativeToPlace,
    PhysicalNativeToPlace,
)


def _tags(mt) -> list[CodeBlockTag | None]:
    return [
        stmt.code_block
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, place.NewPinnedQubit)
    ]


def _registers(mt) -> int:
    return sum(isinstance(stmt, Register) for stmt in mt.callable_region.walk())


@kernel
def two_inlined_blocks():
    def new_block():
        q = squin.qalloc(4)
        code_block.register(q)
        return q

    a = new_block()
    b = new_block()
    squin.broadcast.cz(a, b)
    squin.broadcast.measure(a + b)


@kernel
def one_block_and_spare():
    q = squin.qalloc(3)
    spare = squin.qalloc(1)
    code_block.register(q)
    squin.broadcast.measure(q + spare)


@kernel
def every_error():
    big = squin.qalloc(9)
    code_block.register(big)

    loose = squin.qalloc(1)
    p0 = new_at(0, 0, 0)
    code_block.register(ilist.IList([p0, loose[0]]))

    gap = ilist.IList([new_at(0, 2, 0), new_at(0, 2, 2)])
    code_block.register(gap)
    reversed_pins = ilist.IList([new_at(0, 4, 1), new_at(0, 4, 0)])
    code_block.register(reversed_pins)
    two_words = ilist.IList([new_at(0, 6, 0), new_at(0, 8, 1)])
    code_block.register(two_words)

    shared = squin.qalloc(2)
    code_block.register(shared)
    code_block.register(ilist.IList([shared[0]]))

    squin.broadcast.measure(
        big + loose + ilist.IList([p0]) + gap + reversed_pins + two_words + shared
    )


@kernel
def pinned_block():
    a = new_at(0, 2, 3)
    b = new_at(0, 2, 4)
    c = new_at(0, 2, 5)
    code_block.register(ilist.IList([a, b, c]))
    squin.broadcast.measure(ilist.IList([a, b, c]))


def test_inlined_allocator_gives_one_block_per_call():
    out = PhysicalNativeToPlace(arch_spec=get_arch_spec()).emit(
        two_inlined_blocks, no_raise=False
    )
    assert _tags(out) == [CodeBlockTag(0, p) for p in range(4)] + [
        CodeBlockTag(1, p) for p in range(4)
    ]
    assert _registers(out) == 0


def test_unregistered_qubits_have_no_tag():
    out = PhysicalNativeToPlace(arch_spec=get_arch_spec()).emit(
        one_block_and_spare, no_raise=False
    )
    assert _tags(out) == [CodeBlockTag(0, p) for p in range(3)] + [None]


def test_valid_pinned_block_is_accepted():
    out = PhysicalNativeToPlace(arch_spec=get_arch_spec()).emit(
        pinned_block, no_raise=False
    )
    assert _tags(out) == [CodeBlockTag(0, p) for p in range(3)]


def test_every_error_is_reported_at_once():
    with pytest.raises(ValidationErrorGroup) as info:
        PhysicalNativeToPlace(arch_spec=get_arch_spec()).emit(
            every_error, no_raise=False
        )
    messages = [str(err.args[0]) for err in info.value.errors]
    expected = [
        "a word has only 8 sites",
        "some qubits in the block are pinned",
        "got sites [0, 2]",
        "got sites [1, 0]",
        "spans several words",
        "already in another code block",
    ]
    assert len(messages) == len(expected), messages
    for fragment in expected:
        assert sum(fragment in m for m in messages) == 1, (fragment, messages)


def test_no_raise_drops_every_block_with_one_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = PhysicalNativeToPlace(arch_spec=get_arch_spec()).emit(
            every_error, no_raise=True
        )
    block_warnings = [w for w in caught if issubclass(w.category, CodeBlockWarning)]
    assert len(block_warnings) == 1
    assert "ignoring all 7" in str(block_warnings[0].message)
    assert all(tag is None for tag in _tags(out))
    assert _registers(out) == 0


def test_opt_out_matches_unregistered_kernel():
    @kernel
    def unregistered():
        a = squin.qalloc(4)
        b = squin.qalloc(4)
        squin.broadcast.cz(a, b)
        squin.broadcast.measure(a + b)

    with pytest.warns(CodeBlockWarning, match="use_code_blocks=False"):
        opted_out = PhysicalNativeToPlace(
            arch_spec=get_arch_spec(), use_code_blocks=False
        ).emit(two_inlined_blocks, no_raise=False)
    plain = PhysicalNativeToPlace(arch_spec=get_arch_spec()).emit(
        unregistered, no_raise=False
    )
    assert all(tag is None for tag in _tags(opted_out))
    assert _registers(opted_out) == 0
    assert [type(s) for s in opted_out.callable_region.walk()] == [
        type(s) for s in plain.callable_region.walk()
    ]


def test_generic_lowering_strips_registrations_with_warning():
    @squin.kernel.add(code_block)
    def unmeasured():
        a = squin.qalloc(4)
        code_block.register(a)
        squin.broadcast.cz(a[:2], a[2:])

    with pytest.warns(CodeBlockWarning, match="does not support code blocks"):
        out = NativeToPlace(arch_spec=get_arch_spec()).emit(unmeasured, no_raise=False)
    assert _registers(out) == 0


def test_register_has_no_runtime_effect():
    from bloqade.pyqrack import StackMemorySimulator

    @squin.kernel.add(code_block)
    def flip_first():
        q = squin.qalloc(2)
        code_block.register(q)
        squin.x(q[0])
        return squin.broadcast.measure(q)

    result = StackMemorySimulator(min_qubits=2).run(flip_first)
    assert [int(r.value) for r in result] == [1, 0]
