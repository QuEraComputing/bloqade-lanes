"""Turn ``code_block.register`` statements into per-qubit ``CodeBlockTag``s.

``resolve_code_blocks`` runs after ``RewriteQubitsToPinnedQubits`` on IR that
``CodeBlockValidation`` has accepted. Block ids follow program order of the
``Register`` statements. ``strip_code_blocks`` deletes registrations without
recording them, for the opt-out and for lowerings that do not support blocks.
"""

from __future__ import annotations

from kirin import ir
from kirin.dialects import ilist

from bloqade.lanes.analysis.code_blocks import CodeBlockTag
from bloqade.lanes.dialects import place
from bloqade.lanes.dialects.code_block import Register


def _registers(method: ir.Method) -> list[Register]:
    return [
        stmt for stmt in method.callable_region.walk() if isinstance(stmt, Register)
    ]


def has_code_blocks(method: ir.Method) -> bool:
    return any(isinstance(s, Register) for s in method.callable_region.walk())


def resolve_code_blocks(method: ir.Method) -> int:
    """Stamp ``code_block`` on each member's ``NewPinnedQubit``; return the count."""
    registers = _registers(method)
    for block_id, reg in enumerate(registers):
        operand = reg.qubits
        assert isinstance(operand, ir.ResultValue) and isinstance(
            operand.owner, ilist.New
        ), "resolve_code_blocks requires validated IR"
        for position, value in enumerate(operand.owner.values):
            owner = value.owner if isinstance(value, ir.ResultValue) else None
            assert isinstance(
                owner, place.NewPinnedQubit
            ), "resolve_code_blocks must run after RewriteQubitsToPinnedQubits"
            owner.code_block = CodeBlockTag(block=block_id, position=position)
        reg.delete()
    return len(registers)


def strip_code_blocks(method: ir.Method) -> int:
    """Delete every ``Register`` without recording it; return how many there were."""
    registers = _registers(method)
    for reg in registers:
        reg.delete()
    return len(registers)
