"""Validate ``code_block.register`` statements after unrolling.

Runs in the post-unroll window of ``PhysicalNativeToPlace``, where every qubit
allocation is still a ``qubit.stmts.New`` or ``gemini.common.NewAt`` and every
register operand is a ``py.ilist.new`` of them. Each ``Register`` must name a
fixed list of distinct allocations, no qubit may belong to two blocks, a block
must fit in a word, and its qubits must be either all pinned or all unpinned. A
fully pinned block's pins must already have the block shape (one word,
contiguous, ``site = offset + position``): pins are authoritative, so the
compiler never moves one to repair a block.

Every violation is reported in one pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from kirin import ir
from kirin.dialects import ilist
from kirin.validation import ValidationPass

from bloqade import qubit
from bloqade.gemini.common.dialects.qubit import stmts as gemini_common_stmts
from bloqade.lanes.analysis.code_blocks import block_shape_error
from bloqade.lanes.dialects.code_block import Register
from bloqade.lanes.rewrite.circuit2place import _resolve_location_from_new_at

_Allocation = qubit.stmts.New | gemini_common_stmts.NewAt


def _allocation(value: ir.SSAValue) -> _Allocation | None:
    if isinstance(value, ir.ResultValue) and isinstance(
        value.owner, (qubit.stmts.New, gemini_common_stmts.NewAt)
    ):
        return value.owner
    return None


def _check_registers(
    method: ir.Method, sites_per_word: int | None
) -> list[ir.ValidationError]:
    errors: list[ir.ValidationError] = []
    block_of: dict[ir.SSAValue, int] = {}

    registers = [
        stmt for stmt in method.callable_region.walk() if isinstance(stmt, Register)
    ]
    for block_idx, reg in enumerate(registers):
        operand = reg.qubits
        if reg.parent_region is not method.callable_region or not (
            isinstance(operand, ir.ResultValue) and isinstance(operand.owner, ilist.New)
        ):
            errors.append(
                ir.ValidationError(
                    reg,
                    "code_block.register: the qubit list is not a fixed list of "
                    "qubit allocations after unrolling (is it inside control "
                    "flow?), so the block's members cannot be determined.",
                )
            )
            continue

        values = tuple(operand.owner.values)
        allocations = [_allocation(v) for v in values]
        if any(a is None for a in allocations):
            errors.append(
                ir.ValidationError(
                    reg,
                    "code_block.register: every element must be a qubit "
                    "allocated with qalloc or new_at.",
                )
            )
            continue
        if len(values) == 0:
            errors.append(
                ir.ValidationError(reg, "code_block.register: the block is empty.")
            )
            continue
        if len(set(values)) != len(values):
            errors.append(
                ir.ValidationError(
                    reg, "code_block.register: the same qubit appears twice."
                )
            )
        if any(v in block_of for v in values):
            errors.append(
                ir.ValidationError(
                    reg,
                    "code_block.register: a qubit in this block is already in "
                    "another code block. A qubit belongs to at most one block.",
                )
            )
        for v in values:
            block_of.setdefault(v, block_idx)
        if sites_per_word is not None and len(values) > sites_per_word:
            errors.append(
                ir.ValidationError(
                    reg,
                    f"code_block.register: the block has {len(values)} qubits but "
                    f"a word has only {sites_per_word} sites. A block must fit "
                    "in one word.",
                )
            )

        pinned = [isinstance(a, gemini_common_stmts.NewAt) for a in allocations]
        if any(pinned) and not all(pinned):
            errors.append(
                ir.ValidationError(
                    reg,
                    "code_block.register: some qubits in the block are pinned "
                    "with new_at and some are not. Pin all of them or none.",
                )
            )
        elif all(pinned):
            locations = [
                _resolve_location_from_new_at(a)
                for a in allocations
                if isinstance(a, gemini_common_stmts.NewAt)
            ]
            # Non-constant new_at arguments are reported by the address validators.
            resolved = [loc for loc in locations if loc is not None]
            if len(resolved) == len(locations):
                # Size is reported above, so check only word and contiguity here.
                reason = block_shape_error(resolved)
                if reason is not None:
                    errors.append(
                        ir.ValidationError(
                            reg,
                            "code_block.register: the pins do not form a code "
                            f"block: {reason}.",
                        )
                    )
    return errors


def get_code_block_validation(sites_per_word: int | None) -> type[ValidationPass]:
    """Build a ``CodeBlockValidation`` pass bound to ``sites_per_word``.

    ``ValidationSuite`` instantiates passes with no arguments, so the word size is
    closed over (see ``validation.address.get_validation``). ``None`` skips the
    size check.
    """

    @dataclass
    class CodeBlockValidation(ValidationPass):
        SITES_PER_WORD: ClassVar[int | None] = sites_per_word

        def name(self) -> str:
            return "lanes.code_block.validation"

        def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
            return None, _check_registers(method, self.SITES_PER_WORD)

    return CodeBlockValidation
