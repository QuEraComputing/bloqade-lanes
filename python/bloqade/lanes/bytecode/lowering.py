"""BytecodeDecoder -- syntactic Program -> stack_move ir.Method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kirin import ir, types
from kirin.dialects import func

from bloqade.lanes.dialects import stack_move

if TYPE_CHECKING:
    from bloqade.lanes.bytecode import Instruction, Program


@dataclass
class LoweringError(Exception):
    """Raised when the decoder fails.

    Carries the offending instruction's index, opcode, and a snapshot of
    the virtual stack of SSA values at the point of failure.
    """

    instruction_index: int
    opcode_name: str
    stack_snapshot: tuple[ir.SSAValue, ...]
    reason: str

    def __str__(self) -> str:
        return (
            f"LoweringError at instruction {self.instruction_index} "
            f"({self.opcode_name}): {self.reason} "
            f"[stack depth={len(self.stack_snapshot)}]"
        )


def _make_empty_block() -> ir.Block:
    """Create a body block with a single self argument.

    ``ir.Method.__init__`` asserts that the callable region's entry block
    has at least one argument (the self reference). We satisfy that here
    while keeping the block free of statements, so downstream handlers can
    append ops without having to worry about block-argument bookkeeping.
    """

    return ir.Block(argtypes=(types.MethodType,))


@dataclass
class BytecodeDecoder:
    """Turn a bytecode Program into a stack_move ir.Method.

    Maintains a virtual stack of SSA values during decoding: each bytecode
    push emits a stack_move statement whose result is pushed onto the
    virtual stack, and each pop consumes the top SSA reference. Stack ops
    (Pop/Dup/Swap) emit corresponding stack_move statements (linear-IR
    style -- see the design doc).
    """

    stack: list[ir.SSAValue] = field(default_factory=list)
    block: ir.Block = field(default_factory=_make_empty_block)

    def decode(self, program: "Program", kernel_name: str = "main") -> ir.Method:
        for idx, instr in enumerate(program.instructions):
            self._visit(idx, instr)
        return self._finalize(kernel_name)

    def _visit(self, idx: int, instr: "Instruction") -> None:
        name = instr.op_name()
        handler = getattr(self, f"_visit_{name}", None)
        if handler is None:
            raise LoweringError(idx, name, tuple(self.stack), "unknown opcode")
        handler(idx, instr)

    def _visit_return_(self, idx: int, instr: "Instruction") -> None:
        self.block.stmts.append(stack_move.Return())

    def _visit_const_float(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstFloat(value=instr.float_value())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_int(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstInt(value=instr.int_value())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_loc(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLoc(value=instr.location_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_lane(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLane(value=instr.lane_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_zone(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstZone(value=instr.zone_address())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _finalize(self, kernel_name: str) -> ir.Method:
        region = ir.Region(blocks=self.block)
        function = func.Function(
            sym_name=kernel_name,
            signature=func.Signature((), types.NoneType),
            slots=(),
            body=region,
        )
        dialects = ir.DialectGroup([stack_move.dialect, func.dialect])
        return ir.Method(
            dialects=dialects,
            code=function,
            sym_name=kernel_name,
            arg_names=[],
        )


def load_program(program: "Program", kernel_name: str = "main") -> ir.Method:
    """Decode a bytecode Program into a stack_move ir.Method."""
    decoder = BytecodeDecoder()
    return decoder.decode(program, kernel_name)
