"""BytecodeDecoder -- syntactic Program -> stack_move ir.Method."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kirin import ir, types
from kirin.dialects import func

from bloqade.lanes.dialects import stack_move
from bloqade.lanes.layout.encoding import LaneAddress, LocationAddress, ZoneAddress

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

    def _visit_return(self, idx: int, instr: "Instruction") -> None:
        value = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.Return(value=value))

    def _visit_const_float(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstFloat(value=instr.float_value())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_int(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstInt(value=instr.int_value())
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_loc(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLoc(
            value=LocationAddress.from_inner(instr.location_address())
        )
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_lane(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLane(value=LaneAddress.from_inner(instr.lane_address()))
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_const_zone(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstZone(value=ZoneAddress.from_inner(instr.zone_address()))
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _pop_or_raise(self, idx: int, instr: "Instruction") -> ir.SSAValue:
        if not self.stack:
            raise LoweringError(
                idx, instr.op_name(), tuple(self.stack), "stack underflow"
            )
        return self.stack.pop()

    def _visit_pop(self, idx: int, instr: "Instruction") -> None:
        value = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.Pop(value=value))

    def _visit_dup(self, idx: int, instr: "Instruction") -> None:
        if not self.stack:
            raise LoweringError(idx, "dup", tuple(self.stack), "stack underflow")
        top = self.stack[-1]
        stmt = stack_move.Dup(value=top)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_swap(self, idx: int, instr: "Instruction") -> None:
        in_top = self._pop_or_raise(idx, instr)
        in_bot = self._pop_or_raise(idx, instr)
        stmt = stack_move.Swap(in_top=in_top, in_bot=in_bot)
        self.block.stmts.append(stmt)
        # Convention: top-of-stack last. out_bot ≡ in_top (goes below);
        # out_top ≡ in_bot (goes on top).
        self.stack.append(stmt.out_bot)
        self.stack.append(stmt.out_top)

    def _pop_n(self, idx: int, instr: "Instruction", n: int) -> list[ir.SSAValue]:
        """Pop n values from the stack, newest first. Returns them in
        bottom-to-top order so the caller can pass them as a tuple
        matching 'top-of-stack = last argument' convention."""
        if len(self.stack) < n:
            raise LoweringError(
                idx,
                instr.op_name(),
                tuple(self.stack),
                f"stack underflow (need {n}, have {len(self.stack)})",
            )
        popped = [self.stack.pop() for _ in range(n)]
        popped.reverse()  # now in bottom-to-top order
        return popped

    def _visit_initial_fill(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(stack_move.InitialFill(locations=tuple(locs)))

    def _visit_fill(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(stack_move.Fill(locations=tuple(locs)))

    def _visit_move(self, idx: int, instr: "Instruction") -> None:
        lanes = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(stack_move.Move(lanes=tuple(lanes)))

    def _visit_local_r(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops phi first (top of stack), then theta; after rename,
        # these map to axis_angle and rotation_angle respectively.
        axis_angle = self._pop_or_raise(idx, instr)
        rotation_angle = self._pop_or_raise(idx, instr)
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(
            stack_move.LocalR(
                axis_angle=axis_angle,
                rotation_angle=rotation_angle,
                locations=tuple(locs),
            )
        )

    def _visit_local_rz(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops theta (top of stack) -> rotation_angle after rename.
        rotation_angle = self._pop_or_raise(idx, instr)
        locs = self._pop_n(idx, instr, instr.arity())
        self.block.stmts.append(
            stack_move.LocalRz(rotation_angle=rotation_angle, locations=tuple(locs))
        )

    def _visit_global_r(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops phi first (top of stack), then theta; after rename,
        # these map to axis_angle and rotation_angle respectively.
        axis_angle = self._pop_or_raise(idx, instr)
        rotation_angle = self._pop_or_raise(idx, instr)
        self.block.stmts.append(
            stack_move.GlobalR(axis_angle=axis_angle, rotation_angle=rotation_angle)
        )

    def _visit_global_rz(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops theta (top of stack) -> rotation_angle after rename.
        rotation_angle = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.GlobalRz(rotation_angle=rotation_angle))

    def _visit_cz(self, idx: int, instr: "Instruction") -> None:
        zone = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.CZ(zone=zone))

    def _visit_measure(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        stmt = stack_move.Measure(locations=tuple(locs))
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_await_measure(self, idx: int, instr: "Instruction") -> None:
        # Treats await_measure as pure synchronisation — takes the future off
        # the top of the virtual stack, pushes it back so subsequent GetItem
        # calls can access measurement values. Verify the exact stack effect
        # against the Rust source and adjust if the bytecode actually pops
        # the future permanently.
        future = self._pop_or_raise(idx, instr)
        self.block.stmts.append(stack_move.AwaitMeasure(future=future))
        self.stack.append(future)

    def _visit_new_array(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.NewArray(
            type_tag=instr.type_tag(),
            dim0=instr.dim0(),
            dim1=instr.dim1(),
        )
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_get_item(self, idx: int, instr: "Instruction") -> None:
        ndims = instr.ndims()
        indices = self._pop_n(idx, instr, ndims)
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.GetItem(array=array, indices=tuple(indices))
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_set_detector(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.SetDetector(array=array)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_set_observable(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.SetObservable(array=array)
        self.block.stmts.append(stmt)
        self.stack.append(stmt.result)

    def _visit_halt(self, idx: int, instr: "Instruction") -> None:
        self.block.stmts.append(stack_move.Halt())

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
