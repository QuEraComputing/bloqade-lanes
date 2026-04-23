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


def _make_empty_block() -> ir.Block:
    """Create a body block with a single self argument.

    ``ir.Method.__init__`` asserts that the callable region's entry block
    has at least one argument (the self reference). We satisfy that here
    while keeping the block free of statements, so downstream handlers can
    append ops without having to worry about block-argument bookkeeping.
    """

    return ir.Block(argtypes=(types.MethodType,))


@dataclass
class StackMachineFrame:
    """Virtual stack + current block during bytecode -> stack_move lowering.

    Mirrors the role of ``kirin.lowering.Frame`` but specialised for the
    simpler bytecode decoder use case: a single basic block, a
    monotonically-growing stack of SSA values.
    """

    block: ir.Block = field(default_factory=_make_empty_block)
    stack: list[ir.SSAValue] = field(default_factory=list)

    # -- Statement emission --

    def append(self, stmt: ir.Statement) -> None:
        """Append a statement to the block."""
        self.block.stmts.append(stmt)

    # -- Stack manipulation --

    def push_value(self, value: ir.SSAValue) -> None:
        """Push an SSA value onto the virtual stack."""
        self.stack.append(value)

    def pop_value(self) -> ir.SSAValue:
        """Pop the top of the virtual stack. Raises IndexError on underflow."""
        return self.stack.pop()

    def peek_value(self) -> ir.SSAValue:
        """Return the top of the stack without popping. Raises on empty."""
        return self.stack[-1]

    def swap_values(self) -> None:
        """Swap the top two values on the stack in place."""
        self.stack[-1], self.stack[-2] = self.stack[-2], self.stack[-1]

    def snapshot(self) -> tuple[ir.SSAValue, ...]:
        """Return a tuple snapshot of the stack -- used for error reporting."""
        return tuple(self.stack)

    def depth(self) -> int:
        return len(self.stack)


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


@dataclass
class BytecodeDecoder:
    """Turn a bytecode Program into a stack_move ir.Method.

    Maintains a virtual stack of SSA values during decoding: each bytecode
    push emits a stack_move statement whose result is pushed onto the
    virtual stack, and each pop consumes the top SSA reference. Stack ops
    (Pop/Dup/Swap) emit corresponding stack_move statements (linear-IR
    style -- see the design doc).

    All stack and IR manipulation is delegated to ``self.frame``; the
    handlers below are thin translators from bytecode opcodes to
    stack_move statements.
    """

    frame: StackMachineFrame = field(default_factory=StackMachineFrame)

    def decode(self, program: "Program", kernel_name: str = "main") -> ir.Method:
        for idx, instr in enumerate(program.instructions):
            self._visit(idx, instr)
        return self._finalize(kernel_name)

    def _visit(self, idx: int, instr: "Instruction") -> None:
        name = instr.op_name()
        handler = getattr(self, f"_visit_{name}", None)
        if handler is None:
            raise LoweringError(idx, name, self.frame.snapshot(), "unknown opcode")
        handler(idx, instr)

    def _visit_return(self, idx: int, instr: "Instruction") -> None:
        value = self._pop_or_raise(idx, instr)
        self.frame.append(stack_move.Return(value=value))

    def _visit_const_float(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstFloat(value=instr.float_value())
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_const_int(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstInt(value=instr.int_value())
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_const_loc(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLoc(
            value=LocationAddress.from_inner(instr.location_address())
        )
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_const_lane(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstLane(value=LaneAddress.from_inner(instr.lane_address()))
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_const_zone(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.ConstZone(value=ZoneAddress.from_inner(instr.zone_address()))
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _pop_or_raise(self, idx: int, instr: "Instruction") -> ir.SSAValue:
        if self.frame.depth() == 0:
            raise LoweringError(
                idx, instr.op_name(), self.frame.snapshot(), "stack underflow"
            )
        return self.frame.pop_value()

    def _visit_pop(self, idx: int, instr: "Instruction") -> None:
        value = self._pop_or_raise(idx, instr)
        self.frame.append(stack_move.Pop(value=value))

    def _visit_dup(self, idx: int, instr: "Instruction") -> None:
        if self.frame.depth() == 0:
            raise LoweringError(idx, "dup", self.frame.snapshot(), "stack underflow")
        top = self.frame.peek_value()
        stmt = stack_move.Dup(value=top)
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_swap(self, idx: int, instr: "Instruction") -> None:
        in_top = self._pop_or_raise(idx, instr)
        in_bot = self._pop_or_raise(idx, instr)
        stmt = stack_move.Swap(in_top=in_top, in_bot=in_bot)
        self.frame.append(stmt)
        # Convention: top-of-stack last. out_bot ≡ in_top (goes below);
        # out_top ≡ in_bot (goes on top).
        self.frame.push_value(stmt.out_bot)
        self.frame.push_value(stmt.out_top)

    def _pop_n(self, idx: int, instr: "Instruction", n: int) -> list[ir.SSAValue]:
        """Pop n values from the stack, newest first. Returns them in
        bottom-to-top order so the caller can pass them as a tuple
        matching 'top-of-stack = last argument' convention."""
        if self.frame.depth() < n:
            raise LoweringError(
                idx,
                instr.op_name(),
                self.frame.snapshot(),
                f"stack underflow (need {n}, have {self.frame.depth()})",
            )
        popped = [self.frame.pop_value() for _ in range(n)]
        popped.reverse()  # now in bottom-to-top order
        return popped

    def _visit_initial_fill(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        self.frame.append(stack_move.InitialFill(locations=tuple(locs)))

    def _visit_fill(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        self.frame.append(stack_move.Fill(locations=tuple(locs)))

    def _visit_move(self, idx: int, instr: "Instruction") -> None:
        lanes = self._pop_n(idx, instr, instr.arity())
        self.frame.append(stack_move.Move(lanes=tuple(lanes)))

    def _visit_local_r(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops phi first (top of stack), then theta; after rename,
        # these map to axis_angle and rotation_angle respectively.
        axis_angle = self._pop_or_raise(idx, instr)
        rotation_angle = self._pop_or_raise(idx, instr)
        locs = self._pop_n(idx, instr, instr.arity())
        self.frame.append(
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
        self.frame.append(
            stack_move.LocalRz(rotation_angle=rotation_angle, locations=tuple(locs))
        )

    def _visit_global_r(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops phi first (top of stack), then theta; after rename,
        # these map to axis_angle and rotation_angle respectively.
        axis_angle = self._pop_or_raise(idx, instr)
        rotation_angle = self._pop_or_raise(idx, instr)
        self.frame.append(
            stack_move.GlobalR(axis_angle=axis_angle, rotation_angle=rotation_angle)
        )

    def _visit_global_rz(self, idx: int, instr: "Instruction") -> None:
        # bytecode pops theta (top of stack) -> rotation_angle after rename.
        rotation_angle = self._pop_or_raise(idx, instr)
        self.frame.append(stack_move.GlobalRz(rotation_angle=rotation_angle))

    def _visit_cz(self, idx: int, instr: "Instruction") -> None:
        zone = self._pop_or_raise(idx, instr)
        self.frame.append(stack_move.CZ(zone=zone))

    def _visit_measure(self, idx: int, instr: "Instruction") -> None:
        locs = self._pop_n(idx, instr, instr.arity())
        stmt = stack_move.Measure(locations=tuple(locs))
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_await_measure(self, idx: int, instr: "Instruction") -> None:
        # Treats await_measure as pure synchronisation — takes the future off
        # the top of the virtual stack, pushes it back so subsequent GetItem
        # calls can access measurement values. Verify the exact stack effect
        # against the Rust source and adjust if the bytecode actually pops
        # the future permanently.
        future = self._pop_or_raise(idx, instr)
        self.frame.append(stack_move.AwaitMeasure(future=future))
        self.frame.push_value(future)

    def _visit_new_array(self, idx: int, instr: "Instruction") -> None:
        stmt = stack_move.NewArray(
            type_tag=instr.type_tag(),
            dim0=instr.dim0(),
            dim1=instr.dim1(),
        )
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_get_item(self, idx: int, instr: "Instruction") -> None:
        ndims = instr.ndims()
        indices = self._pop_n(idx, instr, ndims)
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.GetItem(array=array, indices=tuple(indices))
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_set_detector(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.SetDetector(array=array)
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_set_observable(self, idx: int, instr: "Instruction") -> None:
        array = self._pop_or_raise(idx, instr)
        stmt = stack_move.SetObservable(array=array)
        self.frame.append(stmt)
        self.frame.push_value(stmt.result)

    def _visit_halt(self, idx: int, instr: "Instruction") -> None:
        self.frame.append(stack_move.Halt())

    def _finalize(self, kernel_name: str) -> ir.Method:
        region = ir.Region(blocks=self.frame.block)
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
