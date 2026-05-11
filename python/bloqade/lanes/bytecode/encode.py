"""BytecodeEncoder -- stack_move ir.Method → syntactic Program.

The encoder assumes the input ir.Method is *stack-consistent*: every
SSA value is used exactly once and the defining statements appear in
the order that satisfies the bytecode stack discipline (deepest arg
defined first, top-of-stack arg defined last).  ``load_program``
(``decode.py``) always produces stack-consistent IR, so round-trips
work unconditionally.  IR produced by compiler rewrites (e.g.
``RewriteMoveToStackMove``) may violate the ordering invariant and
must be normalised by ``stackify`` before encoding.

Implemented as a kirin ``EmitABC`` pass: each dialect registers its
own ``MethodTable`` under the key ``"emit.bytecode"``, and the encoder
dispatches via the standard kirin interpreter machinery.  The encoded
instructions accumulate in ``BytecodeEncoder.instructions``; call
``dump_program`` for the one-shot public API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kirin import ir
from kirin.dialects import func
from kirin.emit import EmitABC, EmitFrame
from kirin.interp import MethodTable, impl

from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.dialects import stack_move


class EncodingError(Exception):
    """Raised when the encoder encounters an unrecognized statement."""

    def __init__(self, stmt: ir.Statement, reason: str | None = None) -> None:
        default = f"unrecognized statement type {type(stmt).__name__!r}"
        super().__init__(reason or default)


@dataclass
class BytecodeEncoder(EmitABC[EmitFrame, Program]):
    """Turn a stack_move ir.Method into a bytecode Program.

    Constructed with the method's dialect group
    (``BytecodeEncoder(dialects=method.dialects)``).  Call ``run(method)``
    to populate ``self.instructions``, then build the ``Program`` from them.
    Prefer ``dump_program`` for the one-shot public API.
    """

    keys = ("emit.bytecode",)
    void = Program(version=(1, 0), instructions=[])

    instructions: list[Instruction] = field(default_factory=list)

    def initialize_frame(
        self, node: ir.Statement, *, has_parent_access: bool = False
    ) -> EmitFrame:
        return EmitFrame(node, has_parent_access=has_parent_access)

    def reset(self) -> None:
        self.instructions = []

    def eval_fallback(self, frame: EmitFrame, node: ir.Statement) -> tuple:  # type: ignore[override]
        raise EncodingError(node)


@stack_move.dialect.register(key="emit.bytecode")
class _StackMoveEmit(MethodTable):

    # ── Constants ──────────────────────────────────────────────────────────

    @impl(stack_move.ConstFloat)
    def const_float(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.ConstFloat
    ) -> tuple:
        emit.instructions.append(Instruction.const_float(stmt.value))
        return ()

    @impl(stack_move.ConstInt)
    def const_int(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.ConstInt
    ) -> tuple:
        emit.instructions.append(Instruction.const_int(stmt.value))
        return ()

    @impl(stack_move.ConstLoc)
    def const_loc(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.ConstLoc
    ) -> tuple:
        v = stmt.value
        emit.instructions.append(Instruction.const_loc(v.zone_id, v.word_id, v.site_id))
        return ()

    @impl(stack_move.ConstLane)
    def const_lane(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.ConstLane
    ) -> tuple:
        v = stmt.value
        emit.instructions.append(
            Instruction.const_lane(
                v.move_type, v.zone_id, v.word_id, v.site_id, v.bus_id, v.direction
            )
        )
        return ()

    @impl(stack_move.ConstZone)
    def const_zone(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.ConstZone
    ) -> tuple:
        emit.instructions.append(Instruction.const_zone(stmt.value.zone_id))
        return ()

    # ── Stack manipulation ─────────────────────────────────────────────────

    @impl(stack_move.Pop)
    def pop(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.Pop
    ) -> tuple:
        emit.instructions.append(Instruction.pop())
        return ()

    @impl(stack_move.Dup)
    def dup(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.Dup
    ) -> tuple:
        emit.instructions.append(Instruction.dup())
        return ()

    @impl(stack_move.Swap)
    def swap(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.Swap
    ) -> tuple:
        emit.instructions.append(Instruction.swap())
        return ()

    # ── Atom operations ────────────────────────────────────────────────────

    @impl(stack_move.InitialFill)
    def initial_fill(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.InitialFill
    ) -> tuple:
        emit.instructions.append(Instruction.initial_fill(len(stmt.locations)))
        return ()

    @impl(stack_move.Fill)
    def fill(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.Fill
    ) -> tuple:
        emit.instructions.append(Instruction.fill(len(stmt.locations)))
        return ()

    @impl(stack_move.Move)
    def move(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.Move
    ) -> tuple:
        emit.instructions.append(Instruction.move_(len(stmt.lanes)))
        return ()

    # ── Gates ──────────────────────────────────────────────────────────────

    @impl(stack_move.LocalR)
    def local_r(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.LocalR
    ) -> tuple:
        emit.instructions.append(Instruction.local_r(len(stmt.locations)))
        return ()

    @impl(stack_move.LocalRz)
    def local_rz(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.LocalRz
    ) -> tuple:
        emit.instructions.append(Instruction.local_rz(len(stmt.locations)))
        return ()

    @impl(stack_move.GlobalR)
    def global_r(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.GlobalR
    ) -> tuple:
        emit.instructions.append(Instruction.global_r())
        return ()

    @impl(stack_move.GlobalRz)
    def global_rz(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.GlobalRz
    ) -> tuple:
        emit.instructions.append(Instruction.global_rz())
        return ()

    @impl(stack_move.CZ)
    def cz(self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.CZ) -> tuple:
        emit.instructions.append(Instruction.cz())
        return ()

    # ── Measurement ────────────────────────────────────────────────────────

    @impl(stack_move.Measure)
    def measure(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.Measure
    ) -> tuple:
        emit.instructions.append(Instruction.measure(len(stmt.zones)))
        return ()

    @impl(stack_move.AwaitMeasure)
    def await_measure(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.AwaitMeasure
    ) -> tuple:
        emit.instructions.append(Instruction.await_measure())
        return ()

    # ── Arrays ─────────────────────────────────────────────────────────────

    @impl(stack_move.NewArray)
    def new_array(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.NewArray
    ) -> tuple:
        emit.instructions.append(
            Instruction.new_array(stmt.type_tag, stmt.dim0, stmt.dim1)
        )
        return ()

    @impl(stack_move.GetItem)
    def get_item(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.GetItem
    ) -> tuple:
        emit.instructions.append(Instruction.get_item(len(stmt.indices)))
        return ()

    # ── Annotations ────────────────────────────────────────────────────────

    @impl(stack_move.SetDetector)
    def set_detector(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.SetDetector
    ) -> tuple:
        emit.instructions.append(Instruction.set_detector())
        return ()

    @impl(stack_move.SetObservable)
    def set_observable(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: stack_move.SetObservable
    ) -> tuple:
        emit.instructions.append(Instruction.set_observable())
        return ()


@func.dialect.register(key="emit.bytecode")
class _FuncEmit(MethodTable):

    @impl(func.Function)
    def function(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: func.Function
    ) -> tuple:
        for block in stmt.body.blocks:
            for s in block.stmts:
                emit.frame_eval(frame, s)
        return ()

    @impl(func.ConstantNone)
    def constant_none(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: func.ConstantNone
    ) -> tuple:
        return ()

    @impl(func.Return)
    def return_(
        self, emit: BytecodeEncoder, frame: EmitFrame, stmt: func.Return
    ) -> tuple:
        if (
            stmt.args
            and isinstance(stmt.args[0], ir.ResultValue)
            and isinstance(stmt.args[0].owner, func.ConstantNone)
        ):
            emit.instructions.append(Instruction.halt())
        else:
            emit.instructions.append(Instruction.return_())
        return ()


def dump_program(method: ir.Method, version: tuple[int, int] = (1, 0)) -> Program:
    """Encode a stack_move ir.Method into a bytecode Program.

    Inverse of ``load_program`` in ``decode.py``:
    ``load_program(dump_program(method))`` round-trips through bytecode
    and back to an equivalent ir.Method.

    The *method* must be stack-consistent (each SSA value used exactly
    once, defining statements in stack order).  ``load_program`` always
    produces stack-consistent output; IR from compiler rewrites must be
    normalised by ``stackify`` before calling this function.
    """
    encoder = BytecodeEncoder(dialects=method.dialects)
    encoder.run(method)
    return Program(version=version, instructions=encoder.instructions)
