"""BytecodeEncoder -- stack_move ir.Method → syntactic Program.

The encoder assumes the input ir.Method is *stack-consistent*: every
SSA value is used exactly once and the defining statements appear in
the order that satisfies the bytecode stack discipline (deepest arg
defined first, top-of-stack arg defined last).  ``load_program``
(``decode.py``) always produces stack-consistent IR, so round-trips
work unconditionally.  IR produced by compiler rewrites (e.g.
``RewriteMoveToStackMove``) may violate the ordering invariant and
must be normalised by the rewrite pass itself before encoding.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kirin import ir
from kirin.dialects import func

from bloqade.lanes.bytecode import Instruction, Program
from bloqade.lanes.dialects import stack_move


class EncodingError(Exception):
    """Raised when the encoder encounters an unrecognized statement."""

    def __init__(self, stmt: ir.Statement, reason: str | None = None) -> None:
        default = f"unrecognized statement type {type(stmt).__name__!r}"
        super().__init__(reason or default)


@dataclass
class BytecodeEncoder:
    """Turn a stack_move ir.Method into a bytecode Program.

    Walks the entry block's statements in order and emits the
    corresponding Instruction for each stack_move statement.
    func.ConstantNone is silently skipped; func.Return emits ``halt``
    when its argument is defined by func.ConstantNone and ``return_``
    otherwise.
    """

    instructions: list[Instruction] = field(default_factory=list)

    def encode(self, method: ir.Method, version: tuple[int, int] = (1, 0)) -> Program:
        block = method.callable_region.blocks[0]
        for stmt in block.stmts:
            self._visit(stmt)
        return Program(version=version, instructions=self.instructions)

    def _visit(self, stmt: ir.Statement) -> None:
        name = type(stmt).__name__
        handler = getattr(self, f"_visit_{name}", None)
        if handler is None:
            raise EncodingError(stmt)
        handler(stmt)

    # ── Constants ──────────────────────────────────────────────────────────

    def _visit_ConstFloat(self, stmt: stack_move.ConstFloat) -> None:
        self.instructions.append(Instruction.const_float(stmt.value))

    def _visit_ConstInt(self, stmt: stack_move.ConstInt) -> None:
        self.instructions.append(Instruction.const_int(stmt.value))

    def _visit_ConstLoc(self, stmt: stack_move.ConstLoc) -> None:
        v = stmt.value
        self.instructions.append(Instruction.const_loc(v.zone_id, v.word_id, v.site_id))

    def _visit_ConstLane(self, stmt: stack_move.ConstLane) -> None:
        v = stmt.value
        self.instructions.append(
            Instruction.const_lane(
                v.move_type, v.zone_id, v.word_id, v.site_id, v.bus_id, v.direction
            )
        )

    def _visit_ConstZone(self, stmt: stack_move.ConstZone) -> None:
        self.instructions.append(Instruction.const_zone(stmt.value.zone_id))

    # ── Stack manipulation ─────────────────────────────────────────────────

    def _visit_Pop(self, stmt: stack_move.Pop) -> None:
        self.instructions.append(Instruction.pop())

    def _visit_Dup(self, stmt: stack_move.Dup) -> None:
        self.instructions.append(Instruction.dup())

    def _visit_Swap(self, stmt: stack_move.Swap) -> None:
        self.instructions.append(Instruction.swap())

    # ── Atom operations ────────────────────────────────────────────────────

    def _visit_InitialFill(self, stmt: stack_move.InitialFill) -> None:
        self.instructions.append(Instruction.initial_fill(len(stmt.locations)))

    def _visit_Fill(self, stmt: stack_move.Fill) -> None:
        self.instructions.append(Instruction.fill(len(stmt.locations)))

    def _visit_Move(self, stmt: stack_move.Move) -> None:
        self.instructions.append(Instruction.move_(len(stmt.lanes)))

    # ── Gates ──────────────────────────────────────────────────────────────

    def _visit_LocalR(self, stmt: stack_move.LocalR) -> None:
        self.instructions.append(Instruction.local_r(len(stmt.locations)))

    def _visit_LocalRz(self, stmt: stack_move.LocalRz) -> None:
        self.instructions.append(Instruction.local_rz(len(stmt.locations)))

    def _visit_GlobalR(self, stmt: stack_move.GlobalR) -> None:
        self.instructions.append(Instruction.global_r())

    def _visit_GlobalRz(self, stmt: stack_move.GlobalRz) -> None:
        self.instructions.append(Instruction.global_rz())

    def _visit_CZ(self, stmt: stack_move.CZ) -> None:
        self.instructions.append(Instruction.cz())

    # ── Measurement ────────────────────────────────────────────────────────

    def _visit_Measure(self, stmt: stack_move.Measure) -> None:
        self.instructions.append(Instruction.measure(len(stmt.zones)))

    def _visit_AwaitMeasure(self, stmt: stack_move.AwaitMeasure) -> None:
        self.instructions.append(Instruction.await_measure())

    # ── Arrays ─────────────────────────────────────────────────────────────

    def _visit_NewArray(self, stmt: stack_move.NewArray) -> None:
        self.instructions.append(
            Instruction.new_array(stmt.type_tag, stmt.dim0, stmt.dim1)
        )

    def _visit_GetItem(self, stmt: stack_move.GetItem) -> None:
        self.instructions.append(Instruction.get_item(len(stmt.indices)))

    # ── Annotations ────────────────────────────────────────────────────────

    def _visit_SetDetector(self, stmt: stack_move.SetDetector) -> None:
        self.instructions.append(Instruction.set_detector())

    def _visit_SetObservable(self, stmt: stack_move.SetObservable) -> None:
        self.instructions.append(Instruction.set_observable())

    # ── Control flow (func dialect overlap) ────────────────────────────────

    def _visit_ConstantNone(self, stmt: func.ConstantNone) -> None:
        # Silently skip — ConstantNone is emitted only to support halt encoding
        # in the decoder (func.ConstantNone + func.Return pairs).
        pass

    def _visit_Return(self, stmt: func.Return) -> None:
        # halt was decoded as func.ConstantNone + func.Return(none_result).
        # Detect that case by checking whether the argument is defined by
        # func.ConstantNone; emit return_ for any other value.
        if (
            stmt.args
            and isinstance(stmt.args[0], ir.ResultValue)
            and isinstance(stmt.args[0].owner, func.ConstantNone)
        ):
            self.instructions.append(Instruction.halt())
        else:
            self.instructions.append(Instruction.return_())


def dump_program(method: ir.Method, version: tuple[int, int] = (1, 0)) -> Program:
    """Encode a stack_move ir.Method into a bytecode Program.

    Inverse of ``load_program`` in ``decode.py``:
    ``load_program(dump_program(method))`` round-trips through bytecode
    and back to an equivalent ir.Method.

    The *method* must be stack-consistent (each SSA value used exactly
    once, defining statements in stack order).  ``load_program`` always
    produces stack-consistent output; IR from compiler rewrites must be
    normalised before calling this function.
    """
    return BytecodeEncoder().encode(method, version)
