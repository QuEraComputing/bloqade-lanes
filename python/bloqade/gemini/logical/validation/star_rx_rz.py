"""Reject STAR-X with logical gates that lower to a native Rz."""

from dataclasses import dataclass
from typing import Any

from bloqade.squin import gate
from kirin import ir
from kirin.dialects import func, ilist
from kirin.validation import ValidationPass

from bloqade.gemini.logical.dialects.extensions import stmts as extensions

_ILIST_CALLS = (ilist.Map, ilist.Foldl, ilist.Foldr, ilist.Scan, ilist.ForEach)


def _constant_method(value: ir.SSAValue) -> ir.Method | None:
    """Resolve a compile-time method operand without treating inert data as a call."""
    attribute = getattr(value.owner, "value", None)
    if attribute is None:
        return None
    data = attribute.unwrap() if hasattr(attribute, "unwrap") else attribute
    return data if isinstance(data, ir.Method) else None


@dataclass
class StarRxRzGateValidation(ValidationPass):
    """Conservatively reject STAR-X with any Rz-producing logical gate."""

    def name(self) -> str:
        return "Gemini Logical StarRx Rz Gate Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        statements: list[ir.Statement] = []
        pending = [method]
        visited: set[int] = set()
        while pending:
            current = pending.pop()
            if id(current) in visited:
                continue
            visited.add(id(current))
            for stmt in current.callable_region.walk():
                statements.append(stmt)
                if trait := stmt.get_trait(ir.StaticCall):
                    pending.append(trait.get_callee(stmt))
                elif isinstance(stmt, func.Call):
                    if callee := _constant_method(stmt.callee):
                        pending.append(callee)
                elif isinstance(stmt, _ILIST_CALLS) and (
                    callee := _constant_method(stmt.fn)
                ):
                    pending.append(callee)

        star_rx = [stmt for stmt in statements if isinstance(stmt, extensions.StarRx)]
        conflicting_gates = sorted(
            {
                type(stmt).__name__
                for stmt in statements
                if isinstance(
                    stmt,
                    (gate.stmts.Rz, gate.stmts.Z, gate.stmts.S, gate.stmts.H),
                )
            }
        )

        if not conflicting_gates:
            return None, []

        conflicts = ", ".join(conflicting_gates)
        return None, [
            ir.ValidationError(
                stmt,
                f"StarRx cannot be combined with Rz-producing SQUIN gates: "
                f"{conflicts}. This restriction is conservative and applies "
                "regardless of gate order or qubit overlap.",
            )
            for stmt in star_rx
        ]
