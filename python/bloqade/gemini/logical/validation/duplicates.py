"""Cross-statement validation: each NewAt's address must be unique."""

from dataclasses import dataclass
from typing import Any

from kirin import ir
from kirin.analysis import const
from kirin.validation import ValidationPass

from bloqade.gemini.logical.dialects.operations import stmts


@dataclass
class DuplicateAddressValidation(ValidationPass):
    """Report any pair of gemini.operations.NewAt statements that pin the
    same physical address.

    Pre-condition: per-statement validation (E1's const-foldability + range)
    has already run, so every NewAt's args have const hints populated and are
    int-typed. NewAt statements that don't satisfy this are skipped — E1
    already emitted errors for them.
    """

    def name(self) -> str:
        return "gemini.new_at.duplicates"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        # Lazy import to avoid circular dependency with bloqade.lanes
        from bloqade.lanes.layout.encoding import LocationAddress

        seen: dict[LocationAddress, stmts.NewAt] = {}
        errors: list[ir.ValidationError] = []

        for stmt in method.callable_region.walk():
            if not isinstance(stmt, stmts.NewAt):
                continue

            z = _const_int(stmt.zone_id)
            w = _const_int(stmt.word_id)
            s = _const_int(stmt.site_id)
            if z is None or w is None or s is None:
                # Per-statement validation already errored; skip silently here.
                continue

            addr = LocationAddress(word_id=w, site_id=s, zone_id=z)
            existing = seen.get(addr)
            if existing is not None:
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"address (zone={z}, word={w}, site={s}) is pinned by two "
                        f"operations.new_at calls",
                    )
                )
            else:
                seen[addr] = stmt

        return None, errors


def _const_int(value: ir.SSAValue) -> int | None:
    hint = value.hints.get("const")
    if not isinstance(hint, const.Value) or not isinstance(hint.data, int):
        return None
    return hint.data
