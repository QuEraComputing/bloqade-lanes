"""Reject static calls that survived the inlining passes.

A Gemini kernel lowers to a fixed sequence of physical operations, so by the
time the ``verify`` suite runs the IR has to be flat: every callee spliced in,
every loop unrolled. A surviving static call is a frame boundary the lowering
cannot cross, and the passes downstream neither inline it nor complain about it
-- they walk straight past and fail much later, far from the cause.

Two shapes are deliberately *not* reported here:

- **Dynamic calls** (``func.Call``). Their target is unknown by construction, so
  they are the business of ``NoOpaqueCallValidation``, which rejects the subset
  that is genuinely unanalysable. Flagging the rest was tried and is not viable:
  after inlining, an ordinary stdlib kernel calls a self value belonging to an
  inlined region rather than to the method being scanned, so a blanket rule
  stops the physical pipeline compiling. See ``recursion.NoOpaqueCallValidation``.
- **``ilist`` higher-order statements** (``Map``, ``Foldl``, ``Foldr``, ``Scan``,
  ``ForEach``). These do invoke their ``fn`` argument, but they are functional
  control flow the lowering handles rather than a frame boundary. They carry no
  ``ir.StaticCall`` trait, so a trait-driven walk skips them without needing to
  name them.

The walk is syntactic rather than a dataflow impl on purpose. This check used to
live in ``GeminiLogicalValidation`` as a ``func.Invoke`` impl, but that analysis
is a ``Forward`` pass whose ``scf.For`` impl returns bottom without descending
into the loop body -- so an invoke nested inside a loop was never visited and
never reported. ``walk()`` recurses through every region unconditionally.

Keying on the ``ir.StaticCall`` trait rather than ``isinstance(stmt, func.Invoke)``
means any dialect that declares the trait participates, which is the same choice
``recursion.CallGraph`` makes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kirin import ir
from kirin.validation import ValidationPass

UNROLL_HELP = (
    "decorate the calling kernel with `aggressive_unroll=True` to inline and "
    "unroll the call, or with `verify=False` to skip these checks"
)


def _callee_name(stmt: ir.Statement) -> str:
    trait = stmt.get_trait(ir.StaticCall)
    if trait is None:  # pragma: no cover - callers filter on the trait first
        return "<unknown>"

    return trait.get_callee(stmt).sym_name or "<lambda>"


@dataclass
class NoStaticCallValidation(ValidationPass):
    """Report every static call left in the IR once the passes have run."""

    def name(self) -> str:
        return "Gemini Static Call Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        errors = [
            ir.ValidationError(
                stmt,
                f"call to '{_callee_name(stmt)}' was not inlined; Gemini kernels "
                "must be flat, so every call has to be resolved at compile time",
                help=UNROLL_HELP,
            )
            for stmt in method.code.walk()
            if stmt.get_trait(ir.StaticCall) is not None
        ]
        return None, errors
