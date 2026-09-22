"""Reject the experimental extension statements on paths that cannot support them.

The statements in :mod:`bloqade.gemini.logical.dialects.extensions` are part of
the ``@gemini.logical.kernel`` dialect group, so they lower and compile like any
other logical statement. What sets them apart is that they are *not* logical
operations: ``StarRz`` puts physical Z rotations on the three qubits of a
weight-3 logical-Z representative, which takes the state out of the code space.
The state only comes back if the program post-selects on the error-correction
checks around it -- something the compiler neither inserts nor can verify.

That makes them usable, but only deliberately: in simulation, where the user can
post-select on the detectors themselves. This pass is the guard for everywhere
else. It is deliberately **not** part of ``@gemini.logical.kernel``'s own
validation suite -- putting it there would reject the feature at definition
time, everywhere, which is the opposite of what a gated experimental feature
needs. It is added to the suites that submit a program for execution instead,
and adding it to a suite is what turns the guard on.

The check is on the *dialect*, not on a list of statement types, so a second
extension is covered the day it is added rather than the day someone remembers
to update this file.

One error per offending statement, so a program with three uses reports all
three -- the same report-everything contract the validations next door keep.

Each error is anchored to the statement, like the validations next door. The
user-facing wrappers are `@kernel` stdlib kernels, so by the time any suite runs
the statement has been inlined and its source location points inside
`bloqade.gemini.logical.extensions` rather than at the call site. The message
names the statement for that reason, rather than relying on the location.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kirin import ir
from kirin.validation import ValidationPass

from ..dialects.extensions import dialect as extensions_dialect

EXPERIMENTAL_HELP = (
    "this is an experimental gadget that acts on the physical qubits inside a "
    "logical qubit, leaving the logical subspace; it is supported only where "
    "the caller can post-select on the surrounding error-correction checks, "
    "e.g. `GeminiLogicalSimulator`"
)


@dataclass
class NoLogicalExtensionsValidation(ValidationPass):
    """Reject any use of a ``gemini.logical.extensions`` statement."""

    def name(self) -> str:
        return "Gemini Logical Extensions Validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        kernel = method.sym_name or "<lambda>"

        return None, [
            ir.ValidationError(
                stmt,
                f"kernel '{kernel}' uses the experimental statement "
                f"'{extensions_dialect.name}.{stmt.name}', which is not "
                "supported on this backend",
                help=EXPERIMENTAL_HELP,
            )
            for stmt in method.callable_region.walk()
            if stmt.dialect is extensions_dialect
        ]
