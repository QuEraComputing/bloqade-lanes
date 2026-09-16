"""Validate the qubit-register shape the post-unroll rewrite window assumes.

**Why this exists.** A rewrite in that window that reasons *per qubit* -- as
``EliminateRz`` does, tracking accumulated Z rotation on each atom -- has to
know which qubits a gate addresses. After ``AggressiveUnroll`` that is written
down directly: every gate's ``qubits`` operand is a ``py.ilist.new`` listing
them, so the rewrite reads the list rather than inferring anything.

Two ways that can fail, both structural and both knowable before any rewriting
starts:

**The operand is not an ``ilist.New``.** Then the addressed qubits are not
recoverable by inspection, and a per-qubit rewrite has nothing to key on. This
is the post-unroll shape not holding -- the rewrite is being run on IR from
somewhere else in the pipeline.

**The same qubit appears twice in one register.** No per-qubit state is well
defined for it: a gate cannot apply two different pending phases to one atom,
and there is no reading of the IR that says which was meant. That is malformed
IR rather than an unsupported shape.

Checking here keeps the rewrites free of the question. A rule that had to cope
would need a failure path through every handler that looks at qubits, and
could only report the problem after it had already been driven part-way
through the program. See ``lanes.flat_block.validation``, which covers the
control-flow half of the same window, and ``python/bloqade/lanes/rewrite/
eliminate_rz.py`` for the rewrite that relies on both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kirin import ir
from kirin.dialects import ilist
from kirin.validation import ValidationPass


@dataclass
class QubitRegisterValidation(ValidationPass):
    """Require every ``qubits`` operand to be an ``ilist.New`` of distinct qubits."""

    def name(self) -> str:
        return "lanes.qubit_register.validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        errors: list[ir.ValidationError] = []

        for stmt in method.callable_region.walk():
            register = getattr(stmt, "qubits", None)
            if not isinstance(register, ir.SSAValue):
                continue

            if not isinstance(register, ir.ResultValue) or not isinstance(
                register.owner, ilist.New
            ):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"{stmt.name}: qubit register does not come from "
                        "py.ilist.new, so the qubits it addresses cannot be "
                        "read off the IR. Post-unroll rewrites require that "
                        "shape.",
                    )
                )
                continue

            values = tuple(register.owner.values)
            if len(set(values)) != len(values):
                errors.append(
                    ir.ValidationError(
                        stmt,
                        f"{stmt.name} addresses a qubit more than once. No "
                        "per-qubit state is well defined for it -- a gate "
                        "cannot carry two pending phases for one atom. This "
                        "is malformed IR.",
                    )
                )

        return None, errors
