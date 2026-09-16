"""Validate the flat-block shape that time-ordered rewrites depend on.

**Why this exists.** A phase frame is a *physical* quantity: the Z rotation
accumulated on an atom, which persists for the whole shot and only makes sense
read in the order the hardware executes. ``EliminateRz`` therefore carries it
forward statement by statement, and is correct only if it is handed statements
in execution order.

``kirin.rewrite.Walk`` gives that order within a block, and nowhere else:

* It enqueues a region's blocks **reversed** under the default
  ``reverse=False``, so across two blocks the frame accumulates backwards --
  phase from the later block applied to gates in the earlier one.
* It reaches a nested function's **region** before the ``func.Function``
  statement that owns it, so a rule that resets per-region state has that state
  wiped part-way through the enclosing block.

Both are pinned by ``python/tests/rewrite/test_walk_order.py`` rather than
assumed, because they are kirin's behaviour and not ours.

Neither hazard produces broken-looking output. The emitted program is
structurally valid, verifies, and lowers -- it is just a different circuit than
the one the user wrote, differing by phases that a Z-basis measurement cannot
see. That is why this is a precondition checked *before* rewriting rather than
a condition a rule detects while running: by the time a rule could notice, the
frame it would need to report is already gone.

**Why a direct walk and not a ``Forward`` analysis**, unlike the
analysis-backed validations next door (``lanes.address.validation``,
``gemini.common.qubit.duplicates``). Those resolve *values* and need the
interpreter. This is a structural property of the IR, and a ``Forward`` pass
only visits reachable code -- an unreachable second block would slip through
the very check that exists to reject it. ``gemini.no_recursion`` is the
in-repo precedent for a structural check written this way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kirin import ir
from kirin.dialects import func
from kirin.validation import ValidationPass


@dataclass
class FlatBlockValidation(ValidationPass):
    """Require a single-block callable region with no nested functions."""

    def name(self) -> str:
        return "lanes.flat_block.validation"

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:
        errors: list[ir.ValidationError] = []

        statements = list(method.callable_region.walk())

        regions = [method.callable_region]
        for stmt in statements:
            regions.extend(stmt.regions)

        for region in regions:
            if len(region.blocks) > 1:
                errors.append(
                    ir.ValidationError(
                        region.parent_node or method.code,
                        f"Post-unroll rewrites require a single block per region, "
                        f"found {len(region.blocks)}. A rule carrying state in "
                        "execution order would accumulate it backwards here, "
                        "because Walk visits a region's blocks in reverse.",
                    )
                )

        for stmt in statements:
            if isinstance(stmt, func.Function) and stmt.parent_stmt is not None:
                errors.append(
                    ir.ValidationError(
                        stmt,
                        "Post-unroll rewrites require no nested func.Function. "
                        "Walk reaches a nested function's region before the "
                        "statement owning it, so a rule resetting per-region "
                        "state has it wiped part-way through the enclosing "
                        "block, before any handler can object.",
                    )
                )

        return None, errors
