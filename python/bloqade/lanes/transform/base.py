from __future__ import annotations

import abc
from dataclasses import dataclass

from kirin.ir.method import Method

from bloqade.lanes.utils import raise_if_statements_outside_dialect_group


@dataclass
class TransformABC(abc.ABC):
    """Base class for the ``lanes`` compilation stages.

    Every stage exposes ``emit(mt, no_raise=True) -> Method``. Subclasses
    implement the lowering itself in ``_emit``; ``emit`` wraps it with the
    checks that must hold for *any* stage.

    The wrapper's job is dialect-group hygiene. A stage typically ends with
    ``out.similar(out.dialects.discard(...))``, dropping the source dialect once
    its statements have been rewritten away. Kirin's ``Method.verify()`` does
    **not** validate dialect-group membership, so a statement the rewrite rules
    missed passes ``verify()``/``verify_type()`` cleanly and only surfaces
    lazily — as an ``InterpreterError``/``EncodingError`` far from its cause.
    ``emit`` therefore runs ``raise_if_statements_outside_dialect_group`` on the
    result whenever ``no_raise`` is ``False``, so a missed statement fails fast
    with an error naming the offending statement kinds and the stage that left
    them behind.

    The check is gated on ``no_raise`` for consistency with the rest of the
    stage: ``no_raise=True`` is the best-effort mode used by visualizers and
    partial-compilation callers, which tolerate incompletely lowered IR.

    If Kirin's ``verify()`` grows this check itself (QuEraComputing/kirin#685),
    this wrapper becomes redundant for the stages that already call ``verify()``.

    Subclasses that need to fail even earlier (before a later step garbles the
    error) can call ``raise_if_statements_outside_dialect_group`` inside
    ``_emit`` as well — see ``MoveToStackMove``, which checks before
    ``stackify``.
    """

    @abc.abstractmethod
    def _emit(self, mt: Method, no_raise: bool = True) -> Method:
        """Run the stage's lowering. Called by ``emit``; do not call directly."""
        ...

    def emit(self, mt: Method, no_raise: bool = True) -> Method:
        out = self._emit(mt, no_raise=no_raise)

        if not no_raise:
            raise_if_statements_outside_dialect_group(out, type(self).__name__)

        return out
