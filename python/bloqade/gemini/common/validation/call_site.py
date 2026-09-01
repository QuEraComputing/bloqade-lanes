"""Attribute validation errors back to the call that dragged them in.

Inlining is lossy in exactly the way that matters for diagnostics. ``InlinePass``
splices a callee's body into the caller and the invoke disappears, but the
spliced statements keep the *callee's* source info. If one of them then fails
validation, the error names a file the user never opened::

    Gemini Logical Validation:
      - Non-constant iterable in for loop is not supported ...
          File ".../gemini/logical/stdlib/post_processing.py", line 12, col 65
        │  return logical.default_post_processing(qbs)

and the excerpt is a chimera. ``ValidationError.attach`` overwrites
``source.lineno_begin`` with the *entry* method's and resolves the source lines
against the entry's ``py_func``, so the reader gets the callee's filename against
the caller's line numbers and the caller's code -- line 12 above is in the
caller, not in ``post_processing.py``. When the callee's line number falls past
the end of the caller, which is the usual case, the excerpt comes out empty and
the diagnostic is a bare filename. That is bloqade-internal#449.

The lowering is not wrong to reject the program: ``default_post_processing``
loops over ``range(1, len(register))``, which only ``aggressive_unroll=True`` can
flatten. What is missing is *which call* introduced the offending code and what
to do about it. So ``InlineOrigins`` records the call sites before inlining
destroys them, and afterwards folds them back into the messages and re-points the
excerpt at the kernel the statement actually came from.

A statement is identified with the kernel it was written in by
``(source.file, source.lineno_begin)`` -- filename plus that kernel's offset into
it. File alone is not enough: a helper defined a few lines above the kernel that
calls it is the common case, and it shares the caller's filename while needing
exactly the same attribution.

Which *call* to blame comes from the static call graph: every method reachable
from the entry inherits the call site of the first hop that reaches it,
breadth-first so the shallowest -- the one written in the user's own kernel --
wins. That is coarse enough to survive nested inlining and precise enough to name
the one call the user can change.
"""

from __future__ import annotations

import inspect
import textwrap
from collections import deque
from dataclasses import dataclass, field

from kirin import ir
from kirin.source import SourceInfo

# NOTE: not re-exported from `kirin.validation`, which only publishes
# `ValidationPass` and `ValidationSuite`.
from kirin.validation.validationpass import ValidationResult

from .recursion import CallGraph
from .static_call import UNROLL_HELP


@dataclass(frozen=True)
class CallSite:
    """Where the validated kernel calls into another one."""

    callee: str
    file: str | None
    lineno: int

    def __str__(self) -> str:
        where = f"{self.file}:{self.lineno}" if self.file else f"line {self.lineno}"
        return f"'{self.callee}' called at {where}"


@dataclass
class InlineOrigins:
    """Where each inlined statement came from, and which call brought it in.

    Used across three points in ``run_pass``, in order::

        origins = InlineOrigins.collect(mt)   # before the inliner runs
        origins.snapshot(mt)                  # before the validation suite runs
        origins.annotate(mt, suite.validate(mt)).raise_if_invalid()

    The middle step exists because ``attach`` mutates ``node.source`` in place --
    ``ValidationError.source`` *is* the statement's ``SourceInfo`` object, not a
    copy -- so the statement's true line offset has to be read before the suite
    touches it.
    """

    callees: dict[tuple[str, int], tuple[CallSite, ir.Method]] = field(
        default_factory=dict
    )
    """``(file, line offset)`` -> the kernel defined there and the call reaching it."""

    offsets: dict[int, int] = field(default_factory=dict)
    """``id(stmt)`` -> the statement's own line offset, taken pre-``attach``."""

    @classmethod
    def collect(cls, method: ir.Method) -> InlineOrigins:
        """Record the call sites reachable from ``method``.

        Must run **before** inlining: the invokes it reads are gone afterwards.

        The entry method itself is excluded. Code the user wrote in the kernel
        being compiled is not something a call introduced, and claiming otherwise
        would attach a misleading note to every ordinary error.
        """
        graph = CallGraph(method)
        origins = cls()

        # Direct invokes first, so each callee is credited to the call site
        # actually written in the entry method rather than to one further down.
        frontier: deque[tuple[ir.Method, CallSite]] = deque()
        seen: set[ir.Method] = set()
        for stmt in method.code.walk():
            trait = stmt.get_trait(ir.StaticCall)
            if trait is None or stmt.source is None:
                continue

            callee = trait.get_callee(stmt)
            if callee in seen:
                continue

            seen.add(callee)
            frontier.append(
                (
                    callee,
                    CallSite(
                        callee=callee.sym_name or "<lambda>",
                        file=stmt.source.file or method.file,
                        # `lineno` is relative to the enclosing kernel and
                        # `lineno_begin` is that kernel's offset into the file.
                        lineno=stmt.source.lineno + stmt.source.lineno_begin,
                    ),
                )
            )

        entry = _key(method.file, method.lineno_begin)
        while frontier:
            reached, site = frontier.popleft()
            key = _key(reached.file, reached.lineno_begin)
            if key is not None and key != entry:
                origins.callees.setdefault(key, (site, reached))

            for callee in graph.callees(reached):
                if callee in seen:
                    continue

                seen.add(callee)
                frontier.append((callee, site))

        return origins

    def snapshot(self, method: ir.Method) -> None:
        """Record every inlined statement's line offset before ``attach`` runs."""
        if not self.callees:
            return

        entry = _key(method.file, method.lineno_begin)
        for stmt in method.code.walk():
            source = stmt.source
            if source is None:
                continue

            key = _key(source.file, source.lineno_begin)
            if key is None or key == entry:
                continue

            self.offsets[id(stmt)] = source.lineno_begin

    def annotate(self, method: ir.Method, result: ValidationResult) -> ValidationResult:
        """Explain every error that came from inlined code, and fix its excerpt.

        Returns ``result`` so it can be chained onto ``validate``.

        ``offsets`` is what decides whether an error is about inlined code:
        ``snapshot`` only records statements that came from some other kernel, so
        a hit means the statement was spliced in and a miss means the user wrote
        it where the error says.

        The note goes into the error's *message* rather than its ``help``,
        because ``ValidationResult._format_errors`` always prints the message but
        only prints ``help`` alongside a source caret -- which is precisely what
        fails to render for an inlined statement.
        """
        if not self.callees:
            return result

        for errors in result.errors.values():
            for err in errors:
                source = getattr(err, "source", None)
                if source is None or source.file is None:
                    continue

                offset = self.offsets.get(id(err.node))
                if offset is None:
                    continue

                found = self.callees.get((source.file, offset))
                if found is None:
                    continue

                site, owner = found
                note = f"\n    (inlined from {site}; {UNROLL_HELP})"
                if err.args and isinstance(err.args[0], str):
                    err.args = (err.args[0] + note,) + err.args[1:]
                else:  # pragma: no cover - ValidationError always has a message
                    err.args = (note,) + err.args

                _repoint(err, source, offset, owner)

        return result


def _repoint(
    err: ir.ValidationError, source: SourceInfo, offset: int, owner: ir.Method
) -> None:
    """Undo ``attach``'s rewrite of ``err``'s source excerpt.

    Restores the line offset the statement actually carried and swaps in the
    defining kernel's own source, so the reported line number and the quoted code
    agree with the reported file. If that kernel's source cannot be read -- one
    defined in a REPL, say -- the excerpt is dropped rather than left showing the
    caller's unrelated code under the callee's filename.
    """
    source.lineno_begin = offset
    err.lines = _source_lines(owner)


def _key(file: str | None, lineno_begin: int) -> tuple[str, int] | None:
    """Identify the kernel a statement was written in, or ``None`` if unknowable."""
    return None if file is None else (file, lineno_begin)


def _source_lines(method: ir.Method) -> list[str]:
    if method.py_func is None:
        return []

    try:
        return textwrap.dedent(inspect.getsource(method.py_func)).splitlines()
    except (OSError, TypeError):
        return []
