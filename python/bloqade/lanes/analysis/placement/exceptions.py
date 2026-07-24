"""Exceptions raised by the placement analysis.

The placement analysis runs as a kirin ``Forward`` dataflow analysis. When it
detects an infeasible or invalid placement it raises ``PlacementError`` rather
than silently returning ``AtomState.bottom()``:

- Under ``run`` (the pipeline's ``no_raise=False`` path) the exception
  propagates with its detailed message, giving actionable feedback.
- Under ``run_no_raise`` (``no_raise=True``) the ``Forward`` analysis catches
  it and degrades the whole method to ``bottom()`` — the same silent, best-effort
  behaviour as before, but without masking the cause when the caller opts in.

This is deliberately *not* used for input-guard checks (e.g. an incoming state
that is not a ``ConcreteState``): those merely reflect an error earlier in the
analysis, so the offending state is forwarded through unchanged instead.
"""

from __future__ import annotations


class PlacementError(Exception):
    """The placement analysis detected an infeasible or invalid placement.

    Examples: an invalid user-directed move (``move_to`` to a duplicate or
    occupied destination), an unroutable request the move synthesizer/solver
    could not satisfy, a spurious CZ partner pair that a global CZ pulse would
    entangle, or malformed placement IR (mismatched CZ control/target counts,
    an incomplete terminal measurement).
    """
