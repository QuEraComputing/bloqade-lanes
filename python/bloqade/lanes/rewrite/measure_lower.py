"""measure_lower — validate + rewrite move.Measure to move.EndMeasure."""

from __future__ import annotations

from dataclasses import dataclass

from kirin import ir
from kirin.analysis.forward import ForwardFrame
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade.lanes.analysis.atom.lattice import MeasureFuture, MoveExecution
from bloqade.lanes.dialects import move


@dataclass
class MeasureLower(RewriteRule):
    """Lower move.Measure stmts to move.EndMeasure.

    Driven by an AtomAnalysis frame: each move.Measure's ``future`` SSA
    result is looked up in the frame, and the associated
    ``MeasureFuture`` lattice element supplies both the measurement
    ordinal (``measurement_count``) and the set of zones actually
    observed (the keys of ``results``, whose iteration order mirrors
    insertion order and therefore the original zone order).

    The rewrite gives up silently (returns ``RewriteResult()`` with
    ``has_done_something=False``) when:

    1. The frame has no ``MeasureFuture`` for the node's future SSA.
    2. ``measurement_count`` on that future != 1 — the move.Measure is
       not the first/only final measurement in the program.
    3. ``len(results)`` on that future != 1 — the measurement spans
       multiple zones.

    Validating these invariants as hard errors is the job of dedicated
    validation passes that run between dialect transformations; rewrite
    rules just attempt the rewrite and back off when preconditions
    aren't met.
    """

    frame: ForwardFrame[MoveExecution]

    @classmethod
    def from_method(cls, method: ir.Method, arch_spec) -> "MeasureLower":
        """Build a MeasureLower by running AtomAnalysis on the given method.

        Stashes the returned ForwardFrame so that ``rewrite_Statement``
        can look up each ``move.Measure``'s future to get both the
        measurement ordinal and zone set from the ``MeasureFuture``
        lattice element.
        """
        from bloqade.lanes.analysis.atom import AtomInterpreter

        interp = AtomInterpreter(method.dialects, arch_spec=arch_spec)
        frame, _ = interp.run(method)
        return cls(frame=frame)

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, move.Measure):
            return RewriteResult()
        future = self.frame.entries.get(node.future)
        if not isinstance(future, MeasureFuture):
            return RewriteResult()
        if future.measurement_count != 1:
            return RewriteResult()
        zones = tuple(future.results.keys())
        if len(zones) != 1:
            return RewriteResult()
        replacement = move.EndMeasure(
            current_state=node.current_state,
            zone_addresses=zones,
        )
        replacement.insert_before(node)
        # move.Measure has two results (state, future) but move.EndMeasure
        # has only one (future). Rewire each explicitly — a bare
        # ``replace_by`` would positionally zip state→future and leave the
        # future result dangling.
        #
        # EndMeasure terminates the state chain, so forward the input state
        # to any residual state consumers (e.g. a trailing move.Store).
        node.result.replace_by(node.current_state)
        node.future.replace_by(replacement.result)
        node.delete()
        return RewriteResult(has_done_something=True)
