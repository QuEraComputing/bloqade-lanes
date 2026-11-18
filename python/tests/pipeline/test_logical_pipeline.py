"""Tests for LogicalPipeline."""

import bloqade.squin as squin
import pytest
from bloqade.rewrite.passes.callgraph import CallGraphPass
from bloqade.squin.gate.stmts import CX, CY, CZ, H, S, SqrtX, SqrtY, Swap
from kirin import rewrite
from kirin.analysis import CallGraph

from bloqade.lanes.dialects import place
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic
from bloqade.lanes.heuristics.logical.placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.transform import (
    LogicalPipeline,
    PhysicalNativeToPlace,
)


def _gate_sequence(mt) -> list[str]:
    """Name every squin Clifford statement in order, adjoints marked."""
    names = []
    for stmt in mt.callable_region.walk():
        if isinstance(stmt, (S, SqrtX, SqrtY)):
            names.append(f"{stmt.name}{'_adj' if stmt.adjoint else ''}")
        elif isinstance(stmt, CZ):
            names.append(stmt.name)
    return names


def test_physical_squin_clifford_rules_decompose_without_swapping():
    """The decomposition is shared by both pipelines; only the logical one
    appends the transversal swap.

    A plain ``@squin.kernel`` is not inlined at this stage, so the gates still
    sit inside the squin stdlib kernels — which is why the pass runs over the
    whole call graph rather than just the entry point."""

    @squin.kernel
    def kernel():
        reg = squin.qalloc(2)
        squin.h(reg[0])
        squin.cy(reg[0], reg[1])
        squin.broadcast.measure(reg)

    stage = PhysicalNativeToPlace()
    out = kernel.similar(kernel.dialects.add(place))
    CallGraphPass(out.dialects, rewrite.Chain(*stage._squin_clifford_rules()))(out)

    reachable = {mt.sym_name: mt for mt in CallGraph(out).edges}
    assert _gate_sequence(reachable["cy"]) == ["sqrt_x", "cz", "sqrt_x_adj"]
    assert _gate_sequence(reachable["h"]) == ["s", "sqrt_x", "s"]

    # No composite Clifford survives anywhere in the call graph.
    assert not [
        stmt
        for mt in reachable.values()
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, (H, CX, CY, Swap))
    ]


def test_logical_pipeline_layout_heuristic_default_is_none():
    """LogicalPipeline.layout_heuristic defaults to None."""
    pipeline = LogicalPipeline()
    assert pipeline.layout_heuristic is None


def test_logical_pipeline_layout_heuristic_mismatch_warns():
    """resolved_layout_heuristic warns when the explicit heuristic carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )

    logical_arch = LogicalPipeline().arch_spec
    physical_arch = get_physical_arch_spec()
    assert logical_arch != physical_arch

    mismatched_heuristic = LogicalLayoutHeuristic(arch_spec=physical_arch)
    pipeline = LogicalPipeline(layout_heuristic=mismatched_heuristic)

    with pytest.warns(
        UserWarning, match="layout_heuristic was constructed with a different"
    ):
        result = pipeline.resolved_layout_heuristic

    assert result is mismatched_heuristic


def test_logical_pipeline_placement_strategy_mismatch_warns():
    """resolved_placement_strategy warns when the explicit strategy carries a
    structurally different arch_spec than the pipeline."""
    from bloqade.lanes.arch.gemini.physical import (
        get_arch_spec as get_physical_arch_spec,
    )

    logical_arch = LogicalPipeline().arch_spec
    physical_arch = get_physical_arch_spec()
    assert logical_arch != physical_arch

    mismatched_strategy = LogicalPlacementStrategyNoHome(arch_spec=physical_arch)
    pipeline = LogicalPipeline(placement_strategy=mismatched_strategy)

    with pytest.warns(
        UserWarning, match="placement_strategy was constructed with a different"
    ):
        result = pipeline.resolved_placement_strategy

    assert result is mismatched_strategy
