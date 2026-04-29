"""Load and bundle the Rust entropy-search trace for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bloqade.lanes.bytecode._native import EntropyTrace, EntropyTraceStep
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)


@dataclass(frozen=True)
class TreeTraceStep:
    """Visualization-friendly step derived from a Rust ``EntropyTraceStep``."""

    step_index: int
    event: str
    node_id: int
    parent_node_id: int | None
    depth: int
    entropy: int
    unresolved_count: int
    moveset: frozenset[LaneAddress] | None
    candidate_movesets: tuple[frozenset[LaneAddress], ...]
    candidate_index: int | None
    reason: str | None
    state_seen_node_id: int | None
    no_valid_moves_qubit: int | None
    trigger_node_id: int | None
    configuration: dict[int, LocationAddress]
    parent_configuration: dict[int, LocationAddress] | None
    moveset_score: float | None
    best_buffer_node_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class EntropyTraceBundle:
    """Trace plus the metadata the visualizer needs alongside it."""

    steps: tuple[TreeTraceStep, ...]
    root_node_id: int
    best_buffer_size: int
    arch_spec: Any
    traced_target: dict[int, LocationAddress]
    blocked_locations: tuple[LocationAddress, ...]
    local_to_global_qid: dict[int, int]
    location_to_global_qid: dict[LocationAddress, int]
    kernel_name: str
    executed_cz_count: int


def _decode_lane(lane: tuple[int, int, int, int, int, int]) -> LaneAddress:
    direction, move_type, zone_id, word_id, site_id, bus_id = lane
    return LaneAddress(
        {0: MoveType.SITE, 1: MoveType.WORD, 2: MoveType.ZONE}[move_type],
        word_id,
        site_id,
        bus_id,
        Direction.FORWARD if direction == 0 else Direction.BACKWARD,
        zone_id,
    )


def _decode_config(
    entries: list[tuple[int, int, int, int]],
) -> dict[int, LocationAddress]:
    return {
        qid: LocationAddress(word_id, site_id, zone_id)
        for qid, zone_id, word_id, site_id in entries
    }


def _convert_step(idx: int, step: EntropyTraceStep) -> TreeTraceStep:
    return TreeTraceStep(
        step_index=idx,
        event=step.event,
        node_id=step.node_id,
        parent_node_id=step.parent_node_id,
        depth=step.depth,
        entropy=step.entropy,
        unresolved_count=step.unresolved_count,
        moveset=(
            None
            if step.moveset is None
            else frozenset(_decode_lane(lane) for lane in step.moveset)
        ),
        candidate_movesets=tuple(
            frozenset(_decode_lane(lane) for lane in candidate)
            for candidate in step.candidate_movesets
        ),
        candidate_index=step.candidate_index,
        reason=step.reason,
        state_seen_node_id=step.state_seen_node_id,
        no_valid_moves_qubit=step.no_valid_moves_qubit,
        trigger_node_id=step.trigger_node_id,
        configuration=_decode_config(step.configuration),
        parent_configuration=(
            None
            if step.parent_configuration is None
            else _decode_config(step.parent_configuration)
        ),
        moveset_score=step.moveset_score,
        best_buffer_node_ids=(
            None if not step.best_buffer_node_ids else tuple(step.best_buffer_node_ids)
        ),
    )


def convert_native_trace(
    trace: EntropyTrace,
) -> tuple[tuple[TreeTraceStep, ...], int, int]:
    """Convert a native Rust EntropyTrace into visualization types.

    Returns ``(steps, root_node_id, best_buffer_size)``.
    """
    steps = tuple(_convert_step(i, s) for i, s in enumerate(trace.steps))
    return steps, trace.root_node_id, trace.best_buffer_size


def build_entropy_trace(
    *,
    kernel: Any,
    kernel_name: str,
    layer_index: int = 0,
    max_expansions: int | None = 1000,
    max_goal_candidates: int | None = None,
) -> EntropyTraceBundle:
    """Run the compilation pipeline and capture an entropy trace for ``layer_index``.

    Drives the full squin->move pipeline with a ``RustPlacementTraversal``
    configured to collect an entropy trace. Returns all the state the
    visualizer needs to render the trace.
    """
    from bloqade.analysis import address
    from bloqade.analysis.address.lattice import AddressQubit

    from bloqade.lanes.analysis.layout import LayoutAnalysis
    from bloqade.lanes.arch.gemini.physical import get_arch_spec
    from bloqade.lanes.dialects import move, place
    from bloqade.lanes.heuristics.physical import (
        PhysicalLayoutHeuristicGraphPartitionCenterOut,
        PhysicalPlacementStrategy,
        RustPlacementTraversal,
    )
    from bloqade.lanes.upstream import NativeToPlace, squin_to_move

    arch_spec = get_arch_spec()
    layout_heuristic = PhysicalLayoutHeuristicGraphPartitionCenterOut(
        arch_spec=arch_spec
    )

    goal_candidates = (
        RustPlacementTraversal().max_goal_candidates
        if max_goal_candidates is None
        else max_goal_candidates
    )
    traversal = RustPlacementTraversal(
        strategy="entropy",
        max_expansions=max_expansions,
        max_goal_candidates=goal_candidates,
        collect_entropy_trace=True,
    )
    placement_strategy = PhysicalPlacementStrategy(
        arch_spec=arch_spec,
        traversal=traversal,
    )
    placement_strategy.trace_cz_index = layer_index

    move_main = squin_to_move(
        kernel,
        layout_heuristic=layout_heuristic,
        placement_strategy=placement_strategy,
        no_raise=False,
        logical_initialize=False,
    )
    executed_cz_count = sum(
        1 for stmt in move_main.callable_region.walk() if isinstance(stmt, move.CZ)
    )

    trace = placement_strategy.traced_rust_entropy_trace
    if trace is None:
        raise ValueError(
            f"No entropy trace captured. layer_index={layer_index} may be out of "
            f"range for this kernel ({executed_cz_count} CZ stage(s))."
        )
    steps, root_node_id, best_buffer_size = convert_native_trace(trace)

    traced_target: dict[int, LocationAddress] = placement_strategy.traced_target

    place_main = NativeToPlace(logical_initialize=False).emit(kernel, no_raise=False)
    address_analysis = address.AddressAnalysis(place_main.dialects)
    address_frame, _ = address_analysis.run(place_main)
    all_qubits = tuple(range(address_analysis.next_address))
    initial_layout = LayoutAnalysis(
        place_main.dialects,
        layout_heuristic,
        address_frame.entries,
        all_qubits,
    ).get_layout(place_main)
    location_to_global_qid = {
        location: qid for qid, location in zip(all_qubits, initial_layout, strict=True)
    }
    local_to_global_qid: dict[int, int] = {}
    cz_counter = 0
    for stmt in place_main.callable_region.walk():
        if isinstance(stmt, place.CZ):
            if cz_counter == layer_index:
                owner = stmt.parent_stmt
                while owner is not None and not isinstance(
                    owner, place.StaticPlacement
                ):
                    owner = owner.parent_stmt
                if isinstance(owner, place.StaticPlacement):
                    for local_idx, qubit_ssa in enumerate(owner.qubits):
                        addr = address_frame.entries.get(qubit_ssa)
                        if isinstance(addr, AddressQubit):
                            local_to_global_qid[local_idx] = addr.data
                break
            cz_counter += 1

    return EntropyTraceBundle(
        steps=steps,
        root_node_id=root_node_id,
        best_buffer_size=best_buffer_size,
        arch_spec=arch_spec,
        traced_target=dict(traced_target),
        blocked_locations=tuple(),
        local_to_global_qid=local_to_global_qid,
        location_to_global_qid=location_to_global_qid,
        kernel_name=kernel_name,
        executed_cz_count=executed_cz_count,
    )


def load_kernel_from_file(path: Path, symbol: str = "main") -> Any:
    """Dynamically load a kernel symbol from a Python file."""
    import importlib.util

    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Kernel file does not exist: {resolved}")
    module_name = f"_entropy_tree_kernel_{resolved.stem}_{abs(hash(resolved))}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from: {resolved}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, symbol):
        raise AttributeError(f"{resolved} has no symbol '{symbol}'")
    return getattr(module, symbol)
