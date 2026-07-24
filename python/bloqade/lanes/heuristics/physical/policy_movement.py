"""Move Policy DSL placement strategy.

Sidecar to :mod:`bloqade.lanes.heuristics.physical.movement` —
:class:`PolicyPlacementStrategy` mirrors
:class:`~bloqade.lanes.heuristics.physical.movement.PhysicalPlacementStrategy`
but routes each per-stage move synthesis call through the Starlark Move
Policy DSL (`bloqade.lanes.bytecode._native.PolicyRunner`) instead of the
``TargetSolver``-based path. Keeping the DSL path out of the strategy file
means main can keep evolving ``movement.py`` without colliding with DSL work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PlacementError,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import PolicyRunner
from bloqade.lanes.bytecode.encoding import (
    LaneAddress,
    LocationAddress,
    ZoneAddress,
)
from bloqade.lanes.heuristics.physical.movement import convert_move_layers
from bloqade.lanes.heuristics.physical.target_generator import (
    DefaultTargetGenerator,
    TargetContext,
    TargetGeneratorABC,
    TargetGeneratorCallable,
    _coerce_target_generator,
    _validate_candidate,
)


@dataclass(frozen=True)
class PolicyTraversal:
    """Config for the Move Policy DSL placement strategy.

    Sidecar to
    :class:`~bloqade.lanes.heuristics.physical.movement.RustPlacementTraversal`
    that only carries DSL-relevant knobs.
    """

    policy_path: str
    policy_params: dict[str, Any] | None = None
    max_expansions: int | None = 300
    timeout_s: float | None = None


def _is_acceptable_solve(result: object) -> bool:
    """Return True iff ``result`` represents a usable DSL solution.

    Accepts:
    - DSL solves with ``policy_status == "solved"`` (the canonical success
      path through the kernel).
    - DSL solves that halted via ``invoke_builtin("sequential_fallback")``
      and have a populated ``move_layers``. The Rust kernel writes the
      fallback path into ``move_layers`` and sets
      ``goal_config = target_cfg`` whenever a policy halts with the
      ``fallback`` status; the result is a valid (non-optimal) solution.
    """
    policy_status = getattr(result, "policy_status", None)
    if policy_status is None or not isinstance(policy_status, str):
        return False
    if policy_status == "solved":
        return True
    return bool(
        policy_status.startswith("fallback:")
        and len(getattr(result, "move_layers", [])) > 0
    )


@dataclass
class PolicyPlacementStrategy(PlacementStrategyABC):
    """Physical placement strategy backed by a Starlark Move Policy DSL.

    Each CZ-stage move synthesis call is dispatched through
    :class:`bloqade.lanes.bytecode._native.PolicyRunner` and the policy
    file named by ``traversal.policy_path``. Target generation reuses the
    same ``target_generator`` / default-fallback pipeline as
    :class:`~bloqade.lanes.heuristics.physical.movement.PhysicalPlacementStrategy`.

    ``traversal`` is keyword-only so it can be required without colliding
    with the defaulted fields ``PlacementStrategyABC`` inherits.
    """

    traversal: PolicyTraversal = field(kw_only=True)
    arch_spec: ArchSpec = field(default_factory=get_physical_arch_spec)
    target_generator: TargetGeneratorABC | TargetGeneratorCallable | None = None

    _cz_counter: int = field(default=0, init=False, repr=False)
    _rust_runner: PolicyRunner | None = field(default=None, init=False, repr=False)
    _rust_nodes_expanded_total: int = field(default=0, init=False, repr=False)
    _resolved_target_generator: TargetGeneratorABC | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)
        if not isinstance(self.traversal, PolicyTraversal):
            raise TypeError("traversal must be a PolicyTraversal instance")
        if self.target_generator is not None and not (
            isinstance(self.target_generator, TargetGeneratorABC)
            or callable(self.target_generator)
        ):
            raise TypeError(
                "target_generator must be a TargetGeneratorABC, a callable, or None"
            )
        self._resolved_target_generator = _coerce_target_generator(
            self.target_generator
        )

    def validate_initial_layout(
        self,
        initial_layout: tuple[LocationAddress, ...],
    ) -> None:
        _ = initial_layout

    def _get_rust_runner(self) -> PolicyRunner:
        if self._rust_runner is None:
            self._rust_runner = PolicyRunner.from_arch_spec(self.arch_spec._inner)
        return self._rust_runner

    @property
    def rust_nodes_expanded_total(self) -> int:
        """Total Rust DSL kernel node expansions for this strategy instance."""
        return self._rust_nodes_expanded_total

    def _build_candidates(
        self,
        ctx: TargetContext,
    ) -> list[dict[int, LocationAddress]]:
        plugin = self._resolved_target_generator
        plugin_candidates: list[dict[int, LocationAddress]] = (
            [] if plugin is None else list(plugin.generate(ctx))
        )
        deduped: list[dict[int, LocationAddress]] = []
        for candidate in plugin_candidates:
            _validate_candidate(ctx, candidate)
            if candidate not in deduped:
                deduped.append(candidate)
        default = DefaultTargetGenerator().generate(ctx)[0]
        if default not in deduped:
            deduped.append(default)
        return deduped

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        if state == AtomState.bottom():
            return AtomState.bottom()
        if len(controls) != len(targets):
            raise PlacementError(
                f"CZ has mismatched control/target counts: "
                f"{len(controls)} controls, {len(targets)} targets"
            )
        if not isinstance(state, ConcreteState):
            return AtomState.top()
        return self._cz_placements_rust(state, controls, targets, lookahead_cz_layers)

    def _cz_placements_rust(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        ctx = TargetContext(
            arch_spec=self.arch_spec,
            state=state,
            controls=controls,
            targets=targets,
            lookahead_cz_layers=lookahead_cz_layers,
            cz_stage_index=self._cz_counter,
        )
        candidates = self._build_candidates(ctx)

        runner = self._get_rust_runner()
        initial_native = {qid: loc._inner for qid, loc in ctx.placement.items()}
        blocked_native = [loc._inner for loc in state.occupied]

        remaining = self.traversal.max_expansions
        winning_result = None
        for candidate in candidates:
            if remaining is not None and remaining <= 0:
                break
            target_native = {qid: loc._inner for qid, loc in candidate.items()}
            result = runner.solve(
                initial_native,
                target_native,
                blocked_native,
                self.traversal.policy_path,
                policy_params=self.traversal.policy_params,
                max_expansions=remaining,
                timeout_s=self.traversal.timeout_s,
            )
            self._rust_nodes_expanded_total += int(result.nodes_expanded)
            if remaining is not None:
                remaining -= int(result.nodes_expanded)
            if _is_acceptable_solve(result):
                winning_result = result
                break

        self._cz_counter += 1

        if winning_result is None:
            raise PlacementError(
                f"CZ policy solver failed for pairs {list(zip(controls, targets))}; "
                "no candidate target layout produced an acceptable solve"
            )

        move_layers = convert_move_layers(winning_result.move_layers)
        goal_map = {
            qid: LocationAddress(loc.word_id, loc.site_id, loc.zone_id)
            for qid, loc in winning_result.goal_config.items()
        }
        goal_layout = tuple(goal_map[qid] for qid in range(len(state.layout)))
        move_count = tuple(
            mc + int(src != dst)
            for mc, src, dst in zip(state.move_count, state.layout, goal_layout)
        )
        return ExecuteCZ(
            occupied=state.occupied,
            layout=goal_layout,
            move_count=move_count,
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=move_layers,
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        _ = qubits
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def measure_placements(
        self,
        state: AtomState,
        qubits: tuple[int, ...],
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state
        if len(qubits) != len(state.layout):
            raise PlacementError(
                f"terminal measurement must measure all {len(state.layout)} "
                f"qubits in the block, got {len(qubits)}"
            )
        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(ZoneAddress(loc.zone_id) for loc in state.layout),
        )


def compute_move_layers_dsl(
    *,
    arch_spec: ArchSpec,
    initial: dict[int, LocationAddress],
    target: dict[int, LocationAddress],
    blocked: list[LocationAddress],
    traversal: PolicyTraversal,
    runner: PolicyRunner | None = None,
) -> tuple[tuple[LaneAddress, ...], ...]:
    """Solve a single move synthesis problem via the Move Policy DSL.

    Sidecar to
    :func:`bloqade.lanes.heuristics.move_synthesis.compute_move_layers`,
    routed through :class:`PolicyRunner` rather than :class:`TargetSolver`.
    """
    initial_native = {qid: loc._inner for qid, loc in initial.items()}
    target_native = {qid: loc._inner for qid, loc in target.items()}
    blocked_native = [loc._inner for loc in blocked]

    if runner is None:
        runner = PolicyRunner.from_arch_spec(arch_spec._inner)
    result = runner.solve(
        initial_native,
        target_native,
        blocked_native,
        traversal.policy_path,
        policy_params=traversal.policy_params,
        max_expansions=traversal.max_expansions,
        timeout_s=traversal.timeout_s,
    )
    if not _is_acceptable_solve(result):
        raise RuntimeError(
            f"DSL move synthesis failed with policy_status={result.policy_status!r}"
        )
    return convert_move_layers(result.move_layers)
