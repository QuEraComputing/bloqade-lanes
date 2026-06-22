"""Shared scaffolding for Rust-solver-driven no-return placement strategies.

A *no-return* strategy is one where atoms are not moved back to home
positions between CZ layers; the output configuration of one layer becomes
the input of the next. The Rust solver decides the per-layer target layout
itself (via Hungarian, K-candidate rollouts, or two-phase assignment) rather
than receiving an externally computed target dict.

Subclasses implement :meth:`NoReturnStrategyBase._invoke_placement` to call
their specific CzPlacement entry point. The base owns:

* engine caching (`_get_engine`)
* the shared :class:`SolveOptions` construction (`_build_solve_options`)
* the :meth:`cz_placements` request/response plumbing
* the standard single-zone `sq_placements` / `measure_placements` paths
* the `rust_nodes_expanded_total` observability counter
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.bytecode import _native
from bloqade.lanes.bytecode._native import (
    DeadlockPolicy,
    MoveSearch,
    SearchEngine,
    SearchStrategy,
)
from bloqade.lanes.bytecode.encoding import LocationAddress
from bloqade.lanes.heuristics.physical.movement import convert_move_layers


@dataclass
class NoReturnStrategyBase(PlacementStrategyABC):
    """Abstract base for Rust-solver-driven no-return placement strategies.

    Parameters
    ----------
    arch_spec:
        Architecture specification (inherited from
        :class:`PlacementStrategyABC`). Must expose exactly one CZ-capable
        zone; multi-zone CZ scheduling is not yet supported.
    strategy:
        Inner search strategy as a :class:`SearchStrategy` enum (e.g.
        :py:attr:`SearchStrategy.IDS`, :py:attr:`SearchStrategy.ASTAR`,
        :py:attr:`SearchStrategy.ENTROPY`, …). Passing a typo'd value
        fails at construction rather than at solve time. Default
        :py:attr:`SearchStrategy.IDS`.
    max_expansions:
        Maximum node expansions per solve call. ``None`` means unbounded.
    restarts:
        Number of parallel restarts with perturbed scoring inside each
        Rust solve. Subclasses override the default where appropriate.
    deadlock_policy:
        :class:`DeadlockPolicy` enum value:
        :py:attr:`DeadlockPolicy.SKIP`,
        :py:attr:`DeadlockPolicy.MOVE_BLOCKERS` (default), or
        :py:attr:`DeadlockPolicy.ALL_MOVES`.
    top_c:
        Per-qubit move-candidate pruning cap inside ``HeuristicGenerator``.
        ``None`` keeps all scored bus options (default for this base —
        subclasses such as :class:`NoReturnPlacementStrategy` override to
        ``3`` to match their historical behaviour).

    Notes
    -----
    Subclasses are expected to be :func:`dataclasses.dataclass`-decorated,
    add their strategy-specific options as additional fields (with
    defaults so the inherited ``arch_spec`` field stays the only required
    init argument), and implement :meth:`_invoke_placement`.
    """

    strategy: SearchStrategy = field(default_factory=lambda: SearchStrategy.IDS)
    max_expansions: int | None = 100
    restarts: int = 1
    deadlock_policy: DeadlockPolicy = field(
        default_factory=lambda: DeadlockPolicy.MOVE_BLOCKERS
    )
    top_c: int | None = None

    _engine: SearchEngine | None = field(default=None, init=False, repr=False)
    _rust_nodes_expanded_total: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)

    @property
    def rust_nodes_expanded_total(self) -> int:
        """Total Rust solver node expansions for this strategy instance.

        Mirrors :pyattr:`PhysicalPlacementStrategy.rust_nodes_expanded_total`.
        Accumulates across all ``cz_placements`` calls on the instance.
        """
        return self._rust_nodes_expanded_total

    def _get_engine(self) -> SearchEngine:
        if self._engine is None:
            self._engine = SearchEngine.from_arch_spec(self.arch_spec._inner)
        return self._engine

    def _build_solve_options(self) -> _native.SolveOptions:
        """Build the shared :class:`SolveOptions` from the base fields.

        ``lookahead`` is hardcoded to ``True`` to match the historical
        behaviour of all three concrete strategies; promote to a field if a
        future caller needs it configurable.
        """
        return _native.SolveOptions(
            strategy=self.strategy,
            restarts=self.restarts,
            lookahead=True,
            deadlock_policy=self.deadlock_policy,
            top_c=self.top_c,
        )

    def _build_move_search(self) -> MoveSearch:
        """Build a :class:`MoveSearch` from the base solve-option fields."""
        return MoveSearch.ids().with_options(self._build_solve_options())

    @abc.abstractmethod
    def _invoke_placement(
        self,
        engine: SearchEngine,
        move_search: MoveSearch,
        initial: dict[int, "_native.LocationAddress"],
        cz_pairs: list[tuple[int, int]],
        blocked: list["_native.LocationAddress"],
        future_cz_layers: list[list[tuple[int, int]]] | None,
    ) -> "_native.SolveResult":
        """Call the strategy-specific CzPlacement entry point.

        Implementations should build the appropriate typed placement object
        (e.g. :class:`LooseGoalCzPlacement`, :class:`NoHomeCzPlacement`,
        :class:`RecedingHorizonCzPlacement`) from ``engine`` and
        ``move_search``, then delegate to its ``solve_pairs`` method.
        """

    def validate_initial_layout(
        self, initial_layout: tuple[LocationAddress, ...]
    ) -> None:
        # No-op: no-return strategies accept any valid initial layout from
        # the upstream placement pass. ``PlacementStrategyABC`` declares
        # this method abstract, so the override is required. Matches the
        # same no-op pattern used by :class:`PhysicalPlacementStrategy`.
        _ = initial_layout

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()
        state = self._unwrap_cz_input(state)
        if not isinstance(state, ConcreteState):
            return AtomState.top()

        engine = self._get_engine()
        move_search = self._build_move_search()
        initial = {qid: state.layout[qid]._inner for qid in range(len(state.layout))}
        cz_pairs = list(zip(controls, targets))
        blocked = [loc._inner for loc in state.occupied]

        # lookahead_cz_layers[0] is the current layer (skip it);
        # lookahead_cz_layers[1:] are the future layers.
        future: list[list[tuple[int, int]]] | None = None
        if len(lookahead_cz_layers) > 1:
            future = [list(zip(c, t)) for c, t in lookahead_cz_layers[1:]]

        result = self._invoke_placement(
            engine, move_search, initial, cz_pairs, blocked, future
        )
        self._rust_nodes_expanded_total += int(result.nodes_expanded)

        if result.status != "solved":
            return AtomState.bottom()

        move_layers = convert_move_layers(result.move_layers)
        goal_map = {
            qid: LocationAddress(loc.word_id, loc.site_id, loc.zone_id)
            for qid, loc in result.goal_config.items()
        }
        goal_layout = tuple(
            goal_map.get(qid, state.layout[qid]) for qid in range(len(state.layout))
        )
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
        return self._strip_user_moved(state)
