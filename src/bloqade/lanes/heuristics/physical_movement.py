from dataclasses import dataclass, field, replace

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    SingleZonePlacementStrategyABC,
)
from bloqade.lanes.arch.gemini.physical import get_arch_spec as get_physical_arch_spec
from bloqade.lanes.layout.path import PathFinder


@dataclass
class PhysicalGreedyPlacementStrategy(SingleZonePlacementStrategyABC):
    """Greedy physical placement strategy for single-zone architectures.

    The strategy chooses CZ placements by moving at most one qubit per pair to a
    blockaded partner location, with optional lookahead pressure toward upcoming
    CZ partners. Move routing is synthesized with shortest paths from PathFinder
    and packed into compatible move layers.
    """

    arch_spec: layout.ArchSpec = field(default_factory=get_physical_arch_spec)
    return_to_left_after_cz: bool = True
    lane_move_overhead_cost: float = 0.0
    large_cost: float = 1e9
    _path_finder: PathFinder = field(init=False, repr=False)
    _pending_return_from: tuple[layout.LocationAddress, ...] | None = field(
        init=False, default=None, repr=False
    )
    _pending_return_to: tuple[layout.LocationAddress, ...] | None = field(
        init=False, default=None, repr=False
    )
    _pending_return_layers: tuple[tuple[layout.LaneAddress, ...], ...] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self):
        self._path_finder = PathFinder(self.arch_spec)

    @property
    def _home_site_count(self) -> int:
        # Gemini words are shaped as two columns x word_size_y rows.
        return len(self.arch_spec.words[0].site_indices) // 2

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        _ = initial_layout

    def _path_cost(self, path: tuple[layout.LaneAddress, ...] | None) -> float:
        if path is None:
            return self.large_cost
        return sum(
            self.arch_spec.get_lane_duration_cost(lane) + self.lane_move_overhead_cost
            for lane in path
        )

    def _find_route(
        self,
        src: layout.LocationAddress,
        dst: layout.LocationAddress,
        occupied: frozenset[layout.LocationAddress] = frozenset(),
    ) -> (
        tuple[tuple[layout.LaneAddress, ...], tuple[layout.LocationAddress, ...]] | None
    ):
        if src == dst:
            return (), (src,)
        return self._path_finder.find_path(
            src,
            dst,
            occupied=occupied,
            edge_weight=lambda lane: self.arch_spec.get_lane_duration_cost(lane)
            + self.lane_move_overhead_cost,
        )

    def _occupied_except(
        self,
        qubit_layout: tuple[layout.LocationAddress, ...] | list[layout.LocationAddress],
        occupied_static: (
            set[layout.LocationAddress] | frozenset[layout.LocationAddress]
        ),
        excluded_qids: frozenset[int] = frozenset(),
    ) -> frozenset[layout.LocationAddress]:
        return frozenset(occupied_static) | {
            loc for qid, loc in enumerate(qubit_layout) if qid not in excluded_qids
        }

    def _move_candidate_score(
        self,
        moved_qid: int,
        src: layout.LocationAddress,
        dst: layout.LocationAddress,
        occupied: frozenset[layout.LocationAddress],
        move_count: tuple[int, ...],
    ) -> float:
        route = self._find_route(src, dst, occupied=occupied)
        path_cost = self._path_cost(None if route is None else route[0])
        if path_cost >= self.large_cost:
            return self.large_cost
        # Prefer qubits with fewer accumulated moves on equal geometric cost.
        move_penalty = 1e-6 * float(move_count[moved_qid])
        return path_cost + move_penalty

    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState:
        updated_layout = list(state.layout)
        updated_move_count = list(state.move_count)
        occupied_static = set(state.occupied)

        pair_order = sorted(
            range(len(controls)),
            key=lambda idx: (
                -(state.move_count[controls[idx]] + state.move_count[targets[idx]]),
                controls[idx],
                targets[idx],
            ),
        )

        for idx in pair_order:
            control = controls[idx]
            target = targets[idx]
            c_loc = updated_layout[control]
            t_loc = updated_layout[target]

            c_blockaded = self.arch_spec.get_blockaded_location(c_loc)
            t_blockaded = self.arch_spec.get_blockaded_location(t_loc)

            if c_blockaded == t_loc or t_blockaded == c_loc:
                continue

            occupied_now = self._occupied_except(
                updated_layout,
                occupied_static,
                excluded_qids=frozenset((control, target)),
            )

            candidates: list[tuple[float, int, layout.LocationAddress]] = []
            if (
                c_blockaded is not None
                and c_blockaded not in occupied_now
                and c_blockaded != t_loc
            ):
                candidates.append(
                    (
                        self._move_candidate_score(
                            target,
                            t_loc,
                            c_blockaded,
                            occupied_now,
                            tuple(updated_move_count),
                        ),
                        target,
                        c_blockaded,
                    )
                )

            if (
                t_blockaded is not None
                and t_blockaded not in occupied_now
                and t_blockaded != c_loc
            ):
                candidates.append(
                    (
                        self._move_candidate_score(
                            control,
                            c_loc,
                            t_blockaded,
                            occupied_now,
                            tuple(updated_move_count),
                        ),
                        control,
                        t_blockaded,
                    )
                )

            if len(candidates) == 0:
                raise RuntimeError(
                    "No feasible movement candidate for CZ pair "
                    f"(control={control}, target={target})."
                )

            best_score, moved_qid, dst = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2].word_id, item[2].site_id),
            )
            if best_score >= self.large_cost:
                raise RuntimeError(
                    "No finite-cost movement candidate for CZ pair "
                    f"(control={control}, target={target})."
                )
            if updated_layout[moved_qid] != dst:
                updated_layout[moved_qid] = dst
                updated_move_count[moved_qid] += 1

        return replace(
            state,
            layout=tuple(updated_layout),
            move_count=tuple(updated_move_count),
        )

    def _inverse_layers(
        self,
        move_layers: tuple[tuple[layout.LaneAddress, ...], ...],
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        return tuple(
            tuple(lane.reverse() for lane in reversed(layer))
            for layer in reversed(move_layers)
        )

    def _apply_pending_return(
        self,
        state: ConcreteState,
    ) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
        if (
            not self.return_to_left_after_cz
            or self._pending_return_from is None
            or self._pending_return_to is None
            or self._pending_return_layers is None
        ):
            return state, ()

        if state.layout != self._pending_return_from:
            self._pending_return_from = None
            self._pending_return_to = None
            self._pending_return_layers = None
            return state, ()

        returned_state = replace(
            state,
            layout=self._pending_return_to,
            move_count=tuple(
                mc + int(src != dst)
                for mc, src, dst in zip(
                    state.move_count,
                    state.layout,
                    self._pending_return_to,
                    strict=True,
                )
            ),
        )
        return_layers = self._pending_return_layers
        self._pending_return_from = None
        self._pending_return_to = None
        self._pending_return_layers = None
        return returned_state, return_layers

    def compute_moves(
        self, state_before: ConcreteState, state_after: ConcreteState
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        changed = [
            (qid, src, dst)
            for qid, (src, dst) in enumerate(
                zip(state_before.layout, state_after.layout)
            )
            if src != dst
        ]
        if len(changed) == 0:
            return ()

        targets = {qid: dst for qid, _, dst in changed}
        current_layout = list(state_before.layout)
        layers: list[tuple[layout.LaneAddress, ...]] = []
        while any(current_layout[qid] != dst for qid, dst in targets.items()):
            layer: list[layout.LaneAddress] = []
            reserved_src: set[layout.LocationAddress] = set()
            reserved_dst: set[layout.LocationAddress] = set()
            moved_this_layer: dict[int, layout.LocationAddress] = {}
            occupied_at_layer_start = set(state_before.occupied) | set(current_layout)

            for qid in sorted(targets):
                src = current_layout[qid]
                dst_final = targets[qid]
                if src == dst_final:
                    continue
                occupied_for_path = occupied_at_layer_start - {src}
                route = self._find_route(
                    src,
                    dst_final,
                    occupied=frozenset(occupied_for_path),
                )
                if route is None:
                    continue
                path, locations = route
                if len(path) == 0:
                    continue
                lane = path[0]
                step_src = locations[0]
                step_dst = locations[1]
                if step_src != current_layout[qid]:
                    raise RuntimeError(
                        "Path synthesis produced an invalid first hop "
                        f"for qubit {qid}: expected source {current_layout[qid]}, "
                        f"got {step_src}."
                    )
                if not all(
                    self.arch_spec.compatible_lanes(lane, other) for other in layer
                ):
                    continue
                if step_src in reserved_src or step_src in reserved_dst:
                    continue
                if step_dst in occupied_at_layer_start:
                    continue
                if step_dst in reserved_src or step_dst in reserved_dst:
                    continue
                layer.append(lane)
                reserved_src.add(step_src)
                reserved_dst.add(step_dst)
                moved_this_layer[qid] = step_dst

            if len(layer) == 0:
                blocked_qids = tuple(
                    qid
                    for qid, dst in sorted(targets.items())
                    if current_layout[qid] != dst
                )
                raise RuntimeError(
                    "Physical move synthesis deadlocked: no collision-safe, "
                    "lane-compatible hop could be scheduled for remaining qubits "
                    f"{blocked_qids}."
                )
            for qid, dst in moved_this_layer.items():
                current_layout[qid] = dst
            layers.append(tuple(layer))

        return tuple(layers)

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        _ = lookahead_cz_layers
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()

        if not isinstance(state, ConcreteState):
            return AtomState.top()

        restored_state, return_layers = self._apply_pending_return(state)
        desired = self.desired_cz_layout(
            restored_state,
            controls,
            targets,
        )
        forward_layers = self.compute_moves(restored_state, desired)
        if self.return_to_left_after_cz:
            self._pending_return_from = desired.layout
            self._pending_return_to = restored_state.layout
            self._pending_return_layers = self._inverse_layers(forward_layers)
        move_layers = return_layers + forward_layers

        return ExecuteCZ(
            occupied=desired.occupied,
            layout=desired.layout,
            move_count=desired.move_count,
            active_cz_zones=frozenset([layout.ZoneAddress(0)]),
            move_layers=move_layers,
        )
