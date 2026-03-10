from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import starmap

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    SingleZonePlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.lattice import ExecuteMeasure
from bloqade.lanes.analysis.placement.strategy import PlacementStrategyABC
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers, move_to_left
from bloqade.lanes.layout.path import PathFinder


@dataclass(frozen=True)
class MoveOp:
    """Data class to store a move operation along with its source and destination addresses."""

    arch_spec: layout.ArchSpec
    src: layout.LocationAddress
    dst: layout.LocationAddress

    @cached_property
    def src_position(self) -> tuple[float, float]:
        return self.arch_spec.get_position(self.src)

    @cached_property
    def dst_position(self) -> tuple[float, float]:
        return self.arch_spec.get_position(self.dst)


def check_conflict(m0: MoveOp, m1: MoveOp):
    def check_coord_conflict(
        src0: float, dst0: float, src1: float, dst1: float
    ) -> bool:
        dir_src = (src1 - src0) // abs(src1 - src0) if src1 != src0 else 0
        dir_dst = (dst1 - dst0) // abs(dst1 - dst0) if dst1 != dst0 else 0
        return dir_src != dir_dst

    return any(
        starmap(
            check_coord_conflict,
            zip(m0.src_position, m1.src_position, m0.dst_position, m1.dst_position),
        )
    )


@dataclass
class LogicalPlacementMethods:
    arch_spec: layout.ArchSpec

    @property
    def _n_rows(self) -> int:
        return self.arch_spec.words[0].n_rows

    def desired_cz_layout(
        self,
        state: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
    ) -> ConcreteState:
        start_word_id = self._word_balance(state, controls, targets)
        moves: list[MoveOp] = []
        for c, t in self._sorted_cz_pairs_by_move_count(state, controls, targets):
            moves.append(self._pick_move(state, moves, start_word_id, c, t))
        return self._update_positions(state, moves)

    def validate_initial_layout(
        self,
        initial_layout: tuple[layout.LocationAddress, ...],
    ) -> None:
        for addr in initial_layout:
            if addr.word_id >= 2:
                raise ValueError(
                    "Initial layout contains invalid word id for logical arch"
                )
            if addr.site_id >= self._n_rows:
                raise ValueError(
                    "Initial layout should only site ids < "
                    f"{self._n_rows} for logical arch"
                )

    def _word_balance(
        self, state: ConcreteState, controls: tuple[int, ...], targets: tuple[int, ...]
    ) -> int:
        word_move_counts = {0: 0, 1: 0}
        for c, t in zip(controls, targets):
            c_addr = state.layout[c]
            t_addr = state.layout[t]
            if c_addr.word_id != t_addr.word_id:
                word_move_counts[c_addr.word_id] += state.move_count[c]
                word_move_counts[t_addr.word_id] += state.move_count[t]

        return 0 if word_move_counts[0] <= word_move_counts[1] else 1

    def _pick_move_by_conflict(
        self,
        moves: list[MoveOp],
        move1: MoveOp,
        move2: MoveOp,
    ) -> MoveOp:
        def count_conflicts(proposed_move: MoveOp) -> int:
            return sum(
                check_conflict(
                    proposed_move,
                    existing_move,
                )
                for existing_move in moves
            )

        return move1 if count_conflicts(move1) <= count_conflicts(move2) else move2

    def _pick_move(
        self,
        state: ConcreteState,
        moves: list[MoveOp],
        start_word_id: int,
        control: int,
        target: int,
    ) -> MoveOp:
        c_addr = state.layout[control]
        t_addr = state.layout[target]

        n_rows = self._n_rows
        c_addr_dst = layout.LocationAddress(t_addr.word_id, t_addr.site_id + n_rows)
        t_addr_dst = layout.LocationAddress(c_addr.word_id, c_addr.site_id + n_rows)
        c_move_count = state.move_count[control]
        t_move_count = state.move_count[target]

        move_t_to_c = MoveOp(self.arch_spec, t_addr, t_addr_dst)
        move_c_to_t = MoveOp(self.arch_spec, c_addr, c_addr_dst)

        if c_addr.word_id == t_addr.word_id:
            if c_move_count < t_move_count:
                return move_c_to_t
            if c_move_count > t_move_count:
                return move_t_to_c
            return self._pick_move_by_conflict(moves, move_c_to_t, move_t_to_c)
        if t_addr.word_id == start_word_id:
            return move_t_to_c
        return move_c_to_t

    def _update_positions(
        self,
        state: ConcreteState,
        moves: list[MoveOp],
    ) -> ConcreteState:
        new_positions: dict[int, layout.LocationAddress] = {}
        for move in moves:
            src_qubit = state.get_qubit_id(move.src)
            assert src_qubit is not None, "Source qubit must exist in state"
            new_positions[src_qubit] = move.dst

        new_layout = tuple(
            new_positions.get(i, loc) for i, loc in enumerate(state.layout)
        )
        new_move_count = list(state.move_count)
        for qid in new_positions:
            new_move_count[qid] += 1

        return replace(state, layout=new_layout, move_count=tuple(new_move_count))

    def _sorted_cz_pairs_by_move_count(
        self, state: ConcreteState, controls: tuple[int, ...], targets: tuple[int, ...]
    ) -> list[tuple[int, int]]:
        return sorted(
            zip(controls, targets),
            key=lambda x: state.move_count[x[0]] + state.move_count[x[1]],
            reverse=True,
        )


@dataclass
class LogicalPlacementStrategy(LogicalPlacementMethods, SingleZonePlacementStrategyABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)

    def compute_moves(
        self, state_before: ConcreteState, state_after: ConcreteState
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        return compute_move_layers(self.arch_spec, state_before, state_after)


@dataclass
class LogicalPlacementStrategyNoHome(LogicalPlacementMethods, PlacementStrategyABC):
    arch_spec: layout.ArchSpec = field(default_factory=get_arch_spec, init=False)
    H_lookahead: int = 4
    gamma: float = 0.85
    lambda_lookahead: float = 0.5
    K_candidates: int = 8
    large_cost: float = 1e9
    lane_move_overhead_cost: float = 0.0
    _path_finder: PathFinder = field(init=False, repr=False)
    _best_path_cache: dict[
        tuple[layout.LocationAddress, layout.LocationAddress],
        tuple[layout.LaneAddress, ...] | None,
    ] = field(default_factory=dict, init=False, repr=False)
    top_bus_signatures: int = 6
    bus_reward_rho: float = 0.7

    def __post_init__(self):
        self._path_finder = PathFinder(self.arch_spec)

    def _lane_sig(
        self, lane: layout.LaneAddress
    ) -> tuple[layout.MoveType, int, layout.Direction]:
        return (lane.move_type, lane.bus_id, lane.direction)

    def _sig_sort_key(
        self, sig: tuple[layout.MoveType, int, layout.Direction]
    ) -> tuple[int, int, int]:
        return (sig[0].value, sig[1], sig[2].value)

    def _path_sigs(
        self, path: tuple[layout.LaneAddress, ...] | None
    ) -> frozenset[tuple[layout.MoveType, int, layout.Direction]]:
        if path is None:
            return frozenset()
        return frozenset(self._lane_sig(lane) for lane in path)

    def _path_sig_maxcost(
        self, path: tuple[layout.LaneAddress, ...] | None
    ) -> dict[tuple[layout.MoveType, int, layout.Direction], float]:
        if path is None:
            return {}
        sig_maxcost: dict[tuple[layout.MoveType, int, layout.Direction], float] = {}
        for lane in path:
            sig = self._lane_sig(lane)
            lane_cost = self._get_lane_cost(lane)
            sig_maxcost[sig] = max(sig_maxcost.get(sig, 0.0), lane_cost)
        return sig_maxcost

    def _left_sites(self) -> set[layout.LocationAddress]:
        n_rows = self._n_rows
        return {
            layout.LocationAddress(word_id, site_id)
            for word_id in range(2)
            for site_id in range(n_rows)
        }

    def _distance_key(
        self,
        right_addr: layout.LocationAddress,
        left_addr: layout.LocationAddress,
    ) -> tuple[int, int, int, int]:
        right_row = right_addr.site_id - self._n_rows
        word_distance = 0 if left_addr.word_id == right_addr.word_id else 1
        site_distance = abs(left_addr.site_id - right_row)
        return (
            word_distance,
            site_distance,
            left_addr.word_id,
            left_addr.site_id,
        )

    def _pair_distance(
        self,
        addr0: layout.LocationAddress,
        addr1: layout.LocationAddress,
    ) -> float:
        # Use shortest-path lane cost as the lookahead proximity metric so both
        # immediate return selection and lookahead terms use the same objective.
        return self._path_cost(self._best_path(addr0, addr1))

    def _get_lane_duration(self, lane: layout.LaneAddress) -> float:
        return self.arch_spec.get_lane_duration_us(lane)

    def _get_lane_cost(self, lane: layout.LaneAddress) -> float:
        return self.arch_spec.get_lane_duration_cost(lane)

    def _best_path(
        self,
        src: layout.LocationAddress,
        dst: layout.LocationAddress,
    ) -> tuple[layout.LaneAddress, ...] | None:
        if src == dst:
            return ()
        key = (src, dst)
        if key not in self._best_path_cache:
            # Canonical placement objective: normalized lane duration cost with
            # optional per-move overhead to tune route complexity.
            result = self._path_finder.find_path(
                src,
                dst,
                edge_weight=lambda lane: self._get_lane_cost(lane)
                + self.lane_move_overhead_cost,
            )
            self._best_path_cache[key] = result[0] if result is not None else None
        return self._best_path_cache[key]

    def _path_cost(self, path: tuple[layout.LaneAddress, ...] | None) -> float:
        if path is None:
            return self.large_cost
        return sum(
            self._get_lane_cost(lane) + self.lane_move_overhead_cost for lane in path
        )

    def _nearest_left_layout(
        self, state_before: ConcreteState
    ) -> tuple[layout.LocationAddress, ...]:
        n_rows = self._n_rows
        left_sites = self._left_sites()
        used_left_sites = {
            addr for addr in state_before.layout if addr.site_id < n_rows
        }
        used_left_sites |= {
            addr for addr in state_before.occupied if addr.site_id < n_rows
        }
        available_left_sites = set(left_sites - used_left_sites)
        return_layout = list(state_before.layout)

        for qid, addr in enumerate(state_before.layout):
            if addr.site_id < n_rows:
                continue
            if not available_left_sites:
                raise ValueError(
                    "No empty left-column site available for right-column return move"
                )
            best_left_site = min(
                available_left_sites,
                key=lambda left_site: self._distance_key(addr, left_site),
            )
            return_layout[qid] = best_left_site
            available_left_sites.remove(best_left_site)
        return tuple(return_layout)

    def _partner_weights(
        self,
        lookahead_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> dict[int, dict[int, float]]:
        partners: dict[int, dict[int, float]] = defaultdict(dict)
        for depth, (controls, targets) in enumerate(lookahead_layers):
            weight = self.gamma**depth
            for c, t in zip(controls, targets):
                partners[c][t] = partners[c].get(t, 0.0) + weight
                partners[t][c] = partners[t].get(c, 0.0) + weight
        return partners

    def _lookahead_penalty(
        self,
        layout_after_return: tuple[layout.LocationAddress, ...],
        partner_weights: dict[int, dict[int, float]],
    ) -> float:
        penalty = 0.0
        for qid, neigh in partner_weights.items():
            for pid, weight in neigh.items():
                if pid <= qid:
                    continue
                penalty += weight * self._pair_distance(
                    layout_after_return[qid], layout_after_return[pid]
                )
        return penalty

    def _estimate_layers_time(
        self, layers: tuple[tuple[layout.LaneAddress, ...], ...]
    ) -> float:
        total = 0.0
        for layer in layers:
            if not layer:
                continue
            total += max(self._get_lane_duration(lane) for lane in layer)
        return total

    def _assignment_dp(
        self,
        row_edges: list[list[tuple[int, float]]],
        edge_sigs: dict[
            tuple[int, int], frozenset[tuple[layout.MoveType, int, layout.Direction]]
        ],
        *,
        col_count: int,
    ) -> tuple[float, tuple[int, ...] | None]:
        row_count = len(row_edges)
        if row_count == 0:
            return 0.0, ()
        if col_count < row_count:
            return float("inf"), None
        if any(len(edges) == 0 for edges in row_edges):
            return float("inf"), None

        memo: dict[
            tuple[int, int, tuple[layout.MoveType, int, layout.Direction] | None],
            tuple[float, tuple[int, ...] | None],
        ] = {}

        def solve(
            row: int,
            used_mask: int,
            locked_word_sig: tuple[layout.MoveType, int, layout.Direction] | None,
        ) -> tuple[float, tuple[int, ...] | None]:
            key = (row, used_mask, locked_word_sig)
            if key in memo:
                return memo[key]
            if row == row_count:
                return 0.0, ()

            best_cost = self.large_cost
            best_assign: tuple[int, ...] | None = None
            for col, edge_cost in row_edges[row]:
                if used_mask & (1 << col):
                    continue
                edge_sigset = edge_sigs.get((row, col), frozenset())
                ok, next_locked_word_sig = self._word_sig_transition(
                    locked_word_sig, edge_sigset
                )
                if not ok:
                    continue
                tail_cost, tail_assign = solve(
                    row + 1,
                    used_mask | (1 << col),
                    next_locked_word_sig,
                )
                if tail_assign is None:
                    continue
                total_cost = edge_cost + tail_cost
                candidate_assign = (col,) + tail_assign
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_assign = candidate_assign
                elif (
                    total_cost == best_cost
                    and best_assign is not None
                    and candidate_assign < best_assign
                ):
                    best_assign = candidate_assign
            memo[key] = (best_cost, best_assign)
            return memo[key]

        return solve(0, 0, None)

    def _mid_state_for_layout(
        self,
        state_before: ConcreteState,
        layout_after_return: tuple[layout.LocationAddress, ...],
    ) -> ConcreteState:
        return ConcreteState(
            occupied=state_before.occupied,
            layout=layout_after_return,
            move_count=tuple(
                mc + int(src != dst)
                for mc, src, dst in zip(
                    state_before.move_count,
                    state_before.layout,
                    layout_after_return,
                )
            ),
        )

    def _is_direction_mode_allowed(
        self,
        *,
        src_word: int,
        dst_word: int,
        mode: str,
    ) -> bool:
        if src_word == dst_word:
            return True
        if mode == "none":
            return False
        if mode == "01":
            return src_word == 0 and dst_word == 1
        if mode == "10":
            return src_word == 1 and dst_word == 0
        raise ValueError(f"Unknown direction mode: {mode}")

    def _word_sig_transition(
        self,
        locked_word_sig: tuple[layout.MoveType, int, layout.Direction] | None,
        edge_sigset: frozenset[tuple[layout.MoveType, int, layout.Direction]],
    ) -> tuple[bool, tuple[layout.MoveType, int, layout.Direction] | None]:
        word_sigs = tuple(sig for sig in edge_sigset if sig[0] is layout.MoveType.WORD)
        if not word_sigs:
            return True, locked_word_sig
        unique_word_sigs = set(word_sigs)
        if len(unique_word_sigs) != 1:
            return False, locked_word_sig
        edge_word_sig = next(iter(unique_word_sigs))
        if locked_word_sig is None or locked_word_sig == edge_word_sig:
            return True, edge_word_sig
        return False, locked_word_sig

    def _candidate_layouts(
        self,
        state_before: ConcreteState,
        lookahead_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> list[tuple[layout.LocationAddress, ...]]:
        if not lookahead_layers:
            return [self._nearest_left_layout(state_before)]

        n_rows = self._n_rows
        left_sites = self._left_sites()
        used_left_sites = {
            addr for addr in state_before.layout if addr.site_id < n_rows
        }
        used_left_sites |= {
            addr for addr in state_before.occupied if addr.site_id < n_rows
        }
        holes = sorted(
            left_sites - used_left_sites, key=lambda x: (x.word_id, x.site_id)
        )
        returners = [
            qid
            for qid, addr in enumerate(state_before.layout)
            if addr.site_id >= n_rows
        ]
        if not returners:
            return [state_before.layout]
        if len(holes) < len(returners):
            raise ValueError(
                "No empty left-column site available for right-column return move"
            )

        partner_weights = self._partner_weights(lookahead_layers)
        baseline_layout = self._nearest_left_layout(state_before)
        baseline_guess = {
            qid: baseline_layout[qid]
            for qid in returners
            if baseline_layout[qid].site_id < n_rows
        }
        left_stayers = {
            qid: addr
            for qid, addr in enumerate(state_before.layout)
            if addr.site_id < n_rows
        }
        reference_positions = {**left_stayers, **baseline_guess}

        edge_costs: dict[tuple[int, int], float] = {}
        edge_sigs: dict[
            tuple[int, int], frozenset[tuple[layout.MoveType, int, layout.Direction]]
        ] = {}
        edge_sig_maxcost: dict[
            tuple[int, int], dict[tuple[layout.MoveType, int, layout.Direction], float]
        ] = {}
        candidate_holes_by_returner: dict[int, set[int]] = {}
        for ridx, qid in enumerate(returners):
            src = state_before.layout[qid]
            scored: list[tuple[float, int]] = []
            for hidx, hole in enumerate(holes):
                path = self._best_path(src, hole)
                path_cost = self._path_cost(path)
                if path_cost >= self.large_cost:
                    continue
                path_sigs = self._path_sigs(path)
                path_sig_maxcost = self._path_sig_maxcost(path)
                future_delta = 0.0
                for pid, weight in partner_weights.get(qid, {}).items():
                    partner_pos = reference_positions.get(pid)
                    if partner_pos is None:
                        continue
                    future_delta += weight * self._pair_distance(hole, partner_pos)
                score = path_cost + self.lambda_lookahead * future_delta
                edge_costs[(ridx, hidx)] = score
                edge_sigs[(ridx, hidx)] = path_sigs
                edge_sig_maxcost[(ridx, hidx)] = path_sig_maxcost
                scored.append((score, hidx))

            scored.sort(key=lambda x: (x[0], x[1]))
            keep = {hidx for _, hidx in scored[: max(1, self.K_candidates)]}
            candidate_holes_by_returner[ridx] = keep

        sig_coverage: dict[tuple[layout.MoveType, int, layout.Direction], set[int]] = (
            defaultdict(set)
        )
        sig_typical_duration: dict[
            tuple[layout.MoveType, int, layout.Direction], float
        ] = {}
        for ridx in range(len(returners)):
            for hidx in candidate_holes_by_returner.get(ridx, set()):
                sigs = edge_sigs.get((ridx, hidx), frozenset())
                for sig in sigs:
                    sig_coverage[sig].add(ridx)
                    sig_cost = edge_sig_maxcost.get((ridx, hidx), {}).get(sig, 0.0)
                    sig_typical_duration[sig] = max(
                        sig_typical_duration.get(sig, 0.0), sig_cost
                    )

        duration_values = [dur for dur in sig_typical_duration.values() if dur > 0.0]
        duration_ref = (
            (sum(duration_values) / float(len(duration_values)))
            if duration_values
            else 1.0
        )
        sig_efficiency: dict[tuple[layout.MoveType, int, layout.Direction], float] = {
            sig: duration_ref / dur for sig, dur in sig_typical_duration.items()
        }

        sig_values: list[
            tuple[float, tuple[layout.MoveType, int, layout.Direction]]
        ] = []
        for sig, ridx_set in sig_coverage.items():
            value = float(len(ridx_set)) * sig_efficiency.get(sig, 0.0)
            sig_values.append((value, sig))
        sig_values.sort(key=lambda x: (-x[0], self._sig_sort_key(x[1])))
        top_signatures = [
            sig for _, sig in sig_values[: max(0, self.top_bus_signatures)]
        ]
        max_assignments = 1 + len(top_signatures)

        assignments: set[tuple[int, ...]] = set()

        def maybe_add_assignment(row_edges: list[list[tuple[int, float]]]) -> None:
            if len(assignments) >= max_assignments:
                return
            _, assignment = self._assignment_dp(
                row_edges,
                edge_sigs=edge_sigs,
                col_count=len(holes),
            )
            if assignment is not None:
                assignments.add(assignment)

        row_edges: list[list[tuple[int, float]]] = [[] for _ in returners]
        for ridx, _qid in enumerate(returners):
            for hidx in sorted(candidate_holes_by_returner.get(ridx, set())):
                base_cost = edge_costs.get((ridx, hidx), self.large_cost)
                if base_cost >= self.large_cost:
                    continue
                row_edges[ridx].append((hidx, base_cost))
        maybe_add_assignment(row_edges)

        for sig in top_signatures:
            if len(assignments) >= max_assignments:
                break
            sig_reward_scale = duration_ref * sig_efficiency.get(sig, 0.0)
            row_edges = [[] for _ in returners]
            for ridx, _qid in enumerate(returners):
                for hidx in sorted(candidate_holes_by_returner.get(ridx, set())):
                    base_cost = edge_costs.get((ridx, hidx), self.large_cost)
                    if base_cost >= self.large_cost:
                        continue
                    reward = (
                        self.bus_reward_rho * sig_reward_scale
                        if sig in edge_sigs.get((ridx, hidx), frozenset())
                        else 0.0
                    )
                    row_edges[ridx].append((hidx, base_cost - reward))
            maybe_add_assignment(row_edges)

        candidate_layouts: list[tuple[layout.LocationAddress, ...]] = []
        for assignment in sorted(assignments):
            new_layout = list(state_before.layout)
            for ridx, hidx in enumerate(assignment):
                new_layout[returners[ridx]] = holes[hidx]
            candidate_layouts.append(tuple(new_layout))

        if not candidate_layouts:
            candidate_layouts = [baseline_layout]
        return candidate_layouts[:max_assignments]

    def _single_step_return_choice(
        self,
        state_before: ConcreteState,
        lookahead_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...], float]:
        candidate_layouts = self._candidate_layouts(state_before, lookahead_layers)
        best: (
            tuple[
                float,
                ConcreteState,
                tuple[tuple[layout.LaneAddress, ...], ...],
            ]
            | None
        ) = None

        for layout_after_return in candidate_layouts:
            mid_state = self._mid_state_for_layout(state_before, layout_after_return)
            _, left_move_layers = move_to_left(self.arch_spec, state_before, mid_state)
            return_time = self._estimate_layers_time(left_move_layers)
            objective = return_time
            if best is None or objective < best[0]:
                best = (objective, mid_state, left_move_layers)

        assert best is not None, "At least one return candidate should exist"
        return best[1], best[2], best[0]

    def compute_moves(
        self, state_before: ConcreteState, state_after: ConcreteState
    ) -> tuple[tuple[layout.LaneAddress, ...], ...]:
        return compute_move_layers(self.arch_spec, state_before, state_after)

    def choose_return_layout(
        self,
        state_before: ConcreteState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> tuple[ConcreteState, tuple[tuple[layout.LaneAddress, ...], ...]]:
        _ = controls, targets
        if self.H_lookahead <= 0:
            bounded_lookahead = ()
        else:
            bounded_lookahead = lookahead_cz_layers[: self.H_lookahead]
        mid_state, left_move_layers, _ = self._single_step_return_choice(
            state_before, bounded_lookahead
        )
        return mid_state, left_move_layers

    def cz_placements(
        self,
        state: AtomState,
        controls: tuple[int, ...],
        targets: tuple[int, ...],
        lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = (),
    ) -> AtomState:
        if len(controls) != len(targets) or state == AtomState.bottom():
            return AtomState.bottom()

        if not isinstance(state, ConcreteState):
            return AtomState.top()

        mid_state, left_move_layers = self.choose_return_layout(
            state, controls, targets, lookahead_cz_layers
        )
        state_after = self.desired_cz_layout(mid_state, controls, targets)
        final_move_layers = self.compute_moves(mid_state, state_after)

        return ExecuteCZ(
            occupied=state_after.occupied,
            layout=state_after.layout,
            move_count=state_after.move_count,
            active_cz_zones=frozenset([layout.ZoneAddress(0)]),
            move_layers=(left_move_layers + final_move_layers),
        )

    def sq_placements(self, state: AtomState, qubits: tuple[int, ...]) -> AtomState:
        if isinstance(state, ConcreteState):
            return ConcreteState(
                occupied=state.occupied,
                layout=state.layout,
                move_count=state.move_count,
            )
        return state

    def measure_placements(
        self, state: AtomState, qubits: tuple[int, ...]
    ) -> AtomState:
        if not isinstance(state, ConcreteState):
            return state

        if len(qubits) != len(state.layout):
            return AtomState.bottom()

        return ExecuteMeasure(
            occupied=state.occupied,
            layout=state.layout,
            move_count=state.move_count,
            zone_maps=tuple(layout.ZoneAddress(0) for _ in qubits),
        )
