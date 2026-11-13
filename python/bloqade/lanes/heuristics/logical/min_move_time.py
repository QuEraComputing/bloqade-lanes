"""Initial placement by minimum-time paths and greedy parallel move schedules."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from heapq import heappop, heappush
from itertools import combinations
from math import fsum, inf, isclose

from kirin import interp

from bloqade.lanes.analysis.atom.atom_state_data import AtomStateData
from bloqade.lanes.arch.metrics import MoveMetricCalculator
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import MotionModel
from bloqade.lanes.bytecode.encoding import LaneAddress, LocationAddress
from bloqade.lanes.bytecode.exceptions import MoveValidationError
from bloqade.lanes.heuristics.logical.layout import LogicalLayoutHeuristic

_Group = tuple[int, int, int, int]
_Pairs = tuple[tuple[int, int], ...]


def _group(lane: LaneAddress) -> _Group:
    return (lane.zone_id, int(lane.move_type), lane.bus_id, int(lane.direction))


def _lane_key(lane: LaneAddress) -> tuple[int, ...]:
    return (*_group(lane), lane.word_id, lane.site_id)


def _greater(a: float, b: float) -> bool:
    # Equivalent schedules can accumulate their floating-point durations in
    # different orders. Do not prune a numerical tie before address tie-breaking.
    return a > b and not isclose(a, b, rel_tol=1e-12, abs_tol=1e-9)


@dataclass(frozen=True)
class _Step:
    lane: LaneAddress
    source: LocationAddress
    target: LocationAddress
    group: _Group
    duration_us: float
    return_duration_us: float


@dataclass(frozen=True)
class _Route:
    steps: tuple[_Step, ...]
    # For each bus/direction, the serial work this one atom must perform.
    # Outward and return work are separate: a CZ separates the two phases.
    requirements: tuple[tuple[_Group, float, float], ...]

    @classmethod
    def from_steps(cls, steps: tuple[_Step, ...]) -> "_Route":
        groups: dict[_Group, list[_Step]] = defaultdict(list)
        for step in steps:
            groups[step.group].append(step)
        return cls(
            steps,
            tuple(
                (
                    group,
                    fsum(step.duration_us for step in members),
                    fsum(step.return_duration_us for step in members),
                )
                for group, members in sorted(groups.items())
            ),
        )


@dataclass(frozen=True)
class _Schedule:
    # Outward layers followed by their palindrome return (CZ is implicit).
    layers: tuple[tuple[LaneAddress, ...], ...]
    duration_us: float


class _TimeScorer:
    """Per-invocation path table and deterministic, occupancy-aware scheduler."""

    def __init__(self, arch: ArchSpec, motion_model: MotionModel):
        self.arch = arch
        self.homes = tuple(sorted(arch.home_sites))
        self.metrics = MoveMetricCalculator(arch, motion_model)
        partners = {home: arch.get_cz_partner(home) for home in self.homes}
        if any(
            partner is None or partner in arch.home_sites
            for partner in partners.values()
        ):
            raise ValueError(
                "every logical home must have a non-home CZ staging partner"
            )

        # Keep edges, not just location paths: parallel edges can use different
        # buses and have different durations. Tie-break equal-time paths by lane.
        self.steps: list[_Step] = []
        self.edges: dict[LocationAddress, list[int]] = defaultdict(list)
        for lane in sorted(arch.iter_all_lanes(), key=_lane_key):
            if arch.check_lane_group((lane,)) or arch.check_lane_group(
                (lane.reverse(),)
            ):
                continue
            source, target = arch.get_endpoints(lane)
            self.edges[source].append(len(self.steps))
            self.steps.append(
                _Step(
                    lane,
                    source,
                    target,
                    _group(lane),
                    self.metrics.get_lane_duration_us(lane),
                    self.metrics.get_lane_duration_us(lane.reverse()),
                )
            )
        rows: list[tuple[_Route | None, ...]] = []
        for source in self.homes:
            row: list[_Route | None] = []
            for target in self.homes:
                partner = partners[target]
                assert partner is not None
                row.append(
                    None
                    if source == target
                    else self._shortest_path(source, partner, target)
                )
            rows.append(tuple(row))
        self.routes = tuple(rows)

    def _shortest_path(
        self,
        source: LocationAddress,
        target: LocationAddress,
        stationary_target: LocationAddress,
    ) -> _Route | None:
        """Dijkstra in microseconds, with the interaction's sitter blocked.

        Other atoms are deliberately not known here. Their occupancy is checked
        by the stage scheduler; it does not re-route a blocked precomputed path.
        """
        queue: list[tuple[float, tuple[int, ...], LocationAddress]] = [
            (0.0, (), source)
        ]
        best: dict[LocationAddress, tuple[float, tuple[int, ...]]] = {source: (0.0, ())}
        while queue:
            cost, path, location = heappop(queue)
            if best.get(location) != (cost, path):
                continue
            if location == target:
                return _Route.from_steps(tuple(self.steps[index] for index in path))
            for index in self.edges.get(location, ()):
                step = self.steps[index]
                if step.target == stationary_target:
                    continue
                candidate = (cost + step.duration_us, (*path, index))
                if step.target not in best or candidate < best[step.target]:
                    best[step.target] = candidate
                    heappush(queue, (*candidate, step.target))
        return None

    def schedule(self, occupied: int, pairs: _Pairs) -> _Schedule | None:
        """Greedily batch ready lanes, then reverse the schedule back home.

        The first member of each ordered pair moves to the second's CZ partner.
        Maximize time saved by batching, then atom count; prefer shorter batches
        and finally lane order on ties. Exhaustive subsets within each ready
        group (at most five movers on Gemini) handle incompatible lane geometry,
        AOD limits and moves that can only execute together to vacate a site.
        """
        routes: list[_Route] = []
        for source, target in sorted(pairs):
            route = self.routes[source][target]
            if route is None:
                return None
            routes.append(route)
        state = AtomStateData.new(
            {i: home for i, home in enumerate(self.homes) if occupied & (1 << i)}
        )
        offsets = [0] * len(routes)
        layers: list[tuple[LaneAddress, ...]] = []
        costs: list[float] = []
        while True:
            ready: dict[_Group, list[tuple[int, _Step]]] = defaultdict(list)
            for i, route in enumerate(routes):
                if offsets[i] < len(route.steps):
                    step = route.steps[offsets[i]]
                    ready[step.group].append((i, step))
            if not ready:
                break
            best_key: tuple[float, int, float, tuple[tuple[int, ...], ...]] | None = (
                None
            )
            best_batch: tuple[tuple[int, _Step], ...] = ()
            for members in ready.values():
                for size in range(1, len(members) + 1):
                    for batch in combinations(members, size):
                        lanes = tuple(step.lane for _, step in batch)
                        try:
                            state.validate_moves(lanes, self.arch)
                        except MoveValidationError:
                            continue
                        if not self._fits_capacity(tuple(step for _, step in batch)):
                            continue
                        if self.arch.check_lane_group(
                            tuple(lane.reverse() for lane in lanes)
                        ):
                            continue
                        cost = max(step.duration_us for _, step in batch) + max(
                            step.return_duration_us for _, step in batch
                        )
                        savings = (
                            fsum(
                                step.duration_us + step.return_duration_us
                                for _, step in batch
                            )
                            - cost
                        )
                        key = (
                            -savings,
                            -size,
                            cost,
                            tuple(_lane_key(lane) for lane in lanes),
                        )
                        if best_key is None or key < best_key:
                            best_key, best_batch = key, batch
            if best_key is None:
                return None
            lanes = tuple(step.lane for _, step in best_batch)
            state = state.apply_validated(state.validate_moves(lanes, self.arch))
            layers.append(lanes)
            costs.append(best_key[2])
            for i, _ in best_batch:
                offsets[i] += 1

        returns = [
            tuple(lane.reverse() for lane in layer) for layer in reversed(layers)
        ]
        return _Schedule(tuple(layers + returns), fsum(costs))

    def _fits_capacity(self, steps: tuple[_Step, ...]) -> bool:
        # check_lane_group validates lane geometry, but tone-count limits are
        # enforced separately by the router. Check both ends because this
        # schedule also executes the reversed group on the way home.
        if (capacity := self.arch.aod_capacity) is None:
            return True
        for locations in (
            (step.source for step in steps),
            (step.target for step in steps),
        ):
            positions = [self.arch.get_position(location) for location in locations]
            if (
                len({x for x, _ in positions}) > capacity[0]
                or len({y for _, y in positions}) > capacity[1]
            ):
                return False
        return True


@dataclass
class LogicalLayoutHeuristicMinMoveTime(LogicalLayoutHeuristic):
    """Exhaustively minimize a greedy estimate of CZ transport time.

    Precompute minimum-time directed paths from each home to each other home's
    CZ staging partner using ``MotionModel`` (including pickup and drop). At
    each CZ stage, batch only ready moves on the same zone/type/bus/direction,
    subject to the architecture and occupancy checks. Price each batch by its
    longest lane duration and include the palindrome return back home.

    This estimates a fixed-path greedy schedule, not the downstream router's
    optimum. A blocked schedule is rejected without rerouting. Search includes
    spectators because they can block paths; pinned allocations remain fixed.
    Equal-time layouts are resolved by home-address order. The default greedy
    pipeline heuristic is unchanged. Dense circuits can require up to 10!
    assignments; only admissible per-bus work bounds prune partial layouts.
    """

    motion_model: MotionModel = field(default_factory=MotionModel)

    def _prepare(self) -> _TimeScorer:
        return _TimeScorer(self.arch_spec, self.motion_model)

    def compute_layout(
        self,
        all_qubits: tuple[int, ...],
        stages: list[_Pairs],
        pinned: dict[int, LocationAddress] | None = None,
    ) -> tuple[LocationAddress, ...]:
        pinned = {} if pinned is None else pinned
        self._validate_pinned(all_qubits, pinned)
        if len(set(all_qubits)) != len(all_qubits):
            raise ValueError("all_qubits must contain unique qubit IDs")
        if len(all_qubits) > self.arch_spec.max_qubits:
            raise interp.InterpreterError(
                f"Number of qubits in circuit ({len(all_qubits)}) exceeds "
                f"maximum supported by logical architecture ({self.arch_spec.max_qubits})"
            )
        homes = tuple(sorted(self.arch_spec.home_sites))
        if len(all_qubits) > len(homes):
            raise ValueError("not enough logical home sites for all qubits")
        qubits = tuple(sorted(all_qubits))
        for stage in stages:
            touched: set[int] = set()
            for source, target in stage:
                if source not in qubits or target not in qubits:
                    raise ValueError(
                        "CZ stage refers to a qubit absent from all_qubits"
                    )
                if source == target:
                    raise ValueError("CZ stage cannot pair a qubit with itself")
                if source in touched or target in touched:
                    raise ValueError("CZ stage must contain disjoint qubit pairs")
                touched.update((source, target))

        home_index = {home: index for index, home in enumerate(homes)}
        layout = {q: home_index[home] for q, home in pinned.items()}
        used = sum(1 << slot for slot in layout.values())
        if not any(stages):
            free = iter(slot for slot in range(len(homes)) if not used & (1 << slot))
            return tuple(
                homes[layout[q] if q in layout else next(free)] for q in qubits
            )

        scorer = self._prepare()
        # Identical stages have identical scores because every stage returns
        # home. Sorting pairs also makes scheduling independent of input order.
        counts = Counter(tuple(sorted(stage)) for stage in stages if stage)
        unique_stages = tuple(sorted(counts))
        frequencies = [counts[stage] for stage in unique_stages]
        degrees = Counter(q for stage in unique_stages for pair in stage for q in pair)
        search_qubits = tuple(
            sorted(
                (q for q in qubits if q not in pinned), key=lambda q: (-degrees[q], q)
            )
        )
        ranks = {q: rank for rank, q in enumerate(search_qubits)}
        by_last: list[list[tuple[int, int, int]]] = [[] for _ in search_qubits]
        initial_pairs: list[tuple[int, int, int]] = []
        for layer, stage in enumerate(unique_stages):
            for source, target in stage:
                pending = [q for q in (source, target) if q not in pinned]
                if pending:
                    by_last[max(ranks[q] for q in pending)].append(
                        (layer, source, target)
                    )
                else:
                    initial_pairs.append((layer, source, target))

        @lru_cache(maxsize=16384)
        def stage_time(occupied: int, pairs: _Pairs) -> float:
            schedule = scorer.schedule(occupied, pairs)
            return inf if schedule is None else schedule.duration_us

        # A bus can run only one direction per operation. For any one route,
        # repeated uses of a group must execute sequentially; other atoms can
        # at best share that work. Sum the maximum per-route work per group,
        # separately for outward and return phases. This is a monotone lower
        # bound even though the eventual greedy schedule is not monotone.
        requirements: list[dict[_Group, tuple[float, float]]] = [
            {} for _ in unique_stages
        ]

        def add_pairs(
            pairs: list[tuple[int, int, int]],
        ) -> tuple[float, list[tuple[int, _Group, tuple[float, float]]]]:
            delta = 0.0
            changes: list[tuple[int, _Group, tuple[float, float]]] = []
            for layer, source, target in pairs:
                route = scorer.routes[layout[source]][layout[target]]
                if route is None:
                    return inf, changes
                for group, forward, reverse in route.requirements:
                    previous = requirements[layer].get(group, (0.0, 0.0))
                    current = (max(previous[0], forward), max(previous[1], reverse))
                    if current != previous:
                        changes.append((layer, group, previous))
                        requirements[layer][group] = current
                        delta += (
                            (current[0] - previous[0]) + (current[1] - previous[1])
                        ) * frequencies[layer]
            return delta, changes

        best_time = inf
        best_slots: tuple[int, ...] | None = None

        def visit(index: int, occupied: int, bound: float) -> None:
            nonlocal best_time, best_slots
            if bound == inf or _greater(bound, best_time):
                return
            if index == len(search_qubits):
                slots = tuple(layout[q] for q in qubits)
                costs: list[float] = []
                for layer, stage in enumerate(unique_stages):
                    pairs = tuple(sorted((layout[s], layout[t]) for s, t in stage))
                    cost = stage_time(occupied, pairs)
                    if cost == inf:
                        return
                    costs.append(cost * frequencies[layer])
                    if _greater(fsum(costs), best_time):
                        return
                total = fsum(costs)
                if (
                    best_slots is None
                    or _greater(best_time, total)
                    or (not _greater(total, best_time) and slots < best_slots)
                ):
                    best_time, best_slots = total, slots
                return

            qubit = search_qubits[index]
            # Spectator IDs do not affect scheduling, only their occupied set.
            # Assign unpinned spectators in increasing address order, retaining
            # the lexicographically first representative of each occupied set.
            first_slot = 0
            if degrees[qubit] == 0 and index > 0:
                previous_qubit = search_qubits[index - 1]
                if degrees[previous_qubit] == 0:
                    first_slot = layout[previous_qubit] + 1
            for slot in range(first_slot, len(homes)):
                if occupied & (1 << slot):
                    continue
                layout[qubit] = slot
                delta, changes = add_pairs(by_last[index])
                visit(index + 1, occupied | (1 << slot), bound + delta)
                for layer, group, previous in reversed(changes):
                    if previous == (0.0, 0.0):
                        del requirements[layer][group]
                    else:
                        requirements[layer][group] = previous
                del layout[qubit]

        initial_bound, _ = add_pairs(initial_pairs)
        visit(0, used, initial_bound)
        if best_slots is None:
            raise ValueError(
                "no home layout admits a legal greedy schedule for the precomputed paths"
            )
        return tuple(homes[slot] for slot in best_slots)
