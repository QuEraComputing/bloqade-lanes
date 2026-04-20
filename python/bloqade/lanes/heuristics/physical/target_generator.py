"""Target-generation plugin interface for ``PhysicalPlacementStrategy``.

``PhysicalPlacementStrategy`` asks a :class:`TargetGeneratorABC` for an
ordered list of candidate target placements before each CZ stage. The
strategy always appends :class:`DefaultTargetGenerator`'s output as a
guaranteed last-resort fallback, so plugins may return ``[]`` to defer
entirely to the default.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from bloqade.lanes import layout
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.layout import LocationAddress, MoveType


@dataclass(frozen=True)
class TargetContext:
    """Signals passed to a TargetGenerator.

    Composes ConcreteState to avoid duplicating lattice state fields.
    """

    arch_spec: layout.ArchSpec
    state: ConcreteState
    controls: tuple[int, ...]
    targets: tuple[int, ...]
    lookahead_cz_layers: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    cz_stage_index: int

    @property
    def placement(self) -> dict[int, LocationAddress]:
        return dict(enumerate(self.state.layout))


class TargetGeneratorABC(abc.ABC):
    """Plugin interface for choosing the target configuration of a CZ stage.

    Implementations return an *ordered* list of candidate target
    placements. The strategy framework appends the default candidate
    (``DefaultTargetGenerator``) as a guaranteed last-resort, so a plugin
    may return ``[]`` to defer entirely to the default.
    """

    @abc.abstractmethod
    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]: ...


@dataclass(frozen=True)
class DefaultTargetGenerator(TargetGeneratorABC):
    """Default rule: control qubit moves to the CZ partner of the target's location."""

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        target = dict(ctx.placement)
        for control_qid, target_qid in zip(ctx.controls, ctx.targets):
            target_loc = target[target_qid]
            partner = ctx.arch_spec.get_cz_partner(target_loc)
            assert partner is not None, f"No CZ blockade partner for {target_loc}"
            target[control_qid] = partner
        return [target]


TargetGeneratorCallable = Callable[[TargetContext], list[dict[int, LocationAddress]]]


@dataclass(frozen=True)
class _CallableTargetGenerator(TargetGeneratorABC):
    """Private adapter that lifts a bare callable to TargetGeneratorABC."""

    fn: TargetGeneratorCallable

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        return self.fn(ctx)


def _coerce_target_generator(
    value: TargetGeneratorABC | TargetGeneratorCallable | None,
) -> TargetGeneratorABC | None:
    """Normalize the public union down to TargetGeneratorABC | None."""
    if value is None or isinstance(value, TargetGeneratorABC):
        return value
    return _CallableTargetGenerator(value)


def _validate_candidate(
    ctx: TargetContext,
    candidate: dict[int, LocationAddress],
) -> None:
    """Raise ValueError if the candidate is not a legal CZ target.

    Checks:
    1. Every qid from ``ctx.placement`` appears in ``candidate``; no
       unexpected extra qids are present.
    2. Every location value is recognized by ``ctx.arch_spec`` (via
       ``check_location_group``). Group-level errors such as duplicate
       locations are caught here.
    3. Each ``(control_qid, target_qid)`` pair is CZ-blockade-partnered
       in either direction (matching the convention at
       ``python/bloqade/lanes/analysis/placement/lattice.py:134-135``).
    """
    placement = ctx.placement
    missing = set(placement.keys()) - set(candidate.keys())
    extra = set(candidate.keys()) - set(placement.keys())
    if missing or extra:
        parts: list[str] = []
        if missing:
            parts.append(f"missing {sorted(missing)}")
        if extra:
            parts.append(f"unexpected {sorted(extra)}")
        raise ValueError(f"target-generator candidate qubits: {'; '.join(parts)}")
    # Run the Rust-backed validator on the full candidate so group-level
    # errors (e.g. duplicate locations) are caught, then attribute per-qid
    # where possible for a helpful error message.
    group_errors = list(ctx.arch_spec.check_location_group(list(candidate.values())))
    if group_errors:
        per_qid_bad = [
            f"qid={qid} @ {loc}"
            for qid, loc in candidate.items()
            if ctx.arch_spec.check_location_group([loc])
        ]
        detail = per_qid_bad if per_qid_bad else [str(e) for e in group_errors]
        raise ValueError(f"target-generator candidate has invalid locations: {detail}")
    for control_qid, target_qid in zip(ctx.controls, ctx.targets):
        c_loc = candidate[control_qid]
        t_loc = candidate[target_qid]
        if (
            ctx.arch_spec.get_cz_partner(c_loc) != t_loc
            and ctx.arch_spec.get_cz_partner(t_loc) != c_loc
        ):
            raise ValueError(
                f"target-generator candidate CZ pair "
                f"(control={control_qid}@{c_loc}, target={target_qid}@{t_loc}) "
                f"is not blockade-partnered"
            )


_LaneKey = tuple[MoveType, int, int, int, int]


def _lane_key(lane: layout.LaneAddress) -> _LaneKey:
    """Direction-independent canonical key for a lane.

    A physical lane used FORWARD and BACKWARD is the same resource;
    `_lane_key` strips `direction` so the two map to the same key.
    """
    return (
        lane.move_type,
        lane.word_id,
        lane.site_id,
        lane.bus_id,
        lane.zone_id,
    )


def _sum_base(
    path: tuple[
        tuple[layout.LaneAddress, ...],
        tuple[layout.LocationAddress, ...],
    ],
    pf: layout.PathFinder,
) -> float:
    """Sum of base (no-penalty) lane-duration cost over a path's lanes."""
    return sum(pf.metrics.get_lane_duration_cost(lane) for lane in path[0])


def _sum_weighted(
    path: tuple[
        tuple[layout.LaneAddress, ...],
        tuple[layout.LocationAddress, ...],
    ],
    weight_fn: Callable[[layout.LaneAddress], float],
) -> float:
    """Sum of `weight_fn(lane)` over a path's lanes.

    Returns 0.0 for zero-length paths (empty lane tuple).
    """
    return sum(weight_fn(lane) for lane in path[0])


def _choose_control(cost_c: float, cost_t: float, len_c: float, len_t: float) -> bool:
    """Tiebreak comparator. Returns True iff control-direction wins.

    Hierarchy:
      1. lower cost wins
      2. on cost tie, shorter path wins (fewer physical moves)
      3. on both-tie, prefer control (parity with DefaultTargetGenerator)
    """
    if cost_c < cost_t:
        return True
    if cost_t < cost_c:
        return False
    if len_c < len_t:
        return True
    if len_t < len_c:
        return False
    return True


class _HasPenalties(Protocol):
    opposite_direction_penalty: float
    same_direction_penalty: float
    shared_site_penalty: float


def _make_weight_fn(
    pf: layout.PathFinder,
    committed_lanes: dict[_LaneKey, layout.Direction],
    committed_sites: set[layout.LocationAddress],
    gen: _HasPenalties,
) -> Callable[[layout.LaneAddress], float]:
    """Closure over the running congestion state; passed to `find_path`.

    Penalty precedence (largest to smallest):
      1. lane reused in opposite direction -> opposite_direction_penalty
      2. lane reused in same direction    -> same_direction_penalty
      3. lane not reused, but an endpoint is in committed_sites
                                          -> shared_site_penalty
      4. no overlap                        -> 0 (base only)
    """

    def weight(lane: layout.LaneAddress) -> float:
        base = pf.metrics.get_lane_duration_cost(lane)
        key = _lane_key(lane)
        prior_dir = committed_lanes.get(key)
        if prior_dir is not None:
            if prior_dir != lane.direction:
                return base + gen.opposite_direction_penalty
            return base + gen.same_direction_penalty
        src, dst = pf.get_endpoints(lane)
        if src in committed_sites or dst in committed_sites:
            return base + gen.shared_site_penalty
        return base

    return weight


@dataclass(frozen=True)
class CongestionAwareTargetGenerator(TargetGeneratorABC):
    """Joint, longest-first, congestion-aware target generator.

    For each CZ pair, picks whether to move the control or the target
    based on schedule-time cost computed against a working placement
    that reflects all prior pairs' committed moves and a running
    directional congestion record.

    Penalty weights are additive on top of
    ``MoveMetricCalculator.get_lane_duration_cost(lane)`` and are tuned
    relative to that base-duration scale. Defaults are illustrative;
    empirical tuning is a documented follow-up.
    """

    opposite_direction_penalty: float = 10.0
    same_direction_penalty: float = 1.0
    shared_site_penalty: float = 0.1

    def generate(self, ctx: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        placement = ctx.placement
        if not ctx.controls:
            return [dict(placement)]

        pf = layout.PathFinder(ctx.arch_spec)
        working: dict[int, layout.LocationAddress] = dict(placement)
        committed_lanes: dict[_LaneKey, layout.Direction] = {}
        committed_sites: set[layout.LocationAddress] = set()

        pairs = self._sort_pairs_longest_first(ctx, pf)

        for ctrl, tgt in pairs:
            result = self._commit_pair(
                ctx.arch_spec,
                pf,
                working,
                committed_lanes,
                committed_sites,
                ctrl,
                tgt,
            )
            if result is None:
                return []  # both directions infeasible -> fallback to default
            mover, new_loc, chosen = result
            working[mover] = new_loc
            for lane in chosen[0]:
                committed_lanes[_lane_key(lane)] = lane.direction
            committed_sites.update(chosen[1])

        return [working]

    def _sort_pairs_longest_first(
        self, ctx: TargetContext, pf: layout.PathFinder
    ) -> list[tuple[int, int]]:
        placement = ctx.placement
        arch = ctx.arch_spec

        def score(pair: tuple[int, int]) -> float:
            ctrl, tgt = pair
            ctrl_loc = placement[ctrl]
            tgt_loc = placement[tgt]
            ctrl_partner = arch.get_cz_partner(tgt_loc)
            tgt_partner = arch.get_cz_partner(ctrl_loc)
            assert ctrl_partner is not None, f"No CZ partner for qid={tgt} at {tgt_loc}"
            assert (
                tgt_partner is not None
            ), f"No CZ partner for qid={ctrl} at {ctrl_loc}"
            occ_ctrl = frozenset(loc for q, loc in placement.items() if q != ctrl)
            occ_tgt = frozenset(loc for q, loc in placement.items() if q != tgt)
            p_ctrl = pf.find_path(ctrl_loc, ctrl_partner, occupied=occ_ctrl)
            p_tgt = pf.find_path(tgt_loc, tgt_partner, occupied=occ_tgt)
            c_ctrl = _sum_base(p_ctrl, pf) if p_ctrl is not None else math.inf
            c_tgt = _sum_base(p_tgt, pf) if p_tgt is not None else math.inf
            return min(c_ctrl, c_tgt)

        pairs = list(zip(ctx.controls, ctx.targets))
        pairs.sort(key=score, reverse=True)
        return pairs

    def _commit_pair(
        self,
        arch_spec: layout.ArchSpec,
        pf: layout.PathFinder,
        working: dict[int, layout.LocationAddress],
        committed_lanes: dict[_LaneKey, layout.Direction],
        committed_sites: set[layout.LocationAddress],
        ctrl: int,
        tgt: int,
    ) -> (
        tuple[
            int,
            layout.LocationAddress,
            tuple[
                tuple[layout.LaneAddress, ...],
                tuple[layout.LocationAddress, ...],
            ],
        ]
        | None
    ):
        # Placeholder: picks control direction, no path computation.
        # Task 7 replaces this with real direction choice.
        ctrl_loc = working[ctrl]
        tgt_loc = working[tgt]
        partner = arch_spec.get_cz_partner(tgt_loc)
        assert partner is not None, f"No CZ partner for qid={tgt} at {tgt_loc}"
        if ctrl_loc == partner:
            # Already partnered; trivial zero-length path at ctrl_loc.
            return (ctrl, partner, ((), (ctrl_loc,)))
        # Walk a real path so the already-partnered case works end-to-end
        # even before Task 7 lands.
        occ = frozenset(loc for q, loc in working.items() if q != ctrl)
        path = pf.find_path(ctrl_loc, partner, occupied=occ)
        if path is None:
            return None
        return (ctrl, partner, path)
