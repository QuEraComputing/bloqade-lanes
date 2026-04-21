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


class _HasFactors(Protocol):
    @property
    def opposite_direction_factor(self) -> float: ...
    @property
    def same_direction_factor(self) -> float: ...
    @property
    def shared_site_factor(self) -> float: ...


def _make_weight_fn(
    pf: layout.PathFinder,
    committed_lanes: dict[_LaneKey, layout.Direction],
    committed_sites: set[layout.LocationAddress],
    gen: _HasFactors,
) -> Callable[[layout.LaneAddress], float]:
    """Closure over the running congestion state; passed to `find_path`.

    Factor precedence (which field multiplies the base lane cost):
      1. lane reused in opposite direction -> opposite_direction_factor
      2. lane reused in same direction    -> same_direction_factor
      3. lane not reused, but an endpoint is in committed_sites
                                          -> shared_site_factor
      4. no overlap                        -> 1.0 (base only)

    Each field is a non-negative multiplier applied to the base
    ``get_lane_duration_cost``. Values ``< 1`` act as bonuses (reward
    that case), ``> 1`` act as penalties, ``= 1`` is neutral. Factors
    must be non-negative for the Dijkstra-based path finder. Canonical
    tuning: ``opposite_direction_factor`` large (penalty), ``same_direction_factor``
    below 1 (reward AOD-parallel moves on a shared bus), ``shared_site_factor``
    slightly above 1 (mild crossing penalty).
    """

    def weight(lane: layout.LaneAddress) -> float:
        base = pf.metrics.get_lane_duration_cost(lane)
        key = _lane_key(lane)
        prior_dir = committed_lanes.get(key)
        if prior_dir is not None:
            if prior_dir != lane.direction:
                return base * gen.opposite_direction_factor
            return base * gen.same_direction_factor
        src, dst = pf.get_endpoints(lane)
        if src in committed_sites or dst in committed_sites:
            return base * gen.shared_site_factor
        return base

    return weight


@dataclass
class _GenerateState:
    """Mutable state threaded through the congestion-aware commit loop.

    ``arch_spec`` and ``pf`` are fixed for the whole ``generate()`` call;
    ``working``, ``committed_lanes``, and ``committed_sites`` accumulate
    as pairs commit.
    """

    arch_spec: layout.ArchSpec
    pf: layout.PathFinder
    working: dict[int, layout.LocationAddress]
    committed_lanes: dict[_LaneKey, layout.Direction]
    committed_sites: set[layout.LocationAddress]


@dataclass(frozen=True)
class CongestionAwareTargetGenerator(TargetGeneratorABC):
    """Joint, longest-first, congestion-aware target generator.

    For each CZ pair, picks whether to move the control or the target
    based on schedule-time cost computed against a working placement
    that reflects all prior pairs' committed moves and a running
    directional congestion record.

    Factor fields are non-negative multipliers on
    ``MoveMetricCalculator.get_lane_duration_cost(lane)``. Values below
    1.0 reward the case (cheaper than base), above 1.0 penalize, and
    1.0 is neutral. Dijkstra requires non-negative edge weights, so
    factors must remain ``>= 0``. Defaults reflect the canonical
    tuning — same-direction lane reuse rewarded (AOD transport layer
    parallelism), opposite-direction reuse penalized, shared-site
    crossings mildly penalized. Empirical tuning is a documented
    follow-up.
    """

    opposite_direction_factor: float = 10.0
    same_direction_factor: float = 0.25
    shared_site_factor: float = 1.1

    def generate(self, ctx: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        placement = ctx.placement
        if not ctx.controls:
            return [dict(placement)]

        pf = layout.PathFinder(ctx.arch_spec)
        state = _GenerateState(
            arch_spec=ctx.arch_spec,
            pf=pf,
            working=dict(placement),
            committed_lanes={},
            committed_sites=set(),
        )

        pairs = self._sort_pairs_longest_first(ctx, pf)

        for ctrl, tgt in pairs:
            result = self._commit_pair(state, ctrl, tgt)
            if result is None:
                return []  # both directions infeasible -> fallback to default
            mover, new_loc, chosen = result
            state.working[mover] = new_loc
            for lane in chosen[0]:
                state.committed_lanes[_lane_key(lane)] = lane.direction
            state.committed_sites.update(chosen[1])

        return [state.working]

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
        state: _GenerateState,
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
        ctrl_loc = state.working[ctrl]
        tgt_loc = state.working[tgt]
        ctrl_partner = state.arch_spec.get_cz_partner(tgt_loc)
        tgt_partner = state.arch_spec.get_cz_partner(ctrl_loc)
        assert ctrl_partner is not None, f"No CZ partner for qid={tgt} at {tgt_loc}"
        assert tgt_partner is not None, f"No CZ partner for qid={ctrl} at {ctrl_loc}"

        occ_ctrl = frozenset(loc for q, loc in state.working.items() if q != ctrl)
        occ_tgt = frozenset(loc for q, loc in state.working.items() if q != tgt)

        weight = _make_weight_fn(
            state.pf, state.committed_lanes, state.committed_sites, self
        )
        path_ctrl = state.pf.find_path(
            ctrl_loc, ctrl_partner, occupied=occ_ctrl, edge_weight=weight
        )
        path_tgt = state.pf.find_path(
            tgt_loc, tgt_partner, occupied=occ_tgt, edge_weight=weight
        )

        cost_ctrl = _sum_weighted(path_ctrl, weight) if path_ctrl else math.inf
        cost_tgt = _sum_weighted(path_tgt, weight) if path_tgt else math.inf

        if cost_ctrl == math.inf and cost_tgt == math.inf:
            return None

        len_ctrl = float(len(path_ctrl[0])) if path_ctrl else math.inf
        len_tgt = float(len(path_tgt[0])) if path_tgt else math.inf

        if _choose_control(cost_ctrl, cost_tgt, len_ctrl, len_tgt):
            assert path_ctrl is not None
            return (ctrl, ctrl_partner, path_ctrl)
        assert path_tgt is not None
        return (tgt, tgt_partner, path_tgt)


# (move_type, zone_id, bus_id, direction) — the scheduler's batching key.
# Lanes sharing this tuple are physically packable into one AOD shot (see
# ArchSpec.check_lane_group in python/bloqade/lanes/layout/arch.py and
# BusContext in python/bloqade/lanes/search/generators/aod_grouping.py).
_AODSig = tuple[MoveType, int, int, layout.Direction]


def _first_hop_sig(
    path: (
        tuple[
            tuple[layout.LaneAddress, ...],
            tuple[layout.LocationAddress, ...],
        ]
        | None
    ),
) -> _AODSig | None:
    """Signature of a path's first hop. ``None`` if the path is missing or empty.

    An empty path means the qubit is already at its destination — no
    lane to cluster.
    """
    if path is None:
        return None
    lanes = path[0]
    if not lanes:
        return None
    lane = lanes[0]
    return (lane.move_type, lane.zone_id, lane.bus_id, lane.direction)


@dataclass(frozen=True)
class AODClusterTargetGenerator(TargetGeneratorABC):
    """Choose CZ directions to maximise AOD-shot packing.

    For each CZ pair, enumerate both direction candidates (control-moves
    or target-moves) and classify each by the ``(move_type, zone_id,
    bus_id, direction)`` signature of its first path hop — the same
    signature the downstream move scheduler uses to batch lanes into
    one AOD shot. Directions are then assigned greedily: the signature
    appearing in the most candidate first-hops is filled first, then
    the next largest, and so on. When no further clustering gain is
    possible (largest remaining bucket has a single unresolved pair),
    remaining pairs default to control-direction for parity with
    :class:`DefaultTargetGenerator`.

    Rationale: CongAware rewards same-direction lane reuse per edge
    via a Dijkstra weight. That is a local proxy for shot-sharing; the
    scheduler actually batches by the triple above, not by individual
    lane reuse. Choosing CZ directions that align first-hops on shared
    signatures produces targets the scheduler can pack more tightly.
    """

    def generate(self, ctx: TargetContext) -> list[dict[int, layout.LocationAddress]]:
        placement = ctx.placement
        if not ctx.controls:
            return [dict(placement)]

        pf = layout.PathFinder(ctx.arch_spec)
        pairs = list(zip(ctx.controls, ctx.targets))

        # Classification pass. A pair is either:
        #  - forced to one direction (other infeasible, or other path empty);
        #  - contributing two (pair_idx, direction, sig) entries to buckets.
        forced: dict[int, str] = {}
        partners: list[tuple[layout.LocationAddress, layout.LocationAddress]] = []
        bucket_entries: list[tuple[int, str, _AODSig]] = []

        for i, (ctrl, tgt) in enumerate(pairs):
            ctrl_loc = placement[ctrl]
            tgt_loc = placement[tgt]
            ctrl_partner = ctx.arch_spec.get_cz_partner(tgt_loc)
            tgt_partner = ctx.arch_spec.get_cz_partner(ctrl_loc)
            assert ctrl_partner is not None, f"No CZ partner for qid={tgt} at {tgt_loc}"
            assert (
                tgt_partner is not None
            ), f"No CZ partner for qid={ctrl} at {ctrl_loc}"
            partners.append((ctrl_partner, tgt_partner))

            occ_ctrl = frozenset(loc for q, loc in placement.items() if q != ctrl)
            occ_tgt = frozenset(loc for q, loc in placement.items() if q != tgt)
            path_ctrl = pf.find_path(ctrl_loc, ctrl_partner, occupied=occ_ctrl)
            path_tgt = pf.find_path(tgt_loc, tgt_partner, occupied=occ_tgt)

            sig_c = _first_hop_sig(path_ctrl)
            sig_t = _first_hop_sig(path_tgt)
            ctrl_feasible = path_ctrl is not None
            tgt_feasible = path_tgt is not None

            if not ctrl_feasible and not tgt_feasible:
                return []  # defer to Default
            if not ctrl_feasible:
                forced[i] = "tgt"
                continue
            if not tgt_feasible:
                forced[i] = "ctrl"
                continue
            # Empty path => already at destination. Prefer such a
            # direction so it doesn't pollute the clustering signal.
            if sig_c is None and sig_t is None:
                forced[i] = "ctrl"  # parity with Default
                continue
            if sig_c is None:
                forced[i] = "ctrl"
                continue
            if sig_t is None:
                forced[i] = "tgt"
                continue

            bucket_entries.append((i, "ctrl", sig_c))
            bucket_entries.append((i, "tgt", sig_t))

        buckets: dict[_AODSig, list[tuple[int, str]]] = {}
        for i, d, sig in bucket_entries:
            buckets.setdefault(sig, []).append((i, d))

        resolved: dict[int, str] = dict(forced)

        # Greedy fill: commit the bucket with the most unresolved pairs.
        # Stop once no bucket has more than one unresolved pair — the
        # remaining pairs can't cluster further, so committing arbitrary
        # directions for them doesn't help (and control-direction parity
        # with Default is a reasonable tiebreak).
        while True:
            best_sig: _AODSig | None = None
            best_count = 1  # strict > 1 required
            for sig, entries in buckets.items():
                count = sum(1 for (i, _d) in entries if i not in resolved)
                if count > best_count:
                    best_count = count
                    best_sig = sig
            if best_sig is None:
                break
            for i, d in buckets[best_sig]:
                if i not in resolved:
                    resolved[i] = d

        for i in range(len(pairs)):
            if i not in resolved:
                resolved[i] = "ctrl"

        target = dict(placement)
        for i, direction in resolved.items():
            ctrl, tgt = pairs[i]
            ctrl_partner, tgt_partner = partners[i]
            if direction == "ctrl":
                target[ctrl] = ctrl_partner
            else:
                target[tgt] = tgt_partner
        return [target]
