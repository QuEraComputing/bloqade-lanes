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
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.lanes.arch.path import PathFinder
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode.encoding import (
    Direction,
    LaneAddress,
    LocationAddress,
    MoveType,
)

_PairSide = Literal["ctrl", "tgt"]
_Path = tuple[tuple[LaneAddress, ...], tuple[LocationAddress, ...]]
_LaneDirCounts = Counter[Direction]


@dataclass(frozen=True)
class TargetContext:
    """Signals passed to a TargetGenerator.

    Composes ConcreteState to avoid duplicating lattice state fields.
    """

    arch_spec: ArchSpec
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


def _lane_key(lane: LaneAddress) -> _LaneKey:
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


def _sum_base(path: _Path, pf: PathFinder) -> float:
    """Sum of base (no-penalty) lane-duration cost over a path's lanes."""
    return sum(pf.metrics.get_lane_duration_cost(lane) for lane in path[0])


def _sum_weighted(path: _Path, weight_fn: Callable[[LaneAddress], float]) -> float:
    """Sum of `weight_fn(lane)` over a path's lanes.

    Returns 0.0 for zero-length paths (empty lane tuple).
    """
    return sum(weight_fn(lane) for lane in path[0])


def _probe_pair(
    arch_spec: ArchSpec,
    pf: PathFinder,
    placement: Mapping[int, LocationAddress],
    ctrl: int,
    tgt: int,
    edge_weight: Callable[[LaneAddress], float] | None = None,
) -> tuple[
    LocationAddress,
    LocationAddress,
    _Path | None,
    _Path | None,
]:
    """Compute CZ partners and the two candidate paths for pair ``(ctrl, tgt)``.

    Returns ``(ctrl_partner, tgt_partner, path_ctrl, path_tgt)``. The
    control-direction path moves ``ctrl`` to ``ctrl_partner``; the
    target-direction path moves ``tgt`` to ``tgt_partner``. Each path
    is ``None`` if infeasible under the current occupancy.

    Locations are unique per placement (one atom per site), so the
    occupancy frozensets can be derived by set difference from a single
    shared base set rather than re-iterating placement per direction.
    """
    ctrl_loc = placement[ctrl]
    tgt_loc = placement[tgt]
    ctrl_partner = arch_spec.get_cz_partner(tgt_loc)
    tgt_partner = arch_spec.get_cz_partner(ctrl_loc)
    assert ctrl_partner is not None, f"No CZ partner for qid={tgt} at {tgt_loc}"
    assert tgt_partner is not None, f"No CZ partner for qid={ctrl} at {ctrl_loc}"

    occupied_all = frozenset(placement.values())
    path_ctrl = pf.find_path(
        ctrl_loc,
        ctrl_partner,
        occupied=occupied_all - {ctrl_loc},
        edge_weight=edge_weight,
    )
    path_tgt = pf.find_path(
        tgt_loc,
        tgt_partner,
        occupied=occupied_all - {tgt_loc},
        edge_weight=edge_weight,
    )
    return ctrl_partner, tgt_partner, path_ctrl, path_tgt


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
    def direction_factor(self) -> float: ...
    @property
    def shared_site_factor(self) -> float: ...


def _make_weight_fn(
    pf: PathFinder,
    committed_lanes: dict[_LaneKey, _LaneDirCounts],
    committed_sites: set[LocationAddress],
    gen: _HasFactors,
) -> Callable[[LaneAddress], float]:
    """Closure over the running congestion state; passed to `find_path`.

    The two signals — direction reuse and shared-site crossings —
    measure orthogonal physical phenomena (AOD shot packing on the
    lane itself vs path waypoints in transit) and compose
    multiplicatively:

        weight = base * direction_factor ** (N - M)
                      * shared_site_factor (iff an endpoint was
                                            previously traversed)

    where ``N`` and ``M`` are prior same- and opposite-direction commit
    counts on the lane. Reward and penalty share the direction exponent:
    ``direction_factor ** N`` rewards same-direction AOD-sharing,
    ``direction_factor ** -M`` penalises opposite-direction conflicts.
    Balanced traffic (``N == M``) zeroes the direction exponent, which
    on its own is neutral but still lets a coincident shared-site
    crossing contribute.

    ``direction_factor`` must be strictly positive (Dijkstra requires
    non-negative edges, and ``0 ** -M`` is undefined).
    ``shared_site_factor`` must be non-negative.
    """

    df = gen.direction_factor

    def weight(lane: LaneAddress) -> float:
        base = pf.metrics.get_lane_duration_cost(lane)
        factor = 1.0
        counts = committed_lanes.get(_lane_key(lane))
        if counts:
            net = counts.get(lane.direction, 0) - counts.get(
                _opposite(lane.direction), 0
            )
            if net != 0:
                factor *= df**net
        src, dst = pf.get_endpoints(lane)
        if src in committed_sites or dst in committed_sites:
            factor *= gen.shared_site_factor
        return base * factor

    return weight


def _opposite(direction: Direction) -> Direction:
    return Direction.BACKWARD if direction == Direction.FORWARD else Direction.FORWARD


@dataclass
class _GenerateState:
    """Mutable state threaded through the congestion-aware commit loop.

    ``arch_spec`` and ``pf`` are fixed for the whole ``generate()`` call;
    ``working``, ``committed_lanes``, and ``committed_sites`` accumulate
    as pairs commit. ``committed_lanes`` tracks a per-direction count so
    the weight function can reward strong same-direction clusters and
    penalise mixed-direction contention multiplicatively.
    """

    arch_spec: ArchSpec
    pf: PathFinder
    working: dict[int, LocationAddress]
    committed_lanes: dict[_LaneKey, _LaneDirCounts]
    committed_sites: set[LocationAddress]


@dataclass(frozen=True)
class CongestionAwareTargetGenerator(TargetGeneratorABC):
    """Joint, longest-first, congestion-aware target generator.

    For each CZ pair, picks whether to move the control or the target
    based on schedule-time cost computed against a working placement
    that reflects all prior pairs' committed moves and a running
    directional congestion record.

    Per-lane weighting composes two orthogonal multiplicative factors
    (see :func:`_make_weight_fn`):

    * ``direction_factor ** (N - M)`` — signed net of same-direction
      (N) minus opposite-direction (M) prior commits on the lane. With
      ``direction_factor < 1``, net-positive traffic rewards AOD-parallel
      reuse and net-negative traffic penalises contention; balanced
      traffic (``N == M``) is neutral.
    * ``shared_site_factor`` — applied whenever an endpoint of the
      candidate lane is a site a prior committed path already traversed.
      Applies independently of ``direction_factor``; the two signals
      compose multiplicatively.

    Dijkstra requires non-negative edge weights. ``direction_factor``
    must be strictly positive (the negative exponent is otherwise
    undefined); ``shared_site_factor`` must be ``>= 0``. Defaults
    reflect the canonical tuning; empirical retuning is a follow-up.
    """

    direction_factor: float = 0.5
    shared_site_factor: float = 1.1

    def __post_init__(self) -> None:
        if self.direction_factor <= 0:
            raise ValueError(
                f"direction_factor={self.direction_factor!r} must be strictly "
                f"positive; the opposite-direction branch raises it to a "
                f"negative exponent which is undefined at zero"
            )
        if self.shared_site_factor < 0:
            raise ValueError(
                f"shared_site_factor={self.shared_site_factor!r} must be "
                f"non-negative; Dijkstra requires non-negative edge weights"
            )

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        placement = ctx.placement
        if not ctx.controls:
            return [dict(placement)]

        pf = PathFinder(ctx.arch_spec)
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
                return []
            mover, new_loc, chosen = result
            state.working[mover] = new_loc
            for lane in chosen[0]:
                counts = state.committed_lanes.setdefault(_lane_key(lane), Counter())
                counts[lane.direction] += 1
            state.committed_sites.update(chosen[1])

        return [state.working]

    def _sort_pairs_longest_first(
        self, ctx: TargetContext, pf: PathFinder
    ) -> list[tuple[int, int]]:
        placement = ctx.placement
        arch = ctx.arch_spec

        def score(pair: tuple[int, int]) -> float:
            ctrl, tgt = pair
            _, _, p_ctrl, p_tgt = _probe_pair(arch, pf, placement, ctrl, tgt)
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
    ) -> tuple[int, LocationAddress, _Path] | None:
        weight = _make_weight_fn(
            state.pf, state.committed_lanes, state.committed_sites, self
        )
        ctrl_partner, tgt_partner, path_ctrl, path_tgt = _probe_pair(
            state.arch_spec,
            state.pf,
            state.working,
            ctrl,
            tgt,
            edge_weight=weight,
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
# ArchSpec.check_lane_group in python/bloqade/lanes/layout/arch.py).
_AODSig = tuple[MoveType, int, int, Direction]


def _first_hop_sig(path: _Path | None) -> _AODSig | None:
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
    scheduler actually batches by the 4-tuple above, not by individual
    lane reuse. Choosing CZ directions that align first-hops on shared
    signatures produces targets the scheduler can pack more tightly.
    """

    def generate(self, ctx: TargetContext) -> list[dict[int, LocationAddress]]:
        placement = ctx.placement
        if not ctx.controls:
            return [dict(placement)]

        pf = PathFinder(ctx.arch_spec)
        pairs = list(zip(ctx.controls, ctx.targets))

        # One entry per pair: chosen side (or None = still to be decided),
        # the two partner locations to commit against, and — for pairs
        # still in the greedy pool — their two candidate signatures.
        decisions: list[_PairSide | None] = [None] * len(pairs)
        partners: list[tuple[LocationAddress, LocationAddress]] = []
        pair_sigs: dict[int, tuple[_AODSig, _AODSig]] = {}
        buckets: dict[_AODSig, list[tuple[int, _PairSide]]] = {}
        live_count: dict[_AODSig, int] = {}

        for i, (ctrl, tgt) in enumerate(pairs):
            ctrl_partner, tgt_partner, path_ctrl, path_tgt = _probe_pair(
                ctx.arch_spec, pf, placement, ctrl, tgt
            )
            partners.append((ctrl_partner, tgt_partner))

            sig_c = _first_hop_sig(path_ctrl)
            sig_t = _first_hop_sig(path_tgt)
            ctrl_feasible = path_ctrl is not None
            tgt_feasible = path_tgt is not None

            if not ctrl_feasible and not tgt_feasible:
                return []
            if not ctrl_feasible:
                decisions[i] = "tgt"
                continue
            if not tgt_feasible:
                decisions[i] = "ctrl"
                continue
            # Empty path (``sig is None``) means the qubit is already at
            # its destination: no lane to cluster, no pollution of bucket
            # signals. Commit that side immediately; default to ctrl when
            # both sides are trivial, matching DefaultTargetGenerator.
            if sig_t is None:
                decisions[i] = "ctrl"
                continue
            if sig_c is None:
                decisions[i] = "tgt"
                continue

            pair_sigs[i] = (sig_c, sig_t)
            buckets.setdefault(sig_c, []).append((i, "ctrl"))
            buckets.setdefault(sig_t, []).append((i, "tgt"))
            live_count[sig_c] = live_count.get(sig_c, 0) + 1
            live_count[sig_t] = live_count.get(sig_t, 0) + 1

        # Greedy fill: the signature with the most unresolved candidates
        # wins, and every pair in its bucket commits to that side. Once
        # no bucket has more than one live candidate, remaining pairs
        # can't cluster further, so they default to ctrl-direction for
        # parity with DefaultTargetGenerator.
        while True:
            best_sig: _AODSig | None = None
            best_count = 1
            for sig, count in live_count.items():
                if count > best_count:
                    best_count = count
                    best_sig = sig
            if best_sig is None:
                break
            for i, side in buckets[best_sig]:
                if decisions[i] is not None:
                    continue
                decisions[i] = side
                sig_c, sig_t = pair_sigs[i]
                live_count[sig_c] -= 1
                live_count[sig_t] -= 1

        target = dict(placement)
        for i, (ctrl, tgt) in enumerate(pairs):
            side = decisions[i] if decisions[i] is not None else "ctrl"
            ctrl_partner, tgt_partner = partners[i]
            if side == "ctrl":
                target[ctrl] = ctrl_partner
            else:
                target[tgt] = tgt_partner
        return [target]
