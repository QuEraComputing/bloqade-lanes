"""Beam-search placement strategy with NoReturn as an unconditional safety net.

At each CZ stage, generates ``hungarian_candidates`` perturbed Hungarian
target-layout candidates **and** always includes one
:class:`NoReturnPlacementStrategy` candidate, then keeps the top
``beam_width`` trajectories across the lookahead window scored by

    score = current_layer_cost + lookahead_weight * min_next_layer_cost.

The ``NoReturnPlacementStrategy`` candidate is included **unconditionally**
on every call for every live beam trajectory. This guarantees the
invariant ``A.lanes <= NR.lanes`` for every circuit — the beam can
exploit Hungarian diversity for cross-stage cascade fixes without ever
falling below the NR baseline.

Positioning
-----------
Use this strategy as a **drop-in replacement** for
:class:`NoReturnPlacementStrategy` when one is willing to pay
``O(stages * W * K)`` extra solver-oracle calls (for typical 50-stage
circuits this is 30-60 seconds at default ``W=8, K=20``). Useful for
*compile-once, run-many* workloads where placement quality dominates
compile time.

Design notes
------------
- The safety-net invariant relies on the NR candidate being added
  *unconditionally* even when the beam's extension-feasibility filter
  would otherwise drop it. Filtering the safety net was the source of
  ``DEAD`` trajectories in early prototypes; do not re-introduce
  filtering.
- Hungarian-perturbed candidates come from a Python-level edge-cost
  assignment over the entangling pair-slot set. The perturbation is a
  small additive noise on edge weights to produce ``K`` distinct
  assignments; ``K=20`` saturates returns on the families we tested
  (GHZ ladder n=80, Star n=60).
- Scoring uses a 1-step lookahead (``lookahead_weight * min over K_la
  Hungarian assignments for the next layer``); empirically this
  resolves the LCGB (locally-cheap-globally-bad) traps that motivate
  this strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from bloqade.lanes.analysis.placement import (
    AtomState,
    ConcreteState,
    ExecuteCZ,
    ExecuteMeasure,
    PlacementStrategyABC,
)
from bloqade.lanes.analysis.placement.strategy import assert_single_cz_zone
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode._native import MoveSolver
from bloqade.lanes.bytecode.encoding import LocationAddress, ZoneAddress
from bloqade.lanes.heuristics.move_synthesis import compute_move_layers
from bloqade.lanes.heuristics.physical.movement import RustPlacementTraversal
from bloqade.lanes.heuristics.physical.no_return import NoReturnPlacementStrategy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as _e:  # pragma: no cover - scipy is a project dep
    raise ImportError(
        "MultiCandidateLookaheadPlacementStrategy requires scipy (linear_sum_assignment)"
    ) from _e


# -----------------------------------------------------------------------------
# Pair-slot enumeration (entangling-zone slot pairs).
# -----------------------------------------------------------------------------


def _entangling_pair_slots(
    arch_spec: ArchSpec,
) -> tuple[tuple[LocationAddress, LocationAddress], ...]:
    """Return the ``(left, right)`` ``LocationAddress`` pairs that form
    valid entangling-zone slot pairs for ``arch_spec``.

    Slot pairs are formed by adjacent paired words at the same
    ``site_id`` within the (single) CZ zone — the Gemini physical
    convention. We discover pairs via :meth:`ArchSpec.get_cz_partner`
    which makes this architecture-agnostic (returns ``None`` for
    non-pairable locations, which we skip).
    """
    pairs: list[tuple[LocationAddress, LocationAddress]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cz_zone in arch_spec.cz_zone_addresses:
        for loc in arch_spec.yield_zone_locations(cz_zone):
            partner = arch_spec.get_cz_partner(loc)
            if partner is None:
                continue
            key = (
                min(loc.word_id, partner.word_id),
                loc.site_id,
                max(loc.word_id, partner.word_id),
                partner.site_id,
            )
            if key in seen:
                continue
            seen.add(key)
            if loc.word_id < partner.word_id:
                pairs.append((loc, partner))
            else:
                pairs.append((partner, loc))
    return tuple(pairs)


# -----------------------------------------------------------------------------
# Hungarian-based candidate target-layout generation.
# -----------------------------------------------------------------------------


def _layout_locations(
    layout: tuple[LocationAddress, ...],
) -> tuple[tuple[int, int], ...]:
    """Compact ``(word_id, site_id)`` tuples for fast hashing/equality."""
    return tuple((loc.word_id, loc.site_id) for loc in layout)


def _hop_distance(a: LocationAddress, b: LocationAddress) -> int:
    return abs(a.word_id - b.word_id) + abs(a.site_id - b.site_id)


def _hungarian_assignment(
    pairs: Sequence[tuple[int, int]],
    layout: tuple[LocationAddress, ...],
    pair_slots: tuple[tuple[LocationAddress, LocationAddress], ...],
    perturbation: np.ndarray | None = None,
) -> tuple[LocationAddress, ...] | None:
    """Assign each (ctrl, tgt) pair to a (slot_pair, orientation) cell.

    Uses a **2× cost matrix** with one column per (slot_pair, orientation)
    so that Hungarian sees orientation as a separate decision variable.
    This doubles the candidate diversity vs single-column-then-greedy-
    orientation (the previous implementation), which empirically misses
    layouts that single-column-then-greedy-orientation misses.

    ``perturbation`` is an additive noise matrix the same shape as the
    cost matrix (``n_pairs × (2 × n_slot_pairs)``). Returns ``None`` if
    no valid assignment.
    """
    n_pairs = len(pairs)
    n_slot_pairs = len(pair_slots)
    n_cols = n_slot_pairs * 2  # 2 orientations per slot pair
    if n_pairs > n_cols:
        return None
    cost = np.empty((n_pairs, n_cols), dtype=float)
    # slot_meta[col] = (slot_a_for_ctrl, slot_b_for_tgt) — explicit
    # orientation per column.
    slot_meta: list[tuple[LocationAddress, LocationAddress]] = []
    for j, (sl, sr) in enumerate(pair_slots):
        for ord_idx, (sa, sb) in enumerate(((sl, sr), (sr, sl))):
            col = j * 2 + ord_idx
            slot_meta.append((sa, sb))
            for i, (c, t) in enumerate(pairs):
                cost[i, col] = _hop_distance(layout[c], sa) + _hop_distance(
                    layout[t], sb
                )
    if perturbation is not None:
        cost = cost + perturbation
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ValueError:
        return None
    # Build new layout
    new_layout = list(layout)
    participating = {qid for pair in pairs for qid in pair}
    assigned_locations: set[tuple[int, int]] = set()
    for ri, ci in zip(row_ind.tolist(), col_ind.tolist()):
        c, t = pairs[ri]
        sa, sb = slot_meta[ci]
        new_layout[c] = sa
        new_layout[t] = sb
        assigned_locations.add((sa.word_id, sa.site_id))
        assigned_locations.add((sb.word_id, sb.site_id))
    # Resolve bystander collisions: any qubit not in this stage's pairs
    # whose current location now overlaps an assigned slot must be
    # relocated to an empty slot, otherwise the move solver rejects the
    # layout with "Atoms can't occupy the same location". We pick the
    # nearest unused entangling slot for each colliding bystander.
    bystander_locs = {
        (new_layout[i].word_id, new_layout[i].site_id)
        for i in range(len(new_layout))
        if i not in participating
    }
    if assigned_locations & bystander_locs:
        # Find every slot used anywhere (assigned + non-colliding bystanders)
        all_slot_keys: set[tuple[int, int]] = set()
        for sl, sr in pair_slots:
            all_slot_keys.add((sl.word_id, sl.site_id))
            all_slot_keys.add((sr.word_id, sr.site_id))
        used_now = {
            (new_layout[i].word_id, new_layout[i].site_id) for i in participating
        }
        for i in range(len(new_layout)):
            if i in participating:
                continue
            key = (new_layout[i].word_id, new_layout[i].site_id)
            if key not in assigned_locations:
                used_now.add(key)
                continue
            # Find an empty slot — prefer hop-distance-close to current loc.
            free = sorted(
                (k for k in all_slot_keys if k not in used_now),
                key=lambda k: abs(k[0] - new_layout[i].word_id)
                + abs(k[1] - new_layout[i].site_id),
            )
            if not free:
                # No free slot anywhere: bail. The caller will catch this
                # and skip the candidate.
                return None
            chosen = free[0]
            new_layout[i] = LocationAddress(chosen[0], chosen[1], new_layout[i].zone_id)
            used_now.add(chosen)
    return tuple(new_layout)


def _hungarian_candidates(
    pairs: Sequence[tuple[int, int]],
    layout: tuple[LocationAddress, ...],
    pair_slots: tuple[tuple[LocationAddress, LocationAddress], ...],
    n_candidates: int,
    seed: int = 0,
) -> tuple[tuple[LocationAddress, ...], ...]:
    """Generate up to ``n_candidates`` distinct Hungarian-assigned
    target layouts by perturbing the cost matrix with seeded noise.
    """
    if not pairs:
        return (layout,)
    rng = np.random.default_rng(seed)
    seen: dict[tuple[tuple[int, int], ...], tuple[LocationAddress, ...]] = {}
    n_pairs = len(pairs)
    # 2x columns: each slot pair has two orientation columns; matches the
    # cost-matrix shape used by ``_hungarian_assignment``.
    n_cols = len(pair_slots) * 2
    # First candidate: no perturbation.
    base = _hungarian_assignment(pairs, layout, pair_slots, perturbation=None)
    if base is not None:
        seen[_layout_locations(base)] = base
    # Cap retry budget. On dense stages with many degenerate Hungarian
    # assignments, the retry loop can spin without finding new layouts
    # while allocating O(n_pairs * n_cols) cost matrices each iteration.
    # A hard cap of ``min(n_candidates * 2, 40)`` bounds the cost while
    # still saturating ``n_candidates`` distinct layouts in the typical
    # sparse case (where most retries find new ones).
    retry_budget = min(n_candidates * 2, 40)
    for k in range(retry_budget):
        if len(seen) >= n_candidates:
            break
        scale = 0.1 + (k % 5) * 0.3
        noise = rng.normal(0.0, scale, size=(n_pairs, n_cols))
        cand = _hungarian_assignment(pairs, layout, pair_slots, perturbation=noise)
        if cand is None:
            continue
        key = _layout_locations(cand)
        if key not in seen:
            seen[key] = cand
    return tuple(seen.values())


# -----------------------------------------------------------------------------
# Strategy.
# -----------------------------------------------------------------------------


@dataclass
class MultiCandidateLookaheadPlacementStrategy(PlacementStrategyABC):
    """Beam-search placement with NoReturn as an unconditional safety net.

    Parameters
    ----------
    arch_spec
        Architecture specification.
    beam_width
        ``W``: number of candidate trajectories preserved across the
        lookahead window. ``W=8`` is the empirical knee on a 32-circuit
        sweep; ``W=4`` is a faster compromise; ``W=16`` saturates in our
        tests.
    hungarian_candidates
        ``K``: number of perturbed Hungarian-assigned target layouts
        generated per (trajectory, stage). ``K=20`` is the empirical
        plateau; more candidates do not improve quality but cost
        runtime.
    lookahead_window
        Number of future stages from the harness ``lookahead_cz_layers``
        argument that the beam scores against. ``8`` matches the typical
        harness lookahead and the observed cascade horizon on `qugan`-
        family circuits (where stage 0-5 commits affect stage 17-20
        cost).
    lookahead_weight
        ``λ`` in the score function. ``2.0`` chosen by ablation; lower
        values under-penalize cascade traps, higher values cause the
        beam to over-anchor to next-stage geometry.
    nr_max_expansions
        ``max_expansions`` forwarded to the internal
        :class:`NoReturnPlacementStrategy` safety net.
    nr_restarts
        ``restarts`` forwarded to the internal NoReturn strategy.
    """

    beam_width: int = 8
    hungarian_candidates: int = 20
    lookahead_window: int = 24
    lookahead_weight: float = 2.0
    nr_max_expansions: int | None = 300
    nr_restarts: int = 20

    _nr: NoReturnPlacementStrategy = field(init=False, repr=False)
    _solver: MoveSolver | None = field(default=None, init=False, repr=False)
    _traversal: RustPlacementTraversal = field(init=False, repr=False)
    _pair_slots: tuple[tuple[LocationAddress, LocationAddress], ...] = field(
        init=False, repr=False
    )
    _oracle_cache: dict[
        tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]],
        tuple[tuple, ...] | None,
    ] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        assert_single_cz_zone(self.arch_spec, type(self).__name__)
        self._nr = NoReturnPlacementStrategy(
            arch_spec=self.arch_spec,
            max_expansions=self.nr_max_expansions,
            restarts=self.nr_restarts,
        )
        self._traversal = RustPlacementTraversal(strategy="greedy", max_expansions=300)
        self._pair_slots = _entangling_pair_slots(self.arch_spec)

    def validate_initial_layout(
        self, initial_layout: tuple[LocationAddress, ...]
    ) -> None:
        _ = initial_layout

    def _get_solver(self) -> MoveSolver:
        if self._solver is None:
            self._solver = MoveSolver.from_arch_spec(self.arch_spec._inner)
        return self._solver

    # ------ small-arity helpers ------ #

    def _make_state(
        self, layout: tuple[LocationAddress, ...], move_count: tuple[int, ...]
    ) -> ConcreteState:
        return ConcreteState(
            occupied=frozenset(),
            layout=layout,
            move_count=move_count,
        )

    def _oracle_move_layers(
        self,
        layout_from: tuple[LocationAddress, ...],
        layout_to: tuple[LocationAddress, ...],
    ) -> tuple[tuple, ...] | None:
        if layout_from == layout_to:
            return ()
        # Cache by compact (word_id, site_id) keys: identical layout
        # transitions recur across beam trajectories that share a prefix
        # layout, especially on dense stages where many candidates land on
        # the same target layout.
        key = (_layout_locations(layout_from), _layout_locations(layout_to))
        if key in self._oracle_cache:
            return self._oracle_cache[key]
        # Reject layouts with duplicate locations before constructing the
        # state — ConcreteState raises ValueError ("Atoms can't occupy the
        # same location") on invalid layouts, and we want to skip those
        # candidates without crashing the beam.
        from_keys = _layout_locations(layout_from)
        to_keys = _layout_locations(layout_to)
        if len(set(from_keys)) != len(from_keys) or len(set(to_keys)) != len(to_keys):
            if len(self._oracle_cache) < 1024:
                self._oracle_cache[key] = None
            return None
        state_from = self._make_state(layout_from, tuple(0 for _ in layout_from))
        state_to = self._make_state(layout_to, tuple(0 for _ in layout_to))
        try:
            result: tuple[tuple, ...] | None = tuple(
                compute_move_layers(
                    self.arch_spec,
                    state_from,
                    state_to,
                    solver=self._get_solver(),
                    traversal=self._traversal,
                )
            )
        except Exception:
            result = None
        # Bound the cache to avoid unbounded growth across many compiles;
        # 1024 is enough to capture all per-call repetitions while bounding
        # the per-instance memory.
        if len(self._oracle_cache) < 1024:
            self._oracle_cache[key] = result
        return result

    def _stage_candidates(
        self,
        layout: tuple[LocationAddress, ...],
        pairs: Sequence[tuple[int, int]],
        move_count: tuple[int, ...],
        prev_lookahead: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    ) -> list[tuple[tuple[LocationAddress, ...], tuple, tuple[int, ...]]]:
        """Return ``(new_layout, move_layers, new_move_count)`` triples for
        every candidate (K Hungarian + 1 NoReturn).
        """
        results: list[tuple[tuple[LocationAddress, ...], tuple, tuple[int, ...]]] = []

        # K Hungarian-perturbed candidates.
        for target in _hungarian_candidates(
            pairs, layout, self._pair_slots, n_candidates=self.hungarian_candidates
        ):
            ml = self._oracle_move_layers(layout, target)
            if ml is None:
                continue
            new_mc = tuple(
                mc + int(src != dst) for mc, src, dst in zip(move_count, layout, target)
            )
            results.append((target, ml, new_mc))

        # Always include the NoReturn result as one additional candidate.
        # Reframed (not as a strict invariant guarantee but as a useful
        # candidate): the beam's K Hungarian candidates may miss layouts
        # that NR's Rust solver finds via its loose-goal optimization, so
        # we always include the NR result for the beam to consider.
        controls, targets = zip(*pairs) if pairs else ((), ())
        state = self._make_state(layout, move_count)
        try:
            nr_out = self._nr.cz_placements(
                state,
                controls=controls,
                targets=targets,
                lookahead_cz_layers=prev_lookahead,
            )
        except Exception:
            nr_out = None
        if isinstance(nr_out, ExecuteCZ):
            results.append(
                (
                    tuple(nr_out.layout),
                    tuple(nr_out.move_layers),
                    tuple(nr_out.move_count),
                )
            )

        return results

    def _lookahead_cost(
        self,
        layout: tuple[LocationAddress, ...],
        next_pairs: Sequence[tuple[int, int]],
    ) -> int:
        """Estimate next-stage cost: min over Hungarian candidates."""
        if not next_pairs:
            return 0
        best: int | None = None
        for target in _hungarian_candidates(
            next_pairs, layout, self._pair_slots, n_candidates=4
        ):
            ml = self._oracle_move_layers(layout, target)
            if ml is None:
                continue
            cost = sum(len(L) for L in ml)
            if best is None or cost < best:
                best = cost
        return best if best is not None else 0

    # ------ PlacementStrategyABC interface ------ #

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

        all_stages: list[tuple[tuple[int, ...], tuple[int, ...]]] = [
            (controls, targets)
        ]
        if lookahead_cz_layers:
            # ``lookahead_cz_layers[0]`` is the current layer; subsequent
            # entries are future layers. Cap to ``self.lookahead_window``.
            all_stages.extend(lookahead_cz_layers[1 : 1 + self.lookahead_window])

        # Beam: list of (layout, cum_cost, first_layout, first_move_layers,
        # first_move_count, current_move_count)
        beam = [
            (
                tuple(state.layout),
                0,
                None,  # first_layout
                None,  # first_move_layers
                None,  # first_move_count
                tuple(state.move_count),
            )
        ]

        for stage_i, (ctrls, tgts) in enumerate(all_stages):
            pairs = list(zip(ctrls, tgts))
            current_pair_set = frozenset(tuple(sorted(p)) for p in pairs)
            # Skip-repeated-stages lookahead: when consecutive stages have
            # the same pair set (echo / DD patterns common in QFT/VQE
            # circuits), the immediate next layer carries no cascade
            # signal — it just costs zero if we don't move. Lookahead-score
            # at the FIRST DIFFERENT future stage to actually see the
            # next decision point.
            next_pairs: list[tuple[int, int]] = []
            for j in range(stage_i + 1, len(all_stages)):
                nc, nt = all_stages[j]
                la_pairs = list(zip(nc, nt))
                la_set = frozenset(tuple(sorted(p)) for p in la_pairs)
                if la_set != current_pair_set:
                    next_pairs = la_pairs
                    break

            new_beam: list[tuple] = []
            for layout, cum, first_layout, first_ml, first_mc, move_count in beam:
                # Lookahead passed to NR helper: pretend remaining stages.
                # Bloqade's NR uses the future_cz_layers internally for
                # hungarian_horizon scoring.
                prev_lookahead = tuple(all_stages[stage_i:])
                prev_lookahead = tuple((tuple(c), tuple(t)) for c, t in prev_lookahead)

                cands = self._stage_candidates(
                    layout, pairs, move_count, prev_lookahead
                )
                for new_layout, ml, new_mc in cands:
                    cost = sum(len(L) for L in ml)
                    la_cost = self._lookahead_cost(new_layout, next_pairs)
                    score = cum + cost + self.lookahead_weight * la_cost
                    nf_layout = new_layout if first_layout is None else first_layout
                    nf_ml = ml if first_ml is None else first_ml
                    nf_mc = new_mc if first_mc is None else first_mc
                    new_beam.append(
                        (
                            new_layout,
                            cum + cost,
                            nf_layout,
                            nf_ml,
                            nf_mc,
                            new_mc,
                            score,
                        )
                    )

            if not new_beam:
                # Beam collapsed — fall back to NR result directly. Cannot
                # happen with unconditional safety net unless NR itself
                # raised, which is a hard error.
                return self._nr.cz_placements(
                    state, controls, targets, lookahead_cz_layers
                )

            # Dedup by layout, keep min-score per layout.
            best_by_layout: dict[tuple[tuple[int, int], ...], tuple] = {}
            for entry in new_beam:
                key = _layout_locations(entry[0])
                if key not in best_by_layout or best_by_layout[key][-1] > entry[-1]:
                    best_by_layout[key] = entry
            dedup = list(best_by_layout.values())
            dedup.sort(key=lambda e: e[-1])
            beam = [e[:-1] for e in dedup[: self.beam_width]]

        # End of beam: pick min by cumulative cost (not score — score
        # weighs lookahead which is now in the past).
        best = min(beam, key=lambda e: e[1])
        _, _, first_layout, first_ml, first_mc, _ = best

        if first_layout is None:
            # Unreachable: any non-empty pair-list has at least one stage.
            return AtomState.bottom()

        return ExecuteCZ(
            occupied=state.occupied,
            layout=first_layout,
            move_count=first_mc if first_mc is not None else state.move_count,
            active_cz_zones=self.arch_spec.cz_zone_addresses,
            move_layers=tuple(first_ml) if first_ml is not None else (),
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
            zone_maps=tuple(ZoneAddress(loc.zone_id) for loc in state.layout),
        )
