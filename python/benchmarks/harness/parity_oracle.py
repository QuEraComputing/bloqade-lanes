"""Fixture corpus for the entropy parity oracle.

Draws a deterministic sample of (config, entropy, params) triples from
adder_4, ghz_4, and steane_physical_35 trajectories.  The corpus is
regenerated on demand but stable under a fixed ``seed``.

Usage::

    from benchmarks.harness.parity_oracle import build_corpus, OracleFixture

    corpus = build_corpus(seed=0, per_case=10)
    for fixture in corpus:
        print(fixture.id, fixture.entropy)
"""

from __future__ import annotations

import importlib.resources
import sys
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure benchmark package is importable when this module is used from tests
# ---------------------------------------------------------------------------
_benchmarks_root = Path(__file__).parent.parent.parent
if str(_benchmarks_root) not in sys.path:
    sys.path.insert(0, str(_benchmarks_root))


@dataclass(frozen=True)
class OracleFixture:
    """One snapshot drawn from an entropy-search trajectory.

    Captures everything needed to drive both the Rust oracle bindings and
    the Python CandidateScorer / HeuristicMoveGenerator for an
    apples-to-apples comparison.
    """

    id: str
    """Stable label of the form ``<case_id>__step<N>``."""

    arch_json: str
    """Serialised architecture JSON (the string the oracle bindings expect)."""

    old_config_encoded: list[tuple[int, int]]
    """(qubit_id, encoded_location) pairs for the *before* configuration."""

    new_config_encoded: list[tuple[int, int]]
    """(qubit_id, encoded_location) pairs for the *after* configuration."""

    targets_encoded: list[tuple[int, int]]
    """(qubit_id, encoded_location) pairs for the target placement."""

    blocked_encoded: list[int]
    """Sorted list of encoded locations that are externally blocked."""

    entropy: int
    """Entropy value at the parent node when the descent happened."""

    # Search parameters that produced this fixture (defaults from SearchParams)
    w_d: float
    w_m: float
    w_t: float
    alpha: float
    beta: float
    gamma: float
    max_candidates: int
    max_movesets_per_group: int
    e_max: int

    # -----------------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------------

    def targets_dict_encoded(self) -> dict[int, int]:
        """Return targets as {qubit_id: encoded_location}."""
        return dict(self.targets_encoded)

    def old_config_dict_encoded(self) -> dict[int, int]:
        """Return old config as {qubit_id: encoded_location}."""
        return dict(self.old_config_encoded)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_arch_json(case_id: str) -> str:
    """Return the architecture JSON string for a given benchmark case id."""
    # steane_physical_35 uses the physical arch (no logical qubits).
    # All other cases (adder_4, ghz_4, ...) use the logical arch.
    if "physical" in case_id:
        ref = (
            importlib.resources.files("bloqade.lanes.arch.gemini.physical")
            / "_physical_spec.json"
        )
        return ref.read_text(encoding="utf-8")
    else:
        from bloqade.lanes.arch.gemini.logical.spec import _ARCH_JSON

        return _ARCH_JSON


def _get_arch_spec(case_id: str):
    """Return the Python ArchSpec for a given benchmark case id."""
    if "physical" in case_id:
        from bloqade.lanes.arch.gemini import physical

        return physical.get_arch_spec()
    else:
        from bloqade.lanes.arch.gemini import logical

        return logical.get_arch_spec()


def _get_layout_heuristic(case_id: str):
    """Return the appropriate layout heuristic for a given benchmark case id."""
    if "physical" in case_id:
        from bloqade.lanes.heuristics.physical.layout import (
            PhysicalLayoutHeuristicGraphPartitionCenterOut,
        )

        return PhysicalLayoutHeuristicGraphPartitionCenterOut()
    else:
        from bloqade.lanes.heuristics.logical import layout as logical_layout

        return logical_layout.LogicalLayoutHeuristic()


def _encode_config(config: dict) -> list[tuple[int, int]]:
    """Encode a {qubit_id: LocationAddress} dict to sorted (qid, enc) pairs."""
    return sorted((qid, loc._inner.encode()) for qid, loc in config.items())


def _encode_targets(target: dict) -> list[tuple[int, int]]:
    """Encode a target dict to sorted (qid, enc) pairs."""
    return sorted((qid, loc._inner.encode()) for qid, loc in target.items())


def _encode_blocked(blocked: frozenset) -> list[int]:
    """Encode a frozenset of LocationAddress to sorted list of ints."""
    return sorted(loc._inner.encode() for loc in blocked)


def _downsample(items: list, n: int) -> list:
    """Pick at most *n* items evenly spaced through *items*, keeping first/last."""
    if len(items) <= n:
        return list(items)
    if n <= 0:
        return []
    if n == 1:
        return [items[0]]
    step = (len(items) - 1) / (n - 1)
    indices = {round(i * step) for i in range(n)}
    return [items[i] for i in sorted(indices)]


# ---------------------------------------------------------------------------
# Snapshot collector
# ---------------------------------------------------------------------------


def _collect_snapshots_for_case(case_id: str, params) -> list[dict]:
    """Run the Python entropy search for *case_id* and collect descent snapshots.

    Returns a list of dicts with keys:
        old_config, new_config, entropy, target, blocked, params
    """
    from benchmarks.kernels import select_benchmark_cases

    from bloqade.lanes.heuristics.physical.placement import (
        EntropyPlacementTraversal,
        PhysicalPlacementStrategy,
    )
    from bloqade.lanes.search.traversal.step_info import DescendStepInfo
    from bloqade.lanes.upstream import squin_to_move

    cases = select_benchmark_cases({case_id})
    case = cases[0]

    arch_spec = _get_arch_spec(case_id)
    layout_heuristic = _get_layout_heuristic(case_id)

    all_snapshots: list[dict] = []

    # We patch EntropyPlacementTraversal.path_to_target_config locally
    # to inject our on_step recorder.  We restore the original after the
    # search so there are no side-effects.
    original_path_to_target = EntropyPlacementTraversal.path_to_target_config

    cz_index: list[int] = [0]

    def patched_path_to_target(self, *, tree, target):  # type: ignore[no-untyped-def]
        local_snaps: list[dict] = []

        def on_step(event, node, info):  # type: ignore[no-untyped-def]
            if event != "descend":
                return
            if not isinstance(info, DescendStepInfo):
                return
            if node.parent is None:
                return
            local_snaps.append(
                {
                    "old_config": dict(node.parent.configuration),
                    "new_config": dict(node.configuration),
                    "entropy": info.entropy,
                    "target": dict(target),
                    "blocked": frozenset(tree.blocked_locations),
                    "cz_index": cz_index[0],
                }
            )

        patched_self = replace(self, on_search_step=on_step)
        result = original_path_to_target(patched_self, tree=tree, target=target)
        all_snapshots.extend(local_snaps)
        cz_index[0] += 1
        return result

    EntropyPlacementTraversal.path_to_target_config = patched_path_to_target
    try:
        strategy = PhysicalPlacementStrategy(
            arch_spec=arch_spec,
            traversal=EntropyPlacementTraversal(search_params=params),
        )
        squin_to_move(
            case.kernel,
            layout_heuristic=layout_heuristic,
            placement_strategy=strategy,
            insert_return_moves=True,
            logical_initialize=case.logical_initialize,
        )
    finally:
        EntropyPlacementTraversal.path_to_target_config = original_path_to_target

    return all_snapshots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_corpus(
    cases: Iterable[str] = ("adder_4", "ghz_4", "steane_physical_35"),
    per_case: int = 30,
    seed: int = 0,
) -> list[OracleFixture]:
    """Return a deterministic corpus sampled from entropy-search trajectories.

    For each case: run the Python entropy strategy with default
    ``SearchParams``, and at each descent step snapshot
    ``(parent_config, child_config, entropy)``.  Down-sample to at most
    ``per_case`` fixtures per case (evenly spaced through the trajectory).

    Determinism: ``build_corpus(seed=0)`` returns fixtures with identical
    ``id`` lists across runs because:
      1. The entropy search is deterministic under a fixed arch + placement.
      2. The down-sampling uses evenly-spaced indices (no RNG).
      3. The *seed* parameter is accepted for API compatibility and reserved
         for future use (e.g., random tie-breaking in candidate selection).

    Args:
        cases: Iterable of benchmark case IDs.
        per_case: Maximum fixtures to return per case.
        seed: Reserved for future use; does not affect the current
            deterministic trajectory.

    Returns:
        Flat list of OracleFixture objects (case 1, then case 2, ...).
    """
    from bloqade.lanes.search.search_params import SearchParams

    params = SearchParams()
    max_movesets_per_group = 3  # matches default in HeuristicMoveGenerator callers

    corpus: list[OracleFixture] = []
    for case_id in cases:
        arch_json = _get_arch_json(case_id)
        snapshots = _collect_snapshots_for_case(case_id, params)
        selected = _downsample(snapshots, per_case)

        for step_idx, snap in enumerate(selected):
            fixture = OracleFixture(
                id=f"{case_id}__step{step_idx}",
                arch_json=arch_json,
                old_config_encoded=_encode_config(snap["old_config"]),
                new_config_encoded=_encode_config(snap["new_config"]),
                targets_encoded=_encode_targets(snap["target"]),
                blocked_encoded=_encode_blocked(snap["blocked"]),
                entropy=snap["entropy"],
                w_d=params.w_d,
                w_m=params.w_m,
                w_t=params.w_t,
                alpha=params.alpha,
                beta=params.beta,
                gamma=params.gamma,
                max_candidates=params.max_candidates,
                max_movesets_per_group=max_movesets_per_group,
                e_max=params.e_max,
            )
            corpus.append(fixture)

    return corpus
