"""Per-call parity oracle for Rust ↔ Python entropy internals.

The tests in this module import oracle bindings from the Rust
`bloqade-lanes-bytecode-python` crate compiled with the
`parity_oracle` feature. If those bindings are not available the
whole module is skipped -- CI builds that do not enable the feature
will see these tests as skipped, not failed.

Task 0.5 adds:
- ``test_corpus_is_deterministic`` — verifies build_corpus is stable.
- Parametrized variants of the three binding tests over a real fixture
  corpus drawn from adder_4, ghz_4, steane_physical_35.
- The original inline-stopgap versions renamed to ``*_simple`` suffixes.
"""

from __future__ import annotations

from typing import Any

import pytest

# `_parity_oracle` is a PyO3 submodule attached to the _native extension.
# PyO3 submodules are accessible as attributes, not as importable packages,
# so we cannot use `pytest.importorskip("...._native._parity_oracle")`.
# Instead, import the parent and skip the whole module at collection time
# when the attribute is absent (i.e. feature not compiled).
# `oracle` is typed `Any` because pyright cannot narrow through the
# feature-gated try/except + `pytest.skip(allow_module_level=True)` pattern.
try:
    from bloqade.lanes.bytecode._native import (
        _parity_oracle as _imported_oracle,  # type: ignore[attr-defined]
    )
except ImportError:
    pytest.skip("parity_oracle feature not compiled", allow_module_level=True)

oracle: Any = _imported_oracle

# ── Minimal inline arch JSON (same as Rust test_utils::example_arch_json) ──
#
# Two-word architecture with site buses (src=[0..4] -> dst=[5..9]) and
# word buses (src=[0] -> dst=[1]).  No transport paths, so w_t is irrelevant
# (blended distance degrades to hop-count only when fastest_lane_us is None).
_SIMPLE_ARCH_JSON = r"""
{
    "version": "2.0",
    "words": [
        { "sites": [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0],
                    [0, 1], [1, 1], [2, 1], [3, 1], [4, 1]] },
        { "sites": [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2],
                    [0, 3], [1, 3], [2, 3], [3, 3], [4, 3]] }
    ],
    "zones": [
        {
            "grid": {
                "x_start": 1.0, "y_start": 2.5,
                "x_spacing": [2.0, 2.0, 2.0, 2.0],
                "y_spacing": [2.5, 7.5, 2.5]
            },
            "site_buses": [
                { "src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9] }
            ],
            "word_buses": [
                { "src": [0], "dst": [1] }
            ],
            "words_with_site_buses": [0, 1],
            "sites_with_word_buses": [5, 6, 7, 8, 9],
            "entangling_pairs": [[0, 1]]
        }
    ],
    "zone_buses": [],
    "modes": [
        { "name": "default", "zones": [0], "bitstring_order": [] }
    ]
}
"""

# Target locations: site 5 in word 0 and site 5 in word 1 (the BFS roots).
# Encoded via the Rust LocationAddress: zone_id=0, word_id, site_id.
# We import LocationAddress from the native module to get consistent encoding.
from bloqade.lanes.bytecode._native import (  # noqa: E402
    LocationAddress as _RustLocationAddr,
)


def _enc(zone: int, word: int, site: int) -> int:
    return _RustLocationAddr(zone, word, site).encode()


# word=0 site=5 and word=1 site=5 are both valid BFS roots.
_TARGET_ENCS = [_enc(0, 0, 5), _enc(0, 1, 5)]


# ── Helpers ────────────────────────────────────────────────────────────────


def _loc_py(zone: int, word: int, site: int):
    """Create a Python LocationAddress (word_id, site_id, zone_id ordering)."""
    from bloqade.lanes.layout import LocationAddress

    return LocationAddress(word, site, zone)


def _rust_enc_to_py(enc: int):
    """Decode a Rust-encoded LocationAddress to a Python LocationAddress."""
    from bloqade.lanes.bytecode._native import LocationAddress as _RustAddr
    from bloqade.lanes.layout import LocationAddress

    rust_addr = _RustAddr.decode(enc)
    return LocationAddress(rust_addr.word_id, rust_addr.site_id, rust_addr.zone_id)


def _decode_loc(enc: int):
    """Alias of _rust_enc_to_py for readability in corpus tests."""
    return _rust_enc_to_py(enc)


def _build_tree_from_fixture(fixture):
    """Reconstruct a Python ConfigurationTree from an OracleFixture."""
    from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
    from bloqade.lanes.layout.arch import ArchSpec
    from bloqade.lanes.search.tree import ConfigurationTree

    arch_spec = ArchSpec(_RustArchSpec.from_json(fixture.arch_json))
    old_config = {qid: _decode_loc(enc) for qid, enc in fixture.old_config_encoded}
    blocked = frozenset(_decode_loc(enc) for enc in fixture.blocked_encoded)
    return arch_spec, ConfigurationTree.from_initial_placement(
        arch_spec, old_config, blocked_locations=blocked
    )


# ── Corpus fixture ─────────────────────────────────────────────────────────

# Build corpus once at module load (scope="module" would require a separate
# fixture but module-level constant works fine for parametrize).
#
# Using per_case=10 to keep test time reasonable.  The corpus is deterministic
# so `pytest -v` always shows the same IDs.
import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_benchmarks_root = _Path(__file__).parent.parent.parent / "benchmarks"
if str(_benchmarks_root) not in _sys.path:
    _sys.path.insert(0, str(_benchmarks_root))


def _get_corpus():
    from benchmarks.harness.parity_oracle import build_corpus

    return build_corpus(per_case=10)


# Use module-level lazy init so collection is fast when oracle is missing.
_CORPUS: list | None = None


def _corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = _get_corpus()
    return _CORPUS


# ── Distance-table smoke tests (unchanged) ─────────────────────────────────


def test_distance_table_lookup_returns_dict():
    """Smoke test: oracle returns a dict with int keys and float values."""
    result = oracle.distance_table_lookup(
        arch_json=_SIMPLE_ARCH_JSON,
        targets=_TARGET_ENCS,
        w_t=0.0,
    )
    assert isinstance(result, dict)
    # Should have entries for both targets.
    assert len(result) > 0
    for (src_enc, tgt_enc), value in result.items():
        assert isinstance(src_enc, int)
        assert isinstance(tgt_enc, int)
        assert isinstance(value, float)
        assert value >= 0.0


def test_distance_table_lookup_self_distance_zero():
    """Each target should have distance 0.0 to itself."""
    result = oracle.distance_table_lookup(
        arch_json=_SIMPLE_ARCH_JSON,
        targets=_TARGET_ENCS,
        w_t=0.0,
    )
    for tgt_enc in _TARGET_ENCS:
        key = (tgt_enc, tgt_enc)
        assert key in result, f"Missing self-distance for target {tgt_enc}"
        assert result[key] == pytest.approx(
            0.0, abs=1e-9
        ), f"Self-distance should be 0.0, got {result[key]}"


# ── Simple-fixture (stopgap) variants of the three binding tests ───────────


def test_distance_table_lookup_matches_python_scorer_simple():
    """Rust distance table should match Python CandidateScorer._distance_to_target.

    Uses the inline stopgap two-word arch (w_t=0.0 → hop-count only).
    This is the Phase-0 smoke test; the corpus-parametrized version is
    ``test_distance_table_lookup_matches_python_scorer``.

    NOTE: This test is EXPECTED TO FAIL in Phase 0 because Python uses a
    live Dijkstra pathfinder while Rust uses a precomputed BFS table.
    A mismatch here confirms the oracle is working and reveals the gap.
    """
    from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
    from bloqade.lanes.layout.arch import ArchSpec
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree

    # Build a Python tree using the same arch JSON.
    arch_spec = ArchSpec(_RustArchSpec.from_json(_SIMPLE_ARCH_JSON))
    # Place one atom at site 0 of word 0 — location must exist in the arch.
    placement = {0: _loc_py(0, 0, 0)}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)

    params = SearchParams(w_t=0.0)
    scorer = CandidateScorer(target={}, params=params)

    # Build Rust table.
    rust_table = oracle.distance_table_lookup(
        arch_json=_SIMPLE_ARCH_JSON,
        targets=_TARGET_ENCS,
        w_t=0.0,
    )

    # Check a subset: sources that are valid lane endpoints.
    mismatches: list[str] = []
    for tgt_enc in _TARGET_ENCS:
        tgt_py = _rust_enc_to_py(tgt_enc)
        for src_enc in {k[0] for k in rust_table if k[1] == tgt_enc}:
            src_py = _rust_enc_to_py(src_enc)
            key = (src_enc, tgt_enc)
            rust_val = rust_table[key]
            py_val = scorer._distance_to_target(src_py, tgt_py, tree)
            if abs(rust_val - py_val) > 1e-9:
                mismatches.append(
                    f"  src={src_enc} tgt={tgt_enc}: "
                    f"rust={rust_val:.6f} py={py_val:.6f} diff={abs(rust_val-py_val):.2e}"
                )

    if mismatches:
        pytest.fail(
            "Rust DistanceTable mismatches Python scorer (expected in Phase 0):\n"
            + "\n".join(mismatches[:10])
        )


def test_score_moveset_matches_python_simple():
    """Rust entropy_score_moveset should agree with Python CandidateScorer.score_moveset.

    Uses a single-qubit, single-lane move (site 0 -> site 5, word 0) with
    w_t=0.0 so both Python (Dijkstra) and Rust (BFS) use hop-count distances.
    This is a Phase-0 stopgap; the corpus-parametrized version is
    ``test_score_moveset_matches_python``.

    NOTE: A mismatch here is acceptable in Phase 0 — it reveals an algorithm
    gap (e.g. Python uses Dijkstra live vs Rust precomputed BFS). A match
    confirms the binding is wired correctly.
    """
    from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
    from bloqade.lanes.layout.arch import ArchSpec
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree

    # Arch with one zone, two words, site bus 0->5.
    arch_spec = ArchSpec(_RustArchSpec.from_json(_SIMPLE_ARCH_JSON))

    # Qubit 0 starts at word=0, site=0; target is word=0, site=5.
    src_py = _loc_py(0, 0, 0)  # zone=0, word=0, site=0
    dst_py = _loc_py(0, 0, 5)  # zone=0, word=0, site=5
    target_py = _loc_py(0, 0, 5)

    placement = {0: src_py}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    node = tree.root

    # Find the lane from src -> dst in the tree.
    lane = None
    for candidate_lane in tree.outgoing_lanes(src_py):
        _, endpoint = arch_spec.get_endpoints(candidate_lane)
        if endpoint == dst_py:
            lane = candidate_lane
            break
    assert lane is not None, f"No lane found from {src_py} to {dst_py}"

    moveset = frozenset([lane])
    params = SearchParams(w_t=0.0)
    scorer = CandidateScorer(target={0: target_py}, params=params)
    py_score = scorer.score_moveset(moveset, node, tree)

    # Build Rust inputs: encoded (qubit_id, encoded_location) pairs.
    src_enc = _enc(0, 0, 0)  # zone=0, word=0, site=0
    dst_enc = _enc(0, 0, 5)  # zone=0, word=0, site=5
    tgt_enc = _enc(0, 0, 5)

    old_config = [(0, src_enc)]
    new_config = [(0, dst_enc)]
    targets_rust = [(0, tgt_enc)]
    blocked: list[int] = []

    rust_score = oracle.entropy_score_moveset(
        arch_json=_SIMPLE_ARCH_JSON,
        old_config=old_config,
        new_config=new_config,
        targets=targets_rust,
        blocked=blocked,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        w_t=params.w_t,
    )

    if abs(rust_score - py_score) > 1e-9:
        pytest.fail(
            f"Rust entropy_score_moveset ({rust_score:.9f}) != "
            f"Python CandidateScorer.score_moveset ({py_score:.9f}) "
            f"diff={abs(rust_score - py_score):.2e} "
            f"(mismatch expected in Phase 0 if algorithms differ)"
        )
    else:
        # Scores agree — binding is wired correctly.
        assert rust_score == pytest.approx(py_score, abs=1e-9)


def test_generate_candidates_matches_python_simple():
    """Rust entropy_generate_candidates should return the same ranked list as Python.

    Constructs a minimal one-qubit setup on the stopgap arch, calls both
    Python HeuristicMoveGenerator.generate() and Rust
    oracle.entropy_generate_candidates(), then compares:
      - List length (number of candidates returned).
      - Per-candidate lane sets (as frozensets of encoded u64 lane addresses).
      - Per-candidate resulting configs (as sorted (qid, enc_loc) tuples).
      - Per-candidate alpha/beta/gamma scores via oracle.entropy_score_moveset
        applied to each resulting config (apples-to-apples score comparison).

    NOTE on scores: Rust generate_candidates always returns score=1.0 in its
    output tuple (the actual scoring is used internally for ranking but is NOT
    exposed in the result). To compare scores apples-to-apples, we call
    oracle.entropy_score_moveset on each resulting (old_config, new_config) pair
    from both sides.

    NOTE: A mismatch is acceptable in Phase 0 — it reveals algorithm gaps.
    A match confirms the binding is wired correctly.
    """
    from dataclasses import dataclass

    from bloqade.lanes.bytecode._native import ArchSpec as _RustArchSpec
    from bloqade.lanes.layout.arch import ArchSpec
    from bloqade.lanes.search.generators import HeuristicMoveGenerator
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams
    from bloqade.lanes.search.tree import ConfigurationTree

    @dataclass
    class _EntropyNode:
        entropy: int = 1

    # Use the stopgap arch: one zone, two words, site bus 0→5.
    arch_spec = ArchSpec(_RustArchSpec.from_json(_SIMPLE_ARCH_JSON))

    # Qubit 0 starts at zone=0, word=0, site=0; target is zone=0, word=0, site=5.
    src_py = _loc_py(0, 0, 0)
    tgt_py = _loc_py(0, 0, 5)

    placement = {0: src_py}
    tree = ConfigurationTree.from_initial_placement(arch_spec, placement)
    node = tree.root

    params = SearchParams(w_t=0.0, alpha=80.0, beta=3.0, gamma=3.1, w_d=0.95, w_m=0.8)
    scorer = CandidateScorer(target={0: tgt_py}, params=params)
    search_nodes: dict[int, _EntropyNode] = {id(node): _EntropyNode(entropy=1)}
    gen = HeuristicMoveGenerator(
        scorer=scorer,
        params=params,
        search_nodes=search_nodes,  # type: ignore[arg-type]  # duck-typed EntropyNode stub
    )

    # ── Python side ──────────────────────────────────────────────────────────
    py_movesets = list(gen.generate(node, tree))

    src_enc = _enc(0, 0, 0)
    tgt_enc = _enc(0, 0, 5)

    def _apply_moveset_py(
        moveset: "frozenset",
        node: "object",
        tree: "object",
    ) -> list[tuple[int, int]]:
        """Compute the resulting (qid, enc_loc) pairs after applying moveset."""
        new_cfg: dict[int, object] = dict(node.configuration)  # type: ignore[attr-defined]
        for lane in moveset:
            src, dst = tree.arch_spec.get_endpoints(lane)  # type: ignore[attr-defined]
            qid = node.get_qubit_at(src)  # type: ignore[attr-defined]
            if qid is None:
                continue
            new_cfg[qid] = dst
        # Encode each location to int using the Rust wrapper.
        return sorted(
            (qid, loc._inner.encode())  # type: ignore[attr-defined]
            for qid, loc in new_cfg.items()
        )

    py_items: list[tuple[frozenset[int], list[tuple[int, int]], float]] = []
    for ms in py_movesets:
        lane_set = frozenset(lane._inner.encode() for lane in ms)
        new_cfg_pairs = _apply_moveset_py(ms, node, tree)
        # Score via oracle (same scorer on both sides, apples-to-apples).
        score = oracle.entropy_score_moveset(
            arch_json=_SIMPLE_ARCH_JSON,
            old_config=[(0, src_enc)],
            new_config=new_cfg_pairs,
            targets=[(0, tgt_enc)],
            blocked=[],
            alpha=params.alpha,
            beta=params.beta,
            gamma=params.gamma,
            w_t=params.w_t,
        )
        py_items.append((lane_set, new_cfg_pairs, score))

    # ── Rust side ────────────────────────────────────────────────────────────
    rust_list = oracle.entropy_generate_candidates(
        arch_json=_SIMPLE_ARCH_JSON,
        config=[(0, src_enc)],
        entropy=1,
        alpha=params.alpha,
        beta=params.beta,
        gamma=params.gamma,
        w_d=params.w_d,
        w_m=params.w_m,
        w_t=params.w_t,
        e_max=4,
        max_candidates=4,
        max_movesets_per_group=3,
        targets=[(0, tgt_enc)],
        blocked=[],
        seed=0,
    )

    # For Rust candidates, re-score each resulting config with oracle.entropy_score_moveset.
    rust_items: list[tuple[frozenset[int], list[tuple[int, int]], float]] = []
    for rust_lanes_list, rust_cfg_list, _rust_raw_score in rust_list:
        rust_lanes = frozenset(rust_lanes_list)
        rust_cfg = sorted(rust_cfg_list)
        rust_score = oracle.entropy_score_moveset(
            arch_json=_SIMPLE_ARCH_JSON,
            old_config=[(0, src_enc)],
            new_config=rust_cfg,
            targets=[(0, tgt_enc)],
            blocked=[],
            alpha=params.alpha,
            beta=params.beta,
            gamma=params.gamma,
            w_t=params.w_t,
        )
        rust_items.append((rust_lanes, rust_cfg, rust_score))

    # ── Comparison ───────────────────────────────────────────────────────────
    mismatches: list[str] = []

    if len(rust_items) != len(py_items):
        mismatches.append(f"length: rust={len(rust_items)} py={len(py_items)}")

    for i, (py_item, rust_item) in enumerate(zip(py_items, rust_items)):
        py_lanes, py_cfg, py_score = py_item
        rust_lanes, rust_cfg, rust_score = rust_item

        if rust_lanes != py_lanes:
            mismatches.append(
                f"rank {i} lane mismatch: rust={rust_lanes} py={py_lanes}"
            )
        if rust_cfg != py_cfg:
            mismatches.append(f"rank {i} cfg mismatch: rust={rust_cfg} py={py_cfg}")
        if abs(rust_score - py_score) > 1e-9:
            mismatches.append(
                f"rank {i} score mismatch: "
                f"rust={rust_score:.9f} py={py_score:.9f} "
                f"diff={abs(rust_score - py_score):.2e}"
            )

    if mismatches:
        pytest.fail(
            "Rust entropy_generate_candidates mismatches Python "
            "(acceptable in Phase 0 if algorithms differ):\n" + "\n".join(mismatches)
        )
    else:
        # Both sides agree — binding is wired correctly.
        assert len(rust_items) == len(py_items)
        for i, (py_item, rust_item) in enumerate(zip(py_items, rust_items)):
            py_lanes, py_cfg, py_score = py_item
            rust_lanes, rust_cfg, rust_score = rust_item
            assert rust_lanes == py_lanes, f"rank {i}: lane mismatch"
            assert rust_cfg == py_cfg, f"rank {i}: cfg mismatch"
            assert rust_score == pytest.approx(
                py_score, abs=1e-9
            ), f"rank {i}: score mismatch"


# ── Task 0.5: corpus determinism test ─────────────────────────────────────


def test_corpus_is_deterministic():
    """build_corpus(seed=0) must return the same fixture IDs across two calls."""
    from benchmarks.harness.parity_oracle import build_corpus

    a = build_corpus(seed=0, per_case=5)
    b = build_corpus(seed=0, per_case=5)
    assert [f.id for f in a] == [f.id for f in b]
    # 3 cases × 5 samples; entropy search may terminate early on small cases.
    # Accept between 3 (one per case) and 15 (full sample).
    assert 3 <= len(a) <= 15


# ── Task 0.5: corpus-parametrized binding tests ───────────────────────────


@pytest.fixture(
    params=_corpus(),
    ids=[f.id for f in _corpus()],
    scope="module",
)
def corpus_fixture(request):
    """Parametrize over the real fixture corpus."""
    return request.param


def test_distance_table_lookup_matches_python_scorer(corpus_fixture):
    """Rust distance table should match Python CandidateScorer._distance_to_target.

    Parametrised over the real physical/logical arch corpus.

    EXPECTED to reveal mismatches in Phase 0: Rust uses a precomputed
    BFS table while Python uses live Dijkstra with time-blended weights.
    Failures here are the Phase 0 baseline; Phase 1 will align the algorithms.
    """
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams

    f = corpus_fixture
    arch_spec, tree = _build_tree_from_fixture(f)
    target = {qid: _decode_loc(enc) for qid, enc in f.targets_encoded}
    params = SearchParams(w_t=f.w_t)
    scorer = CandidateScorer(target=target, params=params)

    target_encs = [enc for _, enc in f.targets_encoded]
    rust_table = oracle.distance_table_lookup(
        arch_json=f.arch_json,
        targets=target_encs,
        w_t=f.w_t,
    )

    mismatches: list[str] = []
    for tgt_enc in target_encs:
        tgt_py = _decode_loc(tgt_enc)
        for src_enc in {k[0] for k in rust_table if k[1] == tgt_enc}:
            src_py = _decode_loc(src_enc)
            key = (src_enc, tgt_enc)
            rust_val = rust_table[key]
            py_val = scorer._distance_to_target(src_py, tgt_py, tree)
            if abs(rust_val - py_val) > 1e-9:
                mismatches.append(
                    f"  src={src_enc} tgt={tgt_enc}: "
                    f"rust={rust_val:.6f} py={py_val:.6f} diff={abs(rust_val-py_val):.2e}"
                )

    if mismatches:
        pytest.fail(
            f"[{f.id}] Rust DistanceTable mismatches Python scorer "
            f"(Phase 0 baseline — expected until algorithm is aligned):\n"
            + "\n".join(mismatches[:10])
        )


def test_score_moveset_matches_python(corpus_fixture):
    """Rust entropy_score_moveset should agree with Python CandidateScorer.score_moveset.

    Parametrised over the real fixture corpus.

    The moveset is derived from the (old_config → new_config) config delta:
    for each qubit that moved, find the lane connecting old_loc to new_loc.

    EXPECTED to reveal mismatches in Phase 0: Rust and Python use different
    distance algorithms. Failures here are the Phase 0 baseline.
    """
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams

    f = corpus_fixture
    arch_spec, tree = _build_tree_from_fixture(f)
    target = {qid: _decode_loc(enc) for qid, enc in f.targets_encoded}
    params = SearchParams(
        w_d=f.w_d,
        w_m=f.w_m,
        w_t=f.w_t,
        alpha=f.alpha,
        beta=f.beta,
        gamma=f.gamma,
        max_candidates=f.max_candidates,
        e_max=f.e_max,
    )
    scorer = CandidateScorer(target=target, params=params)

    # Derive moveset from config delta.
    old_config = {qid: _decode_loc(enc) for qid, enc in f.old_config_encoded}
    new_config_map = {qid: _decode_loc(enc) for qid, enc in f.new_config_encoded}
    moveset_lanes = []
    for qid, old_loc in old_config.items():
        new_loc = new_config_map.get(qid)
        if new_loc is not None and new_loc != old_loc:
            lane = arch_spec.get_lane_address(old_loc, new_loc)
            if lane is not None:
                moveset_lanes.append(lane)

    moveset = frozenset(moveset_lanes)
    py_score = scorer.score_moveset(moveset, tree.root, tree)

    rust_score = oracle.entropy_score_moveset(
        arch_json=f.arch_json,
        old_config=f.old_config_encoded,
        new_config=f.new_config_encoded,
        targets=f.targets_encoded,
        blocked=f.blocked_encoded,
        alpha=f.alpha,
        beta=f.beta,
        gamma=f.gamma,
        w_t=f.w_t,
    )

    if abs(rust_score - py_score) > 1e-9:
        pytest.fail(
            f"[{f.id}] Rust entropy_score_moveset ({rust_score:.9f}) != "
            f"Python CandidateScorer.score_moveset ({py_score:.9f}) "
            f"diff={abs(rust_score - py_score):.2e} "
            f"(Phase 0 baseline — expected until algorithm is aligned)"
        )
    else:
        assert rust_score == pytest.approx(py_score, abs=1e-9)


def test_generate_candidates_matches_python(corpus_fixture):
    """Rust entropy_generate_candidates should match Python HeuristicMoveGenerator.generate.

    Parametrised over the real fixture corpus.

    Compares:
      - Number of candidates returned.
      - Per-candidate lane sets (frozensets of encoded lane addresses).
      - Per-candidate resulting configs (sorted (qid, enc_loc) pairs).

    Scores are NOT compared here because Python and Rust may order candidates
    differently; test_score_moveset_matches_python covers the score parity.

    EXPECTED to reveal mismatches in Phase 0. Failures here are the baseline.
    """
    from dataclasses import dataclass

    from bloqade.lanes.search.generators import HeuristicMoveGenerator
    from bloqade.lanes.search.scoring import CandidateScorer
    from bloqade.lanes.search.search_params import SearchParams

    @dataclass
    class _MockEntropyNode:
        entropy: int

    f = corpus_fixture
    arch_spec, tree = _build_tree_from_fixture(f)
    target = {qid: _decode_loc(enc) for qid, enc in f.targets_encoded}
    params = SearchParams(
        w_d=f.w_d,
        w_m=f.w_m,
        w_t=f.w_t,
        alpha=f.alpha,
        beta=f.beta,
        gamma=f.gamma,
        max_candidates=f.max_candidates,
        e_max=f.e_max,
    )
    scorer = CandidateScorer(target=target, params=params)
    search_nodes = {id(tree.root): _MockEntropyNode(entropy=f.entropy)}
    gen = HeuristicMoveGenerator(
        scorer=scorer,
        params=params,
        search_nodes=search_nodes,  # type: ignore[arg-type]  # duck-typed EntropyNode stub
    )

    # ── Python side ──────────────────────────────────────────────────────────
    py_movesets = list(gen.generate(tree.root, tree))

    def _apply_moveset_py(moveset, node, tree) -> list[tuple[int, int]]:
        new_cfg: dict = dict(node.configuration)
        for lane in moveset:
            src, dst = tree.arch_spec.get_endpoints(lane)
            qid = node.get_qubit_at(src)
            if qid is None:
                continue
            new_cfg[qid] = dst
        return sorted(
            (qid, loc._inner.encode())  # type: ignore[attr-defined]
            for qid, loc in new_cfg.items()
        )

    py_items: list[tuple[frozenset[int], list[tuple[int, int]]]] = []
    for ms in py_movesets:
        lane_set = frozenset(lane._inner.encode() for lane in ms)
        new_cfg_pairs = _apply_moveset_py(ms, tree.root, tree)
        py_items.append((lane_set, new_cfg_pairs))

    # ── Rust side ────────────────────────────────────────────────────────────
    rust_list = oracle.entropy_generate_candidates(
        arch_json=f.arch_json,
        config=f.old_config_encoded,
        entropy=f.entropy,
        alpha=f.alpha,
        beta=f.beta,
        gamma=f.gamma,
        w_d=f.w_d,
        w_m=f.w_m,
        w_t=f.w_t,
        e_max=f.e_max,
        max_candidates=f.max_candidates,
        max_movesets_per_group=f.max_movesets_per_group,
        targets=f.targets_encoded,
        blocked=f.blocked_encoded,
        seed=0,
    )
    rust_items: list[tuple[frozenset[int], list[tuple[int, int]]]] = []
    for rust_lanes_list, rust_cfg_list, _score in rust_list:
        rust_lanes = frozenset(rust_lanes_list)
        rust_cfg = sorted(rust_cfg_list)
        rust_items.append((rust_lanes, rust_cfg))

    # ── Comparison ───────────────────────────────────────────────────────────
    mismatches: list[str] = []

    if len(rust_items) != len(py_items):
        mismatches.append(f"length: rust={len(rust_items)} py={len(py_items)}")

    for i, (py_item, rust_item) in enumerate(zip(py_items, rust_items)):
        py_lanes, py_cfg = py_item
        rust_lanes, rust_cfg = rust_item
        if rust_lanes != py_lanes:
            mismatches.append(
                f"rank {i} lane mismatch: rust={rust_lanes} py={py_lanes}"
            )
        if rust_cfg != py_cfg:
            mismatches.append(f"rank {i} cfg mismatch: rust={rust_cfg} py={py_cfg}")

    if mismatches:
        pytest.fail(
            f"[{f.id}] Rust entropy_generate_candidates mismatches Python "
            f"(Phase 0 baseline — expected until algorithm is aligned):\n"
            + "\n".join(mismatches)
        )
    else:
        assert len(rust_items) == len(py_items)
        for i, (py_item, rust_item) in enumerate(zip(py_items, rust_items)):
            py_lanes, py_cfg = py_item
            rust_lanes, rust_cfg = rust_item
            assert rust_lanes == py_lanes, f"[{f.id}] rank {i}: lane mismatch"
            assert rust_cfg == py_cfg, f"[{f.id}] rank {i}: cfg mismatch"
