"""End-to-end smoke test: MoveSolver.solve(policy_path=...) routes through
the Move Policy DSL kernel and returns a SolveResult with DSL fields."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from bloqade.lanes.bytecode._native import LocationAddress, MoveSolver, SolveResult

# ── Policy fixtures ──────────────────────────────────────────────────────────

HALT_ONLY_POLICY = """\
def init(root, ctx):
    return None

def step(graph, gs, ctx, lib):
    return halt("solved", "trivial")
"""

HALT_UNSOLVABLE_POLICY = """\
def init(root, ctx):
    return None

def step(graph, gs, ctx, lib):
    return halt("unsolvable", "no solution")
"""

# ── Architecture fixture ─────────────────────────────────────────────────────

# Two-word arch JSON identical to the one used in test_rust_target_generator.py.
ARCH_JSON = json.dumps(
    {
        "version": "2.0",
        "words": [
            {
                "sites": [
                    [0, 0],
                    [1, 0],
                    [2, 0],
                    [3, 0],
                    [4, 0],
                    [0, 1],
                    [1, 1],
                    [2, 1],
                    [3, 1],
                    [4, 1],
                ]
            },
            {
                "sites": [
                    [0, 2],
                    [1, 2],
                    [2, 2],
                    [3, 2],
                    [4, 2],
                    [0, 3],
                    [1, 3],
                    [2, 3],
                    [3, 3],
                    [4, 3],
                ]
            },
        ],
        "zones": [
            {
                "grid": {
                    "x_start": 1.0,
                    "y_start": 2.5,
                    "x_spacing": [2.0, 2.0, 2.0, 2.0],
                    "y_spacing": [2.5, 7.5, 2.5],
                },
                "site_buses": [{"src": [0, 1, 2, 3, 4], "dst": [5, 6, 7, 8, 9]}],
                "word_buses": [{"src": [0], "dst": [1]}],
                "words_with_site_buses": [0, 1],
                "sites_with_word_buses": [5, 6, 7, 8, 9],
                "entangling_pairs": [[0, 1]],
            }
        ],
        "zone_buses": [],
        "modes": [{"name": "default", "zones": [0], "bitstring_order": []}],
    }
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def write_policy(src: str) -> str:
    """Write a policy string to a temp file, return the path (caller deletes)."""
    f = tempfile.NamedTemporaryFile("w", suffix=".star", delete=False)
    f.write(src)
    f.flush()
    f.close()
    return f.name


def loc(zone: int, word: int, site: int) -> LocationAddress:
    return LocationAddress(zone_id=zone, word_id=word, site_id=site)


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestSolveWithPolicyPath:
    def test_returns_solve_result(self):
        """solve(policy_path=...) returns a SolveResult instance."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
            )
            assert isinstance(result, SolveResult)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_policy_file_echoed(self):
        """policy_file matches the path supplied."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
            )
            assert result.policy_file == path
        finally:
            Path(path).unlink(missing_ok=True)

    def test_policy_status_solved(self):
        """Halt-only policy returns policy_status == 'solved'."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
            )
            assert result.policy_status == "solved"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_policy_status_unsolvable(self):
        """Policy that halts unsolvable returns policy_status == 'unsolvable'."""
        path = write_policy(HALT_UNSOLVABLE_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
            )
            assert result.policy_status == "unsolvable"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_policy_params_echoed(self):
        """policy_params dict is JSON-encoded and echoed back."""
        path = write_policy(HALT_ONLY_POLICY)
        params = {"e_max": 8, "label": "smoke"}
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
                policy_params=params,
            )
            assert result.policy_params is not None
            recovered = json.loads(result.policy_params)
            assert recovered == params
        finally:
            Path(path).unlink(missing_ok=True)

    def test_policy_params_none_when_not_dsl(self):
        """Fields are None for strategy-based solves."""
        solver = MoveSolver(ARCH_JSON)
        initial = {0: loc(0, 0, 0)}
        target = {0: loc(0, 0, 0)}
        result = solver.solve(initial=initial, target=target, blocked=[])
        assert result.policy_file is None
        assert result.policy_params is None
        assert result.policy_status is None

    def test_timeout_kwarg_accepted(self):
        """timeout_s kwarg is accepted without error."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
                timeout_s=10.0,
            )
            assert result.policy_status == "solved"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_max_expansions_kwarg_accepted(self):
        """max_expansions kwarg is passed through to the DSL kernel."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
                max_expansions=500,
            )
            assert result.policy_status == "solved"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_missing_policy_file_raises(self):
        """Non-existent policy_path raises ValueError."""
        solver = MoveSolver(ARCH_JSON)
        with pytest.raises(ValueError):
            solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path="/nonexistent/path/policy.star",
            )

    def test_empty_policy_params_when_omitted(self):
        """When policy_params is omitted, the echo is an empty JSON object."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
            )
            assert result.policy_params is not None
            assert json.loads(result.policy_params) == {}
        finally:
            Path(path).unlink(missing_ok=True)

    def test_move_layers_empty_for_trivial_halt(self):
        """Trivial halt policy produces no move layers."""
        path = write_policy(HALT_ONLY_POLICY)
        try:
            solver = MoveSolver(ARCH_JSON)
            result = solver.solve(
                initial={},
                target={},
                blocked=[],
                policy_path=path,
            )
            assert result.move_layers == []
        finally:
            Path(path).unlink(missing_ok=True)
