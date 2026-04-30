"""Smoke test: policies/reference/entropy.star loads without syntax errors.

Verifies that the file parses and executes its module-level code cleanly
against the kernel's full Starlark global set (stdlib + utilities + actions).
A runtime error arising from an empty arch config is acceptable — the point
is that "syntax_error" does not appear in policy_status.
"""

from __future__ import annotations

import json
from pathlib import Path

from bloqade.lanes.bytecode._native import MoveSolver

# ── Repo root ────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
ENTROPY_STAR = REPO_ROOT / "policies" / "reference" / "entropy.star"

# ── Architecture JSON (two-word arch, same as test_move_policy_dsl.py) ────────

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


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_entropy_star_file_exists():
    """The policy file is present on disk."""
    assert ENTROPY_STAR.exists(), f"entropy.star not found at {ENTROPY_STAR}"


def test_entropy_star_parses():
    """entropy.star loads without a syntax error.

    The policy runs against an empty initial/target config.  The kernel will
    call init() and then step() once; step() detects a placeholder config and
    returns an update_node_state action that bumps entropy.  Because
    max_expansions=1, the kernel terminates with BudgetExhausted before
    completing a second tick.

    The critical assertion is that policy_status does NOT contain
    "syntax_error" — which would indicate the Starlark parser rejected the
    file — or "bad_policy" (missing init/step).
    """
    solver = MoveSolver(ARCH_JSON)
    result = solver.solve(
        initial={},
        target={},
        blocked=[],
        policy_path=str(ENTROPY_STAR),
        policy_params={},
        max_expansions=1,
        timeout_s=10.0,
    )
    assert result.policy_status is not None, "policy_status should be set on DSL solves"
    assert (
        "syntax_error" not in result.policy_status
    ), f"entropy.star failed to parse: {result.policy_status}"
    assert (
        "bad_policy" not in result.policy_status
    ), f"entropy.star missing init/step: {result.policy_status}"
