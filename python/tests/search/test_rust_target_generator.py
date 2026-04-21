"""Tests for the Rust-native target generator plugin system."""

from __future__ import annotations

import json

import pytest

from bloqade.lanes.bytecode._native import (
    DefaultTargetGenerator,
    LocationAddress,
    MoveSolver,
    MultiSolveResult,
)


def loc(zone: int, word: int, site: int) -> LocationAddress:
    return LocationAddress(zone_id=zone, word_id=word, site_id=site)


@pytest.fixture(scope="module")
def arch_json() -> str:
    """Two-word architecture with one CZ entangling pair (word 0 <-> word 1)."""
    return json.dumps(
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


@pytest.fixture(scope="module")
def solver(arch_json: str) -> MoveSolver:
    return MoveSolver(arch_json)


class TestDefaultTargetGenerator:
    def test_repr(self):
        gen = DefaultTargetGenerator()
        assert "DefaultTargetGenerator" in repr(gen)


class TestGenerateCandidates:
    def test_produces_one_candidate(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0), 1: loc(0, 1, 0)}
        candidates = solver.generate_candidates(initial, [0], [1])
        assert len(candidates) == 1
        assert isinstance(candidates[0], dict)
        assert 0 in candidates[0]
        assert isinstance(candidates[0][0], LocationAddress)

    def test_explicit_generator_raises(self, solver: MoveSolver):
        gen = DefaultTargetGenerator()
        initial = {0: loc(0, 0, 0), 1: loc(0, 1, 0)}
        with pytest.raises(ValueError, match="not yet supported"):
            solver.generate_candidates(initial, [0], [1], generator=gen)

    def test_empty_when_target_qubit_missing(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0)}  # qubit 1 missing
        candidates = solver.generate_candidates(initial, [0], [1])
        assert len(candidates) == 0


class TestSolveWithGenerator:
    def test_solves_cz_placement(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0), 1: loc(0, 1, 0)}
        result = solver.solve_with_generator(initial, [], [0], [1], max_expansions=1000)
        assert isinstance(result, MultiSolveResult)
        assert result.status == "solved"
        assert result.candidate_index == 0
        assert result.candidates_tried == 1
        assert result.total_expansions >= 0

    def test_attempts_detail(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0), 1: loc(0, 1, 0)}
        result = solver.solve_with_generator(initial, [], [0], [1], max_expansions=1000)
        assert len(result.attempts) == 1
        attempt = result.attempts[0]
        assert attempt["candidate_index"] == 0
        assert attempt["status"] == "solved"
        assert isinstance(attempt["nodes_expanded"], int)

    def test_unsolvable_when_no_candidates(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0)}  # qubit 1 missing
        result = solver.solve_with_generator(initial, [], [0], [1], max_expansions=1000)
        assert result.status == "unsolvable"
        assert result.candidate_index is None
        assert result.candidates_tried == 0

    def test_move_layers_and_goal_config(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0), 1: loc(0, 1, 0)}
        result = solver.solve_with_generator(initial, [], [0], [1], max_expansions=1000)
        assert isinstance(result.move_layers, list)
        assert isinstance(result.goal_config, list)
        assert result.cost >= 0.0
        assert result.deadlocks >= 0

    def test_repr(self, solver: MoveSolver):
        initial = {0: loc(0, 0, 0), 1: loc(0, 1, 0)}
        result = solver.solve_with_generator(initial, [], [0], [1], max_expansions=1000)
        assert "MultiSolveResult" in repr(result)
