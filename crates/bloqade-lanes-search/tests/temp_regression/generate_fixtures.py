"""Generate randomized regression fixtures for the bloqade-lanes-search crate.

This script drives `MoveSolver` from the Python bindings to produce a fresh
batch of test cases plus their exact expected outputs. The companion Rust
test (`tests/temp_regression.rs`) replays each case through the same Rust
entry point and asserts bitwise equality.

The suite is meant to be **temporary** — see the sibling `README.md`.

Generation pipeline per fixture
-------------------------------
1.  Pick a random qubit count and place them on the Gemini physical arch's
    zone 0 (the entangling zone).
2.  Pick `k` random CZ pairs from those qubits; the controls/targets lists
    are disjoint subsets.
3.  Ask `MoveSolver.generate_candidates(...)` for a default target
    placement; if the default plugin rejects the layout, the case is
    re-rolled. This is the same generator the production
    `PhysicalPlacementStrategy` uses.
4.  Pick a strategy + options from the configured pool.
5.  Call `MoveSolver.solve(...)` once and capture status / cost /
    nodes_expanded / deadlocks / goal_config / move_layers.

The arch JSON is written once at the top of the directory; fixtures only
encode the per-case inputs and outputs.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from bloqade.lanes.arch.gemini.physical import get_arch_spec
from bloqade.lanes.bytecode._native import (
    DeadlockPolicy,
    EntropyOptions,
    LocationAddress,
    MoveSolver,
    SearchStrategy,
    SolveOptions,
)

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE / "fixtures"
ARCH_PATH = HERE / "arch.json"

# Strategies covered by the default suite. Excludes cascade variants
# (those couple two phases, harder to debug regressions on) and the
# weighted/restart sweeps. The refactor target is the core search
# infrastructure exercised by all of these.
DEFAULT_STRATEGIES: tuple[str, ...] = (
    "entropy",
    "astar",
    "ids",
    "dfs",
    "bfs",
    "greedy",
)

_STRATEGY_LOOKUP = {
    "astar": SearchStrategy.ASTAR,
    "dfs": SearchStrategy.DFS,
    "bfs": SearchStrategy.BFS,
    "greedy": SearchStrategy.GREEDY,
    "ids": SearchStrategy.IDS,
    "entropy": SearchStrategy.ENTROPY,
    "cascade-ids": SearchStrategy.CASCADE_IDS,
    "cascade-dfs": SearchStrategy.CASCADE_DFS,
    "cascade-entropy": SearchStrategy.CASCADE_ENTROPY,
}

_STRATEGY_NAME = {v: k for k, v in _STRATEGY_LOOKUP.items()}
_DEADLOCK_NAME = {
    DeadlockPolicy.SKIP: "skip",
    DeadlockPolicy.MOVE_BLOCKERS: "move_blockers",
    DeadlockPolicy.ALL_MOVES: "all_moves",
}


def _zone_storage_sites(arch_json: dict[str, Any]) -> list[tuple[int, int, int]]:
    """All (zone, word, site) triples in the entangling zone (zone 0)."""
    zone_idx = 0
    zone = arch_json["zones"][zone_idx]
    # The physical Gemini arch has one CZ-capable zone with site buses on
    # every word; words_with_site_buses lists the addressable words.
    words = zone.get("words_with_site_buses") or list(range(len(arch_json["words"])))
    sites: list[tuple[int, int, int]] = []
    for word_id in words:
        num_sites = len(arch_json["words"][word_id]["sites"])
        for site_id in range(num_sites):
            sites.append((zone_idx, word_id, site_id))
    return sites


def _to_dict(addr: LocationAddress) -> dict[str, int]:
    return {"zone_id": addr.zone_id, "word_id": addr.word_id, "site_id": addr.site_id}


def _solve_one(
    solver: MoveSolver,
    rng: random.Random,
    sites: Sequence[tuple[int, int, int]],
    strategy_name: str,
    attempt_idx: int,
) -> dict[str, Any] | None:
    """Roll a random case for `strategy_name`. Returns the fixture dict, or
    `None` if the roll did not yield a usable solver call (no candidate or
    duplicate qubit positions)."""
    num_qubits = rng.choice([2, 2, 4, 4, 6, 8, 10])
    if num_qubits > len(sites):
        return None

    # Place qubits at distinct random sites.
    chosen = rng.sample(list(sites), num_qubits)
    initial = {
        qid: LocationAddress(zone_id=z, word_id=w, site_id=s)
        for qid, (z, w, s) in enumerate(chosen)
    }

    # Pick disjoint controls/targets subsets.
    max_pairs = min(num_qubits // 2, 4)
    num_pairs = rng.randint(1, max_pairs)
    qubits = list(range(num_qubits))
    rng.shuffle(qubits)
    controls = qubits[:num_pairs]
    targets = qubits[num_pairs : 2 * num_pairs]
    assert len(controls) == len(targets)

    candidates = solver.generate_candidates(initial, controls, targets)
    if not candidates:
        return None
    target = candidates[0]

    # Bail if the candidate is degenerate (any qubit missing or duplicate
    # location across qubits — the solver would reject it).
    if set(target.keys()) != set(initial.keys()):
        return None
    seen: set[tuple[int, int, int]] = set()
    for loc in target.values():
        key = (loc.zone_id, loc.word_id, loc.site_id)
        if key in seen:
            return None
        seen.add(key)

    # Blocked locations: a small random sprinkle of unoccupied sites in
    # zone 0. Kept small so most cases stay solvable within the budget.
    occupied_keys = {
        (loc.zone_id, loc.word_id, loc.site_id) for loc in initial.values()
    }
    occupied_keys.update(
        (loc.zone_id, loc.word_id, loc.site_id) for loc in target.values()
    )
    free_sites = [s for s in sites if s not in occupied_keys]
    num_blocked = rng.choice([0, 0, 0, 1, 2])
    num_blocked = min(num_blocked, len(free_sites))
    blocked_sites = rng.sample(free_sites, num_blocked)
    blocked = [
        LocationAddress(zone_id=z, word_id=w, site_id=s) for z, w, s in blocked_sites
    ]

    strategy_enum = _STRATEGY_LOOKUP[strategy_name]
    options = SolveOptions(
        strategy=strategy_enum,
        weight=1.0,
        restarts=1,
        deadlock_policy=DeadlockPolicy.SKIP,
        lookahead=False,
    )
    entropy_options = EntropyOptions(
        max_movesets_per_group=3,
        max_goal_candidates=3,
        w_t=0.05,
        collect_entropy_trace=False,
    )
    max_expansions = 2000

    result = solver.solve(
        initial,
        target,
        blocked,
        max_expansions=max_expansions,
        options=options,
        entropy_options=entropy_options,
    )

    expected_goal = {
        str(qid): loc.encode() for qid, loc in sorted(result.goal_config.items())
    }
    move_layers = [[lane.encode() for lane in layer] for layer in result.move_layers]

    return {
        "name": f"case_{attempt_idx:04d}",
        "strategy": strategy_name,
        "initial": {str(qid): loc.encode() for qid, loc in sorted(initial.items())},
        "target": {str(qid): loc.encode() for qid, loc in sorted(target.items())},
        "blocked": [loc.encode() for loc in blocked],
        "options": {
            "strategy": _STRATEGY_NAME[options.strategy],
            "weight": options.weight,
            "restarts": options.restarts,
            "deadlock_policy": _DEADLOCK_NAME[options.deadlock_policy],
            "lookahead": options.lookahead,
            "top_c": options.top_c,
        },
        "entropy_options": {
            "max_movesets_per_group": entropy_options.max_movesets_per_group,
            "max_goal_candidates": entropy_options.max_goal_candidates,
            "w_t": entropy_options.w_t,
            "collect_entropy_trace": entropy_options.collect_entropy_trace,
        },
        "max_expansions": max_expansions,
        "expected": {
            "status": result.status,
            "cost": result.cost,
            "nodes_expanded": int(result.nodes_expanded),
            "deadlocks": int(result.deadlocks),
            "goal_config": expected_goal,
            "move_layers": move_layers,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num-cases", type=int, default=60)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=tuple(_STRATEGY_LOOKUP),
        default=list(DEFAULT_STRATEGIES),
    )
    parser.add_argument(
        "--max-rolls-per-case",
        type=int,
        default=20,
        help="Max re-rolls when generate_candidates rejects a layout.",
    )
    args = parser.parse_args(argv)

    arch_spec = get_arch_spec()
    arch_json_str = arch_spec.to_json()
    arch_dict = json.loads(arch_json_str)

    # Persist arch.json so the Rust test loads the exact same spec used
    # to generate the expected outputs.
    ARCH_PATH.write_text(json.dumps(arch_dict, indent=2) + "\n", encoding="utf-8")

    solver = MoveSolver(arch_json_str)
    sites = _zone_storage_sites(arch_dict)
    rng = random.Random(args.seed)

    # Wipe the fixtures dir so stale cases from prior runs don't linger.
    if FIXTURES_DIR.exists():
        shutil.rmtree(FIXTURES_DIR)
    FIXTURES_DIR.mkdir(parents=True)

    fixtures_written = 0
    status_counts: dict[str, int] = {}
    case_idx = 0
    while fixtures_written < args.num_cases:
        case_idx += 1
        strategy = args.strategies[fixtures_written % len(args.strategies)]
        fixture: dict[str, Any] | None = None
        for _ in range(args.max_rolls_per_case):
            fixture = _solve_one(solver, rng, sites, strategy, fixtures_written + 1)
            if fixture is not None:
                break
        if fixture is None:
            print(
                f"warning: gave up generating case {case_idx} for {strategy} "
                f"after {args.max_rolls_per_case} rolls",
                file=sys.stderr,
            )
            continue
        path = FIXTURES_DIR / f"{fixture['name']}.json"
        path.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        fixtures_written += 1
        status = fixture["expected"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print(
        f"wrote {fixtures_written} fixtures to {FIXTURES_DIR.relative_to(HERE.parent)} "
        f"({status_counts})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
