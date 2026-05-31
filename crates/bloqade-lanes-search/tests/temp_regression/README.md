# temp_regression — search-crate behavior lock-in

Temporary regression suite used as a "source of truth" during the
`bloqade-lanes-search` refactor. Locks in current solver behavior at the
integration level so internal restructuring can be verified to be
behavior-preserving.

**Delete this directory when the refactor is complete.** The fixtures
encode current heuristic choices (move ordering, node expansion counts,
goal-config selection) which we expect to change as the heuristics are
re-tuned. They are not a long-term test suite.

## Layout

```
temp_regression/
├── README.md             — this file
├── generate_fixtures.py  — regenerates arch.json + fixtures/*.json
├── arch.json             — frozen Gemini physical ArchSpec used by all cases
└── fixtures/             — per-case JSON: inputs + expected MoveSolver outputs
```

## Regenerating

The generator drives `MoveSolver.solve` via the Python bindings, so the
Python package must be built and installed first:

```bash
just develop-python
uv run python crates/bloqade-lanes-search/tests/temp_regression/generate_fixtures.py
```

Common flags:

```bash
# bigger or smaller suite (default 60 cases)
uv run python .../generate_fixtures.py --num-cases 120

# different RNG seed
uv run python .../generate_fixtures.py --seed 7

# restrict to a subset of strategies
uv run python .../generate_fixtures.py --strategies entropy astar
```

The generator always wipes `fixtures/` before writing, so each run produces
a fresh suite.

## Running

```bash
cargo test -p bloqade-lanes-search --test temp_regression
```

Each fixture is replayed through `MoveSolver::solve` with the captured
options and asserted for exact equality on:

- `status` (`solved` / `unsolvable` / `budget_exceeded`)
- `cost` (bitwise float equality)
- `nodes_expanded`
- `deadlocks`
- `goal_config` (qubit → location)
- `move_layers` (encoded lane sequence per step, in solver-native order)

A single failing fixture aborts the test with a diff of the offending field.

## Fixture schema

Each `fixtures/case_NNNN.json` looks like:

```jsonc
{
  "name": "case_0001",
  "seed": 42,
  "initial":  { "0": <u64>, "1": <u64>, ... },   // encoded LocationAddr
  "target":   { "0": <u64>, ... },
  "blocked":  [ <u64>, ... ],
  "options": {
    "strategy": "entropy",                       // see SearchStrategy variants
    "weight": 1.0,
    "restarts": 1,
    "deadlock_policy": "skip",
    "lookahead": false,
    "top_c": null
  },
  "entropy_options": {
    "max_movesets_per_group": 3,
    "max_goal_candidates": 3,
    "w_t": 0.05,
    "collect_entropy_trace": false
  },
  "max_expansions": 2000,
  "expected": {
    "status": "solved",
    "cost": 3.0,
    "nodes_expanded": 5,
    "deadlocks": 0,
    "goal_config": { "0": <u64>, ... },
    "move_layers": [ [ <u64 LaneAddr>, ... ], ... ]
  }
}
```

`LocationAddr` and `LaneAddr` are stored as their `encode()` / `encode_u64()`
forms — same packing used by the on-the-wire bytecode representation.
