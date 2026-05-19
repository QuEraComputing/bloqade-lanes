# Plan C snapshot fixtures

Each subdirectory may contain a `problem.json` and one or more
`expected.<policy>.json` files that the snapshot-fixture test driver
(`crates/bloqade-lanes-search/tests/dsl_snapshot.rs`) consumes.

Comparison is **structural**, not byte-for-byte:

- Move:   `{status, halt_reason, expansions, max_depth}`
- Target: `{ok, num_candidates, first_candidate_size}`

Wall-time and policy/problem paths are excluded.

## Sizes

- `move/small/`   — based on `examples/arch/simple.json`; retained as problem data for ad hoc Move policy checks.
- `move/medium/`  — based on `examples/arch/gemini-logical.json`; retained as problem data.
- `move/large/`   — based on `examples/arch/full.json`; retained as problem data.
- `target/small/` — based on `examples/arch/simple.json`. Exercised by `default_target.star`.

## Regenerating

When a target-generator baseline shifts in a way that legitimately changes a result, regenerate:

```bash
just regenerate-fixtures
git diff policies/fixtures/         # eyeball the diff
git add  policies/fixtures/         # commit only intentional shifts
```

CI does **not** auto-regenerate. A failing snapshot test means *either* a real regression *or* an expected baseline shift — review before regenerating.
