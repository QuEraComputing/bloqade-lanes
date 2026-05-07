# Plan C snapshot fixtures

Each subdirectory contains a `problem.json` and one or more
`expected.<policy>.json` files that the snapshot-fixture test driver
(`crates/bloqade-lanes-search/tests/dsl_snapshot.rs`) consumes.

Comparison is **structural**, not byte-for-byte:

- Move:   `{status, halt_reason, expansions, max_depth}`
- Target: `{ok, num_candidates, first_candidate_size}`

Wall-time and policy/problem paths are excluded.

## Sizes

- `move/small/`   — based on `examples/arch/simple.json`. Exercised by every reference policy (entropy, dfs, bfs, ids).
- `move/medium/`  — based on `examples/arch/gemini-logical.json`. Exercised by `entropy.star` only.
- `move/large/`   — based on `examples/arch/full.json`. Exercised by `entropy.star` only.
- `target/small/` — based on `examples/arch/simple.json`. Exercised by `default_target.star`.

## Regenerating

When a baseline (kernel, `entropy.star`, `default_target.star`) shifts in a way that legitimately changes a result, regenerate (Task 17 will add the recipe):

```bash
just regenerate-fixtures
git diff policies/fixtures/         # eyeball the diff
git add  policies/fixtures/         # commit only intentional shifts
```

CI does **not** auto-regenerate. A failing snapshot test means *either* a real regression *or* an expected baseline shift — review before regenerating.
