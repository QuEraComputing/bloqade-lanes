"""Move Policy DSL — autotune candidate.

This file is the subject of the `autotune` loop. The research agent proposes
mutations and the implementation agent edits this file in a worktree.
Aggregated metrics from `python -m benchmarks.cli --strategies dsl_autotune`
are scored against the trivial baseline below; iterations that reduce
`total_events` (and don't regress `success_rate`) are kept.

================================================================================
                         INSTRUCTIONS FOR THE LLM
================================================================================

GOAL: Invent a Move Policy that minimises total move EVENTS (parallel move
timesteps — fewer timesteps = fewer architecture wait/setup cycles) across
the full 9-kernel Squin benchmark suite (`ghz_4`, `ghz_6`, `adder_4`,
`steane_logical_5`, `qpe_9`, `adder_64`, `bv_70`, `steane_physical_35`,
`trotter_rand_35`) on the physical Gemini arch while keeping
`success_rate == 1.0`. Aim to do so with fewer than 1000 node expansions
per CZ stage. Beating `rust_entropy_5` (total_events=3332 on this suite)
is the headline goal.

RULES:
  1. You may read `policies/reference/*.star` ONLY to learn the DSL surface
     (function signatures, action verbs, `graph.*`/`lib.*` accessors). Do
     NOT copy or paraphrase their search strategy — `entropy.star`, `dfs.star`,
     `bfs.star`, and `ids.star` are EXAMPLES OF SYNTAX, not solutions to mimic.
     The point of this loop is to invent a *novel* policy, not to rewrite
     `entropy.star` with different variable names.

  2. You may freely call `invoke_builtin("sequential_fallback")` if you decide
     it's the right escape hatch — but be aware its underlying algorithm
     (greedy per-qubit BFS) is weak on the physical Gemini arch.

  3. Keep this file self-contained. `load("...", ...)` is rejected by the
     sandbox.

  4. Stay deterministic. No randomness — use `lib.stable_sort` / `lib.argmax`
     for any tie-breaking.

DSL surface available as Starlark globals (see `policies/reference/dfs.star`
docstring for full annotations):
  Verbs:      insert_child, update_node_state, update_global_state,
              emit_solution, halt, invoke_builtin
  Utilities:  stable_sort, argmax, normalize
  Handles:    graph (read-only SearchGraph), lib (helpers), ctx (arch/targets)

================================================================================
                              BASELINE POLICY
================================================================================

What follows is a deliberately trivial "uniform-cost / first-child chain
walk" baseline. It picks the first candidate from
`lib.pack_aod_rectangles(...)` at every step with no scoring, no backtracking,
no lookahead. It will solve only the simplest problems. Improve it.
"""

PARAMS_DEFAULTS = {}

def _merge_params(defaults, overrides):
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)


def init(root, ctx):
    # Single-key schema: every key used in `update_global_state` patches
    # must appear here or the kernel raises SchemaError. We track only
    # whether the fallback has been fired so step() is idempotent.
    return {"fired": False}


def step(graph, gs, ctx, lib):
    # First-keep policy: defer routing to the kernel's built-in
    # `sequential_fallback` (greedy per-qubit BFS). The Python bridge
    # (`_is_acceptable_solve` in
    # python/bloqade/lanes/heuristics/physical/movement.py) accepts a
    # DSL result iff either status == "solved", or policy_status starts
    # with "fallback:" AND move_layers is non-empty. The kernel only
    # populates move_layers when sequential_fallback is actually invoked,
    # so we MUST emit invoke_builtin BEFORE halt.
    if not gs["fired"]:
        return [
            invoke_builtin("sequential_fallback", {"from_config": graph.config(graph.root)}),
            update_global_state({"fired": True}),
            halt("fallback", "first-keep: delegate to sequential_fallback"),
        ]
    return halt("fallback", "first-keep: re-entry")
