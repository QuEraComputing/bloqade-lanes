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

PARAMS_DEFAULTS = {
    "top_c": 3,
}

def _merge_params(defaults, overrides):
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)


def _uniform_score(q, lane, ns, ctx_arg):
    """Uniform cost: every lane is equally good. Replace with something smarter."""
    return 0.0


def init(root, ctx):
    """Seed the global state.

    Every key used in `update_global_state` patches MUST appear here, or the
    kernel raises SchemaError.
    """
    return {
        "current": root,
        "pending": None,
    }


def step(graph, gs, ctx, lib):
    """Trivial chain walk: pick first candidate, dive, halt if stuck."""

    pending = gs["pending"]
    if pending != None:
        outcome = graph.last_insert()
        if outcome != None and outcome["is_new"] and outcome["error"] == None:
            child_id = outcome["child_id"]
            return update_global_state({"current": child_id, "pending": None})
        return halt("fallback", "baseline: insert failed or duplicate")

    node = gs["current"]
    if graph.is_goal(node):
        return emit_solution(node)

    cfg = graph.config(node)
    scored = lib.score_lanes(cfg, {}, _uniform_score, ctx)
    topped = lib.top_c_per_qubit(scored, PARAMS["top_c"])
    groups = lib.group_by_triplet(topped)
    cands = lib.pack_aod_rectangles(groups, cfg, ctx)

    if len(cands) == 0:
        return halt("fallback", "baseline: no candidates")

    best = cands[0]
    return [
        insert_child(node, best.move_set.encoded),
        update_global_state({"pending": node}),
    ]
