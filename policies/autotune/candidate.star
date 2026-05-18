"""Move Policy DSL — autotune candidate.

This file is the subject of the `autotune` loop. The research agent proposes
mutations and the implementation agent edits this file in a worktree.
Aggregated metrics from `python -m benchmarks.cli --strategies dsl_autotune`
are scored against the trivial baseline below; iterations that reduce
`total_events` (and don't regress `success_rate`) are kept.

================================================================================
                         INSTRUCTIONS FOR THE LLM
================================================================================

GOAL: Invent and implement a Move Policy SEARCH STRATEGY that produces a
valid move-set for `ghz_4` on the physical Gemini arch (success_rate == 1.0).
Once a working policy is in, subsequent iterations try to beat the prior
iteration's `total_events`. The point of this loop is to have the LLM
*invent and implement* a search strategy in the DSL — not to call a builtin.

RULES:
  1. You may read `policies/reference/*.star` ONLY to learn the DSL surface
     (function signatures, action verbs, `graph.*`/`lib.*` accessors). Do
     NOT copy or paraphrase their search strategy — `entropy.star`, `dfs.star`,
     `bfs.star`, and `ids.star` are EXAMPLES OF SYNTAX, not solutions to mimic.
     The point of this loop is to invent a *novel* policy, not to rewrite
     `entropy.star` with different variable names.

  2. **DO NOT use `invoke_builtin("sequential_fallback")` or any other
     `invoke_builtin(...)` call.** The whole point is for YOU to implement
     the search. Delegating to a Rust builtin defeats the goal of the loop.
     Your policy must terminate via `emit_solution(node)` (preferred when you
     find a goal-state node) or `halt("solved", ...)` (when you've reached
     `graph.is_goal(node)` and want to short-circuit). `halt("fallback", ...)`
     and `halt("unsolvable", ...)` are valid terminal states for the kernel
     to record, but they will NOT produce a valid move-set — the Python
     bridge only accepts results where the policy emitted real moves via
     `insert_child(...)` and converged to a goal node.

  3. Keep this file self-contained. `load("...", ...)` is rejected by the
     sandbox.

  4. Stay deterministic. No randomness — use `lib.stable_sort` / `lib.argmax`
     for any tie-breaking.

  5. You implement the search loop yourself using:
       - `insert_child(parent, move_set_encoded)` to commit a move
       - `graph.last_insert()` to read the outcome of the previous insert
       - `graph.is_goal(node)`, `graph.parent(node)`, `graph.children_of(node)`
         to navigate
       - `graph.config(node)` to get the StarlarkConfig for a node
       - `lib.score_lanes / top_c_per_qubit / group_by_triplet /
          pack_aod_rectangles` to build candidate move-sets
       - `lib.blended_distance(loc_a, loc_b, w_t)` for a distance heuristic
     Common strategies you could invent: DFS, BFS, greedy best-first,
     A*-with-heuristic, beam search, iterative deepening, or something
     novel. Mix and match. The kernel's transposition table dedupes
     duplicate-state inserts automatically.

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
    """Trivial chain walk: pick first candidate, dive, halt if stuck.

    NOT a real search strategy — your job is to replace this with one.
    See the INSTRUCTIONS block above.
    """

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
