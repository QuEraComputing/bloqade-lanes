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
      - `lib.unresolved_qubits`, `lib.legal_lanes`, `lib.scored_lane`, and
        `lib.pack_aod_rectangles` to build candidate move-sets
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

Below is a copy of `policies/reference/dfs.star` — a depth-first search
that picks `cands[branch_index]` at each node, dives on a successful
insert, and halts with `halt("fallback", ...)` when it exhausts the
candidate budget at a node WITHOUT backtracking.

Empirically this baseline EXPANDS ~5 NODES on the synthetic 2-qubit
physical-arch probe before halting (status=Fallback, move_layers=[],
goal not reached). It is NOT a working solver — the goal of the
autotune loop is for YOU to evolve this into one. Typical first
improvements: add parent-backtracking on dead-end, replace
first-child dive with argmax over a real heuristic, use a per-node
`tried_count` in `update_node_state` to make resumption deterministic.
"""

PARAMS_DEFAULTS = {
    "top_c": 6,
    "max_branch": 12,
    "w_t": 0.0,
    "w_arrived": 100.0,
    "w_progress": 1.0,
}

def _merge_params(defaults, overrides):
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)


def _top_c_per_qubit(scored, c):
    """Policy-owned top-c pruning: score desc, lane id asc, per qubit."""
    if len(scored) == 0 or c <= 0:
        return []

    qids = []
    for s in scored:
        if s.qid not in qids:
            qids = qids + [s.qid]

    out = []
    for qid in qids:
        bucket = []
        for s in scored:
            if s.qid == qid:
                bucket = bucket + [s]

        bucket = stable_sort(bucket, lambda s: s.lane.encoded)
        bucket = stable_sort(bucket, lambda s: s.score, desc=True)

        limit = len(bucket)
        if limit > c:
            limit = c
        for i in range(0, limit):
            out = out + [bucket[i]]

    return out


def _score_candidate(cand, ctx):
    arrived = 0
    for t in ctx.targets:
        qid = t[0]
        target_loc = t[1]
        got = cand.new_config.get(qid)
        if got != None and got == target_loc:
            arrived = arrived + 1
    return PARAMS["w_arrived"] * arrived + PARAMS["w_progress"] * cand.score_sum


def init(root, ctx):
    return {
        "current": root,
        "pending": None,
    }


def step(graph, gs, ctx, lib):
    """Greedy best-first search with parent-pointer backtracking."""

    pending = gs["pending"]
    if pending != None:
        outcome = graph.last_insert()
        if outcome != None and outcome["is_new"] and outcome["error"] == None:
            return update_global_state({"current": outcome["child_id"], "pending": None})
        return update_global_state({"pending": None})

    node = gs["current"]
    if graph.is_goal(node):
        return emit_solution(node)

    cfg = graph.config(node)

    scored = []
    for q in lib.unresolved_qubits(cfg):
        for lane in lib.legal_lanes(cfg, q["qid"]):
            dst = ctx.arch_spec.lane_endpoints(lane)[1]
            dd = (
                lib.blended_distance(q["current"], q["target"], PARAMS["w_t"])
                - lib.blended_distance(dst, q["target"], PARAMS["w_t"])
            )
            scored = scored + [lib.scored_lane(q["qid"], lane, dd)]

    topped = _top_c_per_qubit(scored, PARAMS["top_c"])
    cands = lib.pack_aod_rectangles(topped, cfg, ctx)

    raw_ns = graph.ns(node)
    tried = raw_ns["tried"] if "tried" in raw_ns else []

    fresh = []
    for c in cands:
        enc = c.move_set.encoded
        already = False
        for t_enc in tried:
            if t_enc == enc:
                already = True
                break
        if not already:
            fresh = fresh + [c]

    max_branch = PARAMS["max_branch"]
    capped = []
    cap_limit = len(fresh)
    if cap_limit > max_branch:
        cap_limit = max_branch
    for i in range(0, cap_limit):
        capped = capped + [fresh[i]]
    fresh = capped

    if len(fresh) == 0:
        parent = graph.parent(node)
        if parent == None:
            return halt("fallback", "gbfs: root dead-end")
        return update_global_state({"current": parent, "pending": None})

    best = argmax(fresh, lambda c: _score_candidate(c, ctx))
    return [
        insert_child(node, best.move_set.encoded),
        update_node_state(node, {"tried": tried + [best.move_set.encoded]}),
        update_global_state({"current": node, "pending": node}),
    ]
