"""
Depth-First Search reference policy (educational)

Strategy: always recurse on the most recently inserted child, via
`graph.last_insert()`. The kernel's transposition table dedupes
duplicate states automatically.

Demonstrates:
  - Reading `graph.last_insert()` between steps
  - Using `graph.is_goal(node)` to check if a node is the target
  - `insert_child(parent, move_set_encoded)` for tree expansion
  - `update_global_state(patch)` to track the current frontier node
  - Halting via `halt("solved")` and `halt("fallback")`

API surface used:
  graph.root                    — integer root node id
  graph.is_goal(node)           — True when node's config == target
  graph.children_of(node)       — insertion-order child ids
  graph.depth(node)             — depth from root (root = 0)
  graph.last_insert()           — outcome dict of the previous insert_child
  graph.config(node)            — StarlarkConfig for a node
  lib.score_lanes(cfg, ns, fn, ctx)   — enumerate (qubit, lane) pairs with scores
  lib.top_c_per_qubit(scored, c)      — keep top-c scored lanes per qubit
  lib.group_by_triplet(scored)        — group by (move_type, bus_id, direction)
  lib.pack_aod_rectangles(groups, cfg, ctx) — build packed AOD candidates
  insert_child(parent, encoded)       — emit an insert action
  update_global_state(patch)          — mutate the global state bag
  halt(status)                        — terminate the solve loop

IMPORTANT implementation notes (v1 constraints):
  1. Action verbs (`halt`, `insert_child`, `update_global_state`, etc.) are
     top-level globals, NOT under an `actions.` namespace.
  2. `graph.last_insert()` is a METHOD, not an attribute. It returns a dict
     {"child_id": int|None, "is_new": bool, "error": str|None} or None.
  3. The global state schema is determined by `init()`'s return value.
     All keys used in `update_global_state` patches MUST appear in the
     dict returned by `init()` — new keys cause a SchemaError.
  4. `lib.score_lanes` / `lib.pack_aod_rectangles` require a real
     StarlarkConfig (returned by `graph.config(node)`). We use the full
     candidate pipeline to enumerate AOD-compatible move sets.
  5. `starlark-0.13` resolves free variables at module-eval time, so
     `lib`, `graph`, and `ctx` cannot be referenced as bare names inside
     module-level helper functions. They are passed explicitly.

This is an EDUCATIONAL policy. It is not tuned for efficiency on
medium/large problems; budget exhaustion is expected for non-trivial
inputs.
"""

# ── Parameters ───────────────────────────────────────────────────────────────

PARAMS_DEFAULTS = {
    "top_c": 3,
    "max_branch": 4,
    "w_t": 0.0,
}

def _merge_params(defaults, overrides):
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)

# ── Score function (passed to lib.score_lanes) ───────────────────────────────

def _score_fn(q, lane, ns, ctx):
    """Simple distance-reducing score: positive means lane moves qubit closer."""
    # lib and ctx are passed in via the step() closure below.
    # This function is defined at module level to avoid re-definition overhead,
    # but lib_h is not available here — the bound version is created in step().
    return 0.0

# ── init ─────────────────────────────────────────────────────────────────────

def init(root, ctx):
    """
    Called once before the step loop.

    Returns a global-state dict that seeds the schema. Every key used in
    any subsequent `update_global_state` call must appear here.

    Fields:
      current       — node id being expanded this tick
      pending       — node id whose insert_child result we're waiting for
                      (None means no insert is pending)
      branch_index  — how many children of `current` have been inserted so far
    """
    return {
        "current": root,
        "pending": None,
        "branch_index": 0,
    }

# ── step ──────────────────────────────────────────────────────────────────────

def step(graph, gs, ctx, lib):
    """
    DFS step: expand the current node by inserting one child per tick.

    On each tick:
      1. If we just inserted a child (pending != None), read last_insert().
         - If the child is new and valid: dive into it (update current).
         - If duplicate or error: try the next branch candidate.
      2. Check goal.
      3. Enumerate candidates via the full pipeline, pick the next untried
         branch (by branch_index), insert it, and record pending.
      4. If no candidates remain: halt("fallback").
    """

    # ── 1. Handle outcome of the previous insert_child ────────────────────
    pending = gs["pending"]
    if pending != None:
        outcome = graph.last_insert()
        if outcome != None and outcome["is_new"] and outcome["error"] == None:
            # Successfully inserted a new child: dive into it (DFS).
            child_id = outcome["child_id"]
            return [
                update_global_state({"current": child_id, "pending": None, "branch_index": 0}),
            ]
        # Duplicate or error: skip this candidate; stay on current node
        # and let branch_index advance on the next tick.
        return update_global_state({"pending": None})

    # ── 2. Check goal ─────────────────────────────────────────────────────
    node = gs["current"]
    if graph.is_goal(node):
        return emit_solution(node)

    # ── 3. Build candidate move sets via pipeline ─────────────────────────
    cfg = graph.config(node)

    def _score_lane_bound(q, lane, ns, ctx_arg):
        # Simple hop-distance-reducing score: blended_distance drop is positive.
        dst = ctx_arg.arch_spec.lane_endpoints(lane)[1]
        dd = (
            lib.blended_distance(q["current"], q["target"], PARAMS["w_t"])
            - lib.blended_distance(dst, q["target"], PARAMS["w_t"])
        )
        return dd

    scored = lib.score_lanes(cfg, {}, _score_lane_bound, ctx)
    topped = lib.top_c_per_qubit(scored, PARAMS["top_c"])
    groups = lib.group_by_triplet(topped)
    cands = lib.pack_aod_rectangles(groups, cfg, ctx)

    # ── 4. Pick the next untried candidate by branch_index ────────────────
    branch_index = gs["branch_index"]
    max_branch = PARAMS["max_branch"]

    # Clamp to the available candidates and the branch cap.
    limit = len(cands)
    if limit > max_branch:
        limit = max_branch

    if branch_index >= limit:
        # All branches exhausted at this node; halt gracefully.
        return halt("fallback", "dfs: no remaining candidates")

    best = cands[branch_index]
    return [
        insert_child(node, best.move_set.encoded),
        update_global_state({"pending": node, "branch_index": branch_index + 1}),
    ]
