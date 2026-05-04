"""
Depth-Bounded DFS reference policy (educational)

Originally specified as Iterative Deepening Search (IDS), but
`actions.reset_to_root()` is not part of the v1 DSL surface, so a
single solve runs DFS with a fixed depth cap. If the cap is reached
without finding the target, the policy halts with `"fallback"`. To
approximate IDS in practice, run `eval-policy` repeatedly with
increasing `policy_params.cap`.

Strategy: DFS identical to `dfs.star`, but each insert_child action
checks whether the newly created child's depth (read from
`graph.depth(child_id)`) equals or exceeds `policy_params.cap`. When
the cap is hit, the policy refuses to dive deeper and instead backtracks
by reverting `current` to the parent of the capped node, or halts when
the root is reached with no remaining candidates.

Demonstrates:
  - Maintaining a depth cap in policy parameters (`PARAMS["cap"]`)
  - Reading the current node's depth via `graph.depth(node)`
  - Reading the parent of a node via `graph.parent(node)`
  - Halt with `"fallback"` when the cap is reached without progress

API surface used:
  graph.root                    — integer root node id
  graph.is_goal(node)           — True when node's config == target
  graph.depth(node)             — depth from root (root = 0)
  graph.parent(node)            — parent node id, or None for root
  graph.config(node)            — StarlarkConfig for a node
  graph.last_insert()           — outcome dict of the previous insert_child
  lib.score_lanes(cfg, ns, fn, ctx)   — enumerate (qubit, lane) pairs with scores
  lib.top_c_per_qubit(scored, c)      — keep top-c scored lanes per qubit
  lib.group_by_triplet(scored)        — group by (move_type, bus_id, direction)
  lib.pack_aod_rectangles(groups, cfg, ctx) — build packed AOD candidates
  insert_child(parent, encoded)       — emit an insert action
  update_global_state(patch)          — mutate the global state bag
  halt(status)                        — terminate the solve loop

IMPORTANT implementation notes (v1 constraints):
  - Same API constraints as dfs.star: verbs are top-level globals,
    `graph.last_insert()` is a method, global state schema is fixed
    by `init()`'s return value.
  - `graph.depth(node)` is a METHOD on PolicyGraph — it takes a node
    id (int) and returns the depth as an int.

This is an EDUCATIONAL policy.
"""

# ── Parameters ───────────────────────────────────────────────────────────────

PARAMS_DEFAULTS = {
    "top_c": 3,
    "max_branch": 4,
    "w_t": 0.0,
    "cap": 5,         # maximum depth to explore before halting with "fallback"
}

def _merge_params(defaults, overrides):
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)

# ── init ─────────────────────────────────────────────────────────────────────

def init(root, ctx):
    """
    Called once before the step loop.

    Global state fields (all must be declared here to establish schema):
      current       — node id being expanded this tick
      pending       — node id whose insert_child result we're waiting for
      branch_index  — how many children of `current` have been inserted
      capped        — whether we hit the depth cap last tick (bool)
    """
    return {
        "current": root,
        "pending": None,
        "branch_index": 0,
        "capped": False,
    }

# ── step ──────────────────────────────────────────────────────────────────────

def step(graph, gs, ctx, lib):
    """
    Depth-bounded DFS step.

    Mirrors dfs.star but checks `graph.depth(child_id) >= cap` after each
    insert. If the new child is at or beyond the cap:
      - Do NOT dive into it (don't update `current`).
      - Instead, advance `branch_index` and try the next sibling.
      - If no siblings remain, backtrack via `graph.parent(current)`.
      - If already at root with no candidates, halt("fallback").

    The cap is a hard limit: nodes at depth >= cap are inserted (counted
    as expansions for the transposition table) but never expanded further.
    """

    cap = PARAMS["cap"]

    # ── 1. Handle outcome of the previous insert_child ────────────────────
    pending = gs["pending"]
    capped = gs["capped"]

    if pending != None:
        outcome = graph.last_insert()

        if outcome != None and outcome["is_new"] and outcome["error"] == None:
            child_id = outcome["child_id"]

            # Check goal immediately.
            if graph.is_goal(child_id):
                return emit_solution(child_id)

            # Check depth cap.
            child_depth = graph.depth(child_id)
            if child_depth >= cap:
                # At cap: do not dive. Mark capped and stay on current node
                # to try the next sibling on the following tick.
                return update_global_state({"pending": None, "capped": True})

            # Below cap: dive into the new child (DFS).
            return update_global_state({
                "current": child_id,
                "pending": None,
                "branch_index": 0,
                "capped": False,
            })

        # Duplicate or error: skip, advance branch_index implicitly
        # (it was already incremented when we issued insert_child).
        return update_global_state({"pending": None, "capped": False})

    # ── 2. Check goal ─────────────────────────────────────────────────────
    node = gs["current"]
    if graph.is_goal(node):
        return emit_solution(node)

    # ── 3. Build candidate move sets via pipeline ─────────────────────────
    cfg = graph.config(node)

    def _score_fn(q, lane, ns, ctx_arg):
        dst = ctx_arg.arch_spec.lane_endpoints(lane)[1]
        dd = (
            lib.blended_distance(q["current"], q["target"], PARAMS["w_t"])
            - lib.blended_distance(dst, q["target"], PARAMS["w_t"])
        )
        return dd

    scored = lib.score_lanes(cfg, {}, _score_fn, ctx)
    topped = lib.top_c_per_qubit(scored, PARAMS["top_c"])
    groups = lib.group_by_triplet(topped)
    cands = lib.pack_aod_rectangles(groups, cfg, ctx)

    max_branch = PARAMS["max_branch"]
    limit = len(cands)
    if limit > max_branch:
        limit = max_branch

    # ── 4. Pick the next untried candidate ───────────────────────────────
    branch_index = gs["branch_index"]

    if branch_index >= limit:
        # All branches tried at this node: backtrack to parent.
        parent = graph.parent(node)
        if parent == None:
            # At root with no candidates left: depth-bounded DFS exhausted.
            return halt("fallback", "ids: depth cap reached, no solution found")
        return update_global_state({
            "current": parent,
            "pending": None,
            "branch_index": 0,
            "capped": False,
        })

    best = cands[branch_index]
    return [
        insert_child(node, best.move_set.encoded),
        update_global_state({
            "current": node,
            "pending": node,
            "branch_index": branch_index + 1,
            "capped": False,
        }),
    ]
