"""
Breadth-First Search reference policy (educational)

Strategy: visit nodes in insertion order via an explicit FIFO queue
stored in policy global state. Each step pops the head of the queue,
expands all its children, and enqueues the new child ids.

Demonstrates:
  - Mutable policy state via `update_global_state(patch)`
  - Reading and updating a list across steps (FIFO queue)
  - Using `graph.is_goal(node)` and `graph.children_of(node)`
  - Contrast with `dfs.star`: BFS does NOT recurse on `graph.last_insert()`
    but instead tracks a level-order frontier in global state

API surface used:
  graph.root                    — integer root node id
  graph.is_goal(node)           — True when node's config == target
  graph.children_of(node)       — insertion-order child ids of a node
  graph.config(node)            — StarlarkConfig for a node
  lib.score_lanes(cfg, ns, fn, ctx)   — enumerate (qubit, lane) pairs with scores
  lib.top_c_per_qubit(scored, c)      — keep top-c scored lanes per qubit
  lib.group_by_triplet(scored)        — group by (move_type, bus_id, direction)
  lib.pack_aod_rectangles(groups, cfg, ctx) — build packed AOD candidates
  insert_child(parent, encoded)       — emit an insert action
  update_global_state(patch)          — mutate the global state bag
  halt(status)                        — terminate the solve loop

KNOWN LIMITATION: the v1 kernel applies `insert_child` actions atomically
within a tick and reflects them in `graph.last_insert()` on the *next* tick.
Child node ids from a single tick's batch of inserts are not all available
immediately. This policy works around the limitation by tracking the *parent*
node ids in the queue rather than individual child ids: each step pops one
parent, inserts all its children (up to `max_branch`), then advances. On the
next step the newly created children are retrieved via `graph.children_of(parent)`
and enqueued.

This means the queue holds *parent* ids whose children have already been
inserted, waiting to be expanded. One "expand" step per parent is required to
flush their children into the queue. True BFS ordering (child-level FIFO) is
approximated: all nodes at depth D are inserted before any at depth D+1, but
siblings within a depth level are expanded in queue pop order.

This is an EDUCATIONAL policy. It will spend its budget enumerating
shallow nodes; expect budget-exhaust on anything but the smallest
fixtures.
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

# ── init ─────────────────────────────────────────────────────────────────────

def init(root, ctx):
    """
    Seed the BFS queue with the root node.

    Global state fields (all must be declared here to establish schema):
      queue         — list of node ids to expand next (FIFO, head = index 0)
      phase         — "insert" (inserting children of queue[0]) or
                      "advance" (enqueue children of last-expanded parent)
      expanding     — node id currently being expanded (or None)
      branch_index  — how many children of `expanding` have been inserted
    """
    return {
        "queue": [root],
        "phase": "insert",
        "expanding": root,
        "branch_index": 0,
    }

# ── step ──────────────────────────────────────────────────────────────────────

def step(graph, gs, ctx, lib):
    """
    BFS step.

    Two-phase per parent node:

    Phase "insert": insert the next child of `expanding`, increment branch_index.
      When all children (up to max_branch) are inserted, switch to phase "advance".

    Phase "advance": enqueue the children that were just inserted into `expanding`
      (retrieved via graph.children_of), then pop the next parent from the queue
      and switch back to phase "insert".

    Goal check: after each child insert, check if the newly created child is at
      the goal (via graph.is_goal on the child from graph.last_insert()).
    """

    phase = gs["phase"]
    queue = gs["queue"]
    expanding = gs["expanding"]
    branch_index = gs["branch_index"]

    # ── Phase: advance ────────────────────────────────────────────────────
    if phase == "advance":
        # Enqueue children of the node we just finished expanding.
        new_children = graph.children_of(expanding)
        new_queue = queue + new_children

        # Pop the head of the queue as the next parent to expand.
        if len(new_queue) == 0:
            return halt("fallback", "bfs: queue exhausted")

        next_parent = new_queue[0]
        remaining_queue = []
        for i in range(1, len(new_queue)):
            remaining_queue = remaining_queue + [new_queue[i]]

        return update_global_state({
            "queue": remaining_queue,
            "phase": "insert",
            "expanding": next_parent,
            "branch_index": 0,
        })

    # ── Phase: insert ─────────────────────────────────────────────────────

    # Check goal before expanding (catches root-is-goal).
    if graph.is_goal(expanding):
        return emit_solution(expanding)

    # Check goal on last inserted child (from previous tick's insert_child).
    last = graph.last_insert()
    if last != None and last["child_id"] != None and last["error"] == None:
        child_id = last["child_id"]
        if graph.is_goal(child_id):
            return emit_solution(child_id)

    # Build candidate move sets via pipeline.
    cfg = graph.config(expanding)

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

    if branch_index >= limit:
        # Finished inserting all children of `expanding`: advance.
        return update_global_state({
            "queue": queue,
            "phase": "advance",
            "expanding": expanding,
            "branch_index": branch_index,
        })

    best = cands[branch_index]
    return [
        insert_child(expanding, best.move_set.encoded),
        update_global_state({
            "queue": queue,
            "phase": "insert",
            "expanding": expanding,
            "branch_index": branch_index + 1,
        }),
    ]
