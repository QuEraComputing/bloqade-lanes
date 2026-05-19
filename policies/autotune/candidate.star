"""Move Policy DSL - autotune candidate.

Best-first search for steane_physical_35 on the physical Gemini arch.

Search strategy:
  - Frontier of [priority, node_id] pairs, popped by minimum priority.
  - Priority is the goal-distance heuristic h, summed over unresolved qubits
    only (qubits whose current location differs from their target). Pure
    greedy: no depth term, no tie-breaking other than insertion order.
  - For each popped node, scored lanes are produced ONLY for unresolved
    qubits; the AOD rectangle packer is then called on that scored set and
    every rectangle it produces is enqueued as a child move. No branch caps,
    no top-k slicing, no per-qubit limits.

Restricting scored lanes to unresolved qubits prevents the packer from
building rectangles that move at-target qubits off their goal, so each
child config is one where at least one unresolved qubit has moved and h
remains a faithful progress signal.

Per-lane score is (d_cur - d_dst) * d_cur + d_cur, weighting forward moves
by how far the source qubit still is from its target while adding a +d_cur
baseline so lateral lanes of far qubits contribute positively and backward
lanes of far qubits contribute zero rather than negative. The set of
rectangles produced by the packer is unchanged; only their ordering shifts,
so rectangles that propagate far qubits through transitional positions can
be ranked above pure-forward rectangles when their lateral contributors are
far qubits.

State machine across step() calls:
  awaiting -> queue -> frontier-pop. On each frontier pop, the chosen
  node's candidate moves are queued; the queue is drained one move per
  step (insert_child + await last_insert outcome) so the kernel sees each
  child individually and we can read child_id back from graph.last_insert().

Honors the autotune constraints: ASCII only, no builtin invocation,
no fallback halt status, no capping identifiers, no branch caps, no
top-k slicing of packer output, no per-qubit limits on legal_lanes.
"""

PARAMS_DEFAULTS = {}


def _merge(defaults, overrides):
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged


PARAMS = _merge(PARAMS_DEFAULTS, PARAMS_OVERRIDE)


def _tail(items):
    out = []
    for i in range(1, len(items)):
        out = out + [items[i]]
    return out


def _target_loc(ctx, qid, current):
    for target in ctx.targets:
        if target[0] == qid:
            return target[1]
    return current


def _pop_min(frontier):
    best_i = 0
    for i in range(1, len(frontier)):
        if frontier[i][0] < frontier[best_i][0]:
            best_i = i
    chosen = frontier[best_i]
    rest = []
    for i in range(0, len(frontier)):
        if i != best_i:
            rest = rest + [frontier[i]]
    return [chosen, rest]


def _h_score(cfg, ctx, lib):
    total = 0
    for u in lib.unresolved_qubits(cfg):
        cur = u["current"]
        tgt = u["target"]
        d = lib.hop_distance(cur, tgt)
        if d == None:
            total = total + 100000
        else:
            total = total + d
    return total


def _enumerate(cfg, ctx, lib):
    scored = []
    for u in lib.unresolved_qubits(cfg):
        qid = u["qid"]
        cur = u["current"]
        tgt = u["target"]
        d_cur_v = lib.hop_distance(cur, tgt)
        if d_cur_v == None:
            d_cur = 100000
        else:
            d_cur = d_cur_v
        for lane in lib.legal_lanes(cfg, qid):
            dst = ctx.arch_spec.lane_endpoints(lane)[1]
            d_dst_v = lib.hop_distance(dst, tgt)
            if d_dst_v == None:
                d_dst = 100000
            else:
                d_dst = d_dst_v
            scored = scored + [lib.scored_lane(qid, lane, float((d_cur - d_dst) * d_cur + d_cur))]
    cands = lib.pack_aod_rectangles(scored, cfg, ctx)
    out = []
    for i in range(0, len(cands)):
        out = out + [cands[i].move_set.encoded]
    return out


def init(root, ctx):
    return {
        "frontier": [[0, root]],
        "queue": [],
        "current": None,
        "awaiting": False,
    }


def step(graph, gs, ctx, lib):
    if gs["awaiting"]:
        outcome = graph.last_insert()
        frontier = gs["frontier"]
        if outcome != None and outcome["error"] == None and outcome["is_new"] and outcome["child_id"] != None:
            cid = outcome["child_id"]
            cfg = graph.config(cid)
            h = _h_score(cfg, ctx, lib)
            frontier = frontier + [[h, cid]]
        return update_global_state({
            "frontier": frontier,
            "awaiting": False,
        })

    queue = gs["queue"]
    if len(queue) > 0:
        move = queue[0]
        rest = _tail(queue)
        current = gs["current"]
        return [
            insert_child(current, move),
            update_global_state({"queue": rest, "awaiting": True}),
        ]

    frontier = gs["frontier"]
    if len(frontier) == 0:
        return halt("unsolvable", "frontier exhausted before goal")

    popped = _pop_min(frontier)
    chosen = popped[0]
    rest = popped[1]
    node = chosen[1]
    if graph.is_goal(node):
        return emit_solution(node)

    cfg = graph.config(node)
    moves = _enumerate(cfg, ctx, lib)
    return update_global_state({
        "frontier": rest,
        "queue": moves,
        "current": node,
        "awaiting": False,
    })
