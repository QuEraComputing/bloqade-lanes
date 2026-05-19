"""Move Policy DSL - autotune candidate.

Minimal Steane baseline for autotune.

GOAL: Invent and implement a Move Policy SEARCH STRATEGY for
`steane_physical_35` on the physical Gemini arch while keeping
success_rate == 1.0 and reducing total move events.

Hard constraints for generated policies:
  - Do not call `invoke_builtin(...)`.
  - Do not call `halt("fallback", ...)`.
  - Do not copy, rename, or lightly edit existing reference strategies.
  - Do not add tiny branch caps that let the policy give up before the
    kernel expansion budget is reached.
  - Keep generated policy files ASCII-only. Do not write Unicode comments,
    separators, arrows, bullets, box-drawing characters, or typographic
    punctuation. Use plain ASCII comments like "# Parameters" instead.

This seed is intentionally plain. It is a graph enumerator, not a tuned search
strategy: it keeps an explicit stack of nodes, considers legal moves for every
qubit in the current config, and relies on the kernel's expansion budget or
timeout to stop unsuccessful searches. It exists only to provide a valid
starting point that does not teach fallback behavior or old heuristics.
"""

PARAMS_DEFAULTS = {
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


def _tail(items):
    out = []
    for i in range(1, len(items)):
        out = out + [items[i]]
    return out


def _prepend(item, items):
    return [item] + items


def _target_for(ctx, qid, current):
    for target in ctx.targets:
        if target[0] == qid:
            return target[1]
    return current


def init(root, ctx):
    return {
        "stack": [root],
        "pending": None,
    }


def step(graph, gs, ctx, lib):
    pending = gs["pending"]
    if pending != None:
        outcome = graph.last_insert()
        stack = gs["stack"]
        if outcome != None and outcome["is_new"] and outcome["error"] == None:
            return update_global_state({
                "stack": _prepend(outcome["child_id"], stack),
                "pending": None,
            })
        return update_global_state({"pending": None})

    stack = gs["stack"]
    if len(stack) == 0:
        return halt("unsolvable", "seed enumerator exhausted the graph frontier")

    node = stack[0]
    rest = _tail(stack)
    if graph.is_goal(node):
        return emit_solution(node)

    ns = graph.ns(node)
    if "ranked" in ns:
        ranked = ns["ranked"]
        idx = ns["idx"]
    else:
        cfg = graph.config(node)
        scored = []
        for item in cfg.iter():
            qid = item[0]
            current = item[1]
            target = _target_for(ctx, qid, current)
            for lane in lib.legal_lanes(cfg, qid):
                dst = ctx.arch_spec.lane_endpoints(lane)[1]
                score = (
                    lib.blended_distance(current, target, PARAMS["w_t"])
                    - lib.blended_distance(dst, target, PARAMS["w_t"])
                )
                scored = scored + [lib.scored_lane(qid, lane, score)]

        cands = lib.pack_aod_rectangles(scored, cfg, ctx)
        ranked = []
        for i in range(0, len(cands)):
            ranked = ranked + [cands[i].move_set.encoded]
        idx = 0

    if idx >= len(ranked):
        return update_global_state({"stack": rest})

    chosen = ranked[idx]
    return [
        insert_child(node, chosen),
        update_node_state(node, {"ranked": ranked, "idx": idx + 1}),
        update_global_state({"pending": node, "stack": _prepend(node, rest)}),
    ]
