"""Move Policy DSL - autotune candidate.

Minimal Steane baseline for autotune.

GOAL: Invent and implement a Move Policy SEARCH STRATEGY for
`steane_physical_35` on the physical Gemini arch while keeping
success_rate == 1.0 and reducing total move events. Do not call
`invoke_builtin(...)`; implement search in this Starlark policy using
`graph.*`, action verbs, and `lib` primitives.

This seed is intentionally small: a DFS-style first-candidate baseline.
It exists so the first autotune iteration always starts from a valid,
unopinionated policy file rather than from a previous generated strategy.

IMPORTANT: Generated policy files must be ASCII-only. Do not write Unicode
comments, docstrings, separators, arrows, bullets, box-drawing characters, or
typographic punctuation. Use plain ASCII comments like "# Parameters" instead.
"""

PARAMS_DEFAULTS = {
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


def init(root, ctx):
    return {
        "current": root,
        "pending": None,
        "branch_index": 0,
    }


def step(graph, gs, ctx, lib):
    pending = gs["pending"]
    if pending != None:
        outcome = graph.last_insert()
        if outcome != None and outcome["is_new"] and outcome["error"] == None:
            return update_global_state({
                "current": outcome["child_id"],
                "pending": None,
                "branch_index": 0,
            })
        return update_global_state({"pending": None})

    node = gs["current"]
    if graph.is_goal(node):
        return emit_solution(node)

    cfg = graph.config(node)
    scored = []
    for q in lib.unresolved_qubits(cfg):
        for lane in lib.legal_lanes(cfg, q["qid"]):
            dst = ctx.arch_spec.lane_endpoints(lane)[1]
            score = (
                lib.blended_distance(q["current"], q["target"], PARAMS["w_t"])
                - lib.blended_distance(dst, q["target"], PARAMS["w_t"])
            )
            scored = scored + [lib.scored_lane(q["qid"], lane, score)]

    cands = lib.pack_aod_rectangles(scored, cfg, ctx)

    branch_index = gs["branch_index"]
    limit = len(cands)
    if limit > PARAMS["max_branch"]:
        limit = PARAMS["max_branch"]

    if branch_index >= limit:
        return halt("fallback", "baseline dfs: no remaining candidates")

    best = cands[branch_index]
    return [
        insert_child(node, best.move_set.encoded),
        update_global_state({"pending": node, "branch_index": branch_index + 1}),
    ]
