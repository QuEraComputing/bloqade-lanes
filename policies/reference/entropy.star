"""Entropy reference policy — Starlark reproduction of Strategy::Entropy.

Source: spec §9 template.

No-RNG deterministic regime (spec §8):
  All choices are made by deterministic scoring (argmax over scored
  candidates).  No random sampling is performed.  The policy owns top-c
  pruning and tie-breaking in Starlark before calling the Rust-backed
  pack_aod_rectangles helper.

Adaptations from the spec §9 template:
  1. `struct(...)` / `record(...)` / `field(...)` are NOT available in the
     starlark-0.13 standard globals used by this kernel.  All "records" are
     plain dicts.
  2. `PARAMS` is a dict, not a `struct(...)`.
  3. `NodeState` is a plain dict with two keys: "entropy" and "tried".
  4. `gs` (global state) is a plain dict: {"current": int, "pending_parent": int|None}.
  5. Bare action verbs (`halt`, `insert_child`, etc.) are used directly;
     no `actions.X` namespace prefix is needed.
  6. `lib.walk_up(graph, node, n)` is NOT implemented by the kernel
     (deferred).  A local `walk_up` helper replicates it via parent-pointer
     traversal.
  7. `q["current"]` / `q["target"]` are used instead of `q.current_loc` /
     `q.target_loc`; the kernel passes the qubit as a dict
     `{"qid": int, "current": Location, "target": Location}`.
  8. `graph.ns(node)` returns a dict; all field accesses use `["key"]`.
  9. `graph.last_insert()` returns a dict `{"child_id", "is_new", "error"}`
     or None; dict indexing is used throughout.
  10. `graph.config(node)` returns a real StarlarkConfig.  The policy uses
      `lib.unresolved_qubits` and `lib.legal_lanes` to enumerate legal lanes,
      builds ScoredLane values itself, and calls `lib.pack_aod_rectangles`
      only after deciding which lanes to keep.
  11. `c.move_set.encoded` — attribute access works on `StarlarkMoveSet`
      (returns a list[int]); `c.score_sum` works on `StarlarkPackedCandidate`.
      Both are accessed via attribute notation as the kernel wraps them.
  12. `outcome["child_id"]` is None or an int; compare with `== None`.
  13. starlark-0.13 resolves free variable names at module eval time, not at
      call time.  Therefore `lib`, `graph`, and `ctx` — which are bound on
      the per-solve Module AFTER the policy module is loaded — CANNOT be
      referenced as bare names inside helper functions.  Instead:
        - `lib` is passed explicitly as a parameter to `score_lane` and
          `score_moveset`.
        - `graph` and `ctx` are accessed only inside `step()` where they are
          already in scope as positional arguments.
"""

# ── Parameters ──────────────────────────────────────────────────────────────

# Default parameter values.  Callers may supply per-solve overrides via the
# `policy_params` argument to `solve_with_policy`; the kernel binds those as
# the `PARAMS_OVERRIDE` global before this module is evaluated, so the merged
# dict below picks them up at load time.
PARAMS_DEFAULTS = {
    "w_d": 1.0,
    "w_m": 1.0,
    "alpha": 1.0,
    "beta": 1.0,
    "gamma": 1.0,
    "e_max": 8,
    "delta_e": 1,
    "reversion_steps": 2,
    "top_c": 3,
    "max_movesets_per_group": 3,
    "w_t": 0.0,
}

# Merge PARAMS_OVERRIDE (bound by the kernel; always a dict, possibly empty)
# into PARAMS_DEFAULTS.  The result is stored as `PARAMS` so all existing
# references throughout this file continue to work without modification.
def _merge_params(defaults, overrides):
    """Return a new dict: defaults with each key in overrides replaced."""
    merged = {}
    for k, v in defaults.items():
        merged[k] = v
    for k, v in overrides.items():
        merged[k] = v
    return merged

PARAMS = _merge_params(PARAMS_DEFAULTS, PARAMS_OVERRIDE)

# ── Helpers ──────────────────────────────────────────────────────────────────

def walk_up(g, node, n):
    """Walk n parent-pointer hops up from node; stop at root. Returns cursor."""
    cursor = node
    for _i in range(n):
        p = g.parent(cursor)
        if p == None:
            break
        cursor = p
    return cursor

def _is_starlark_config(cfg):
    """
    Heuristic: graph.config(node) returns a plain dict {len, node_id} in v1.
    A StarlarkConfig would NOT have a "node_id" key.
    Returns True if cfg looks like a real StarlarkConfig (attribute-based).
    In Starlark, type(x) returns a string.
    """
    return type(cfg) == "Config"

# ── Lane scorer (called from policy-owned lane enumeration) ───────────────────

def score_lane(q, lane, ns, ctx, lib_h):
    """
    Score a single (qubit, lane) pair.

    q     — dict {"qid": int, "current": Location, "target": Location}
    lane  — Lane value
    ns    — per-node state dict {"entropy": int, "tried": list}
    ctx   — Ctx value
    lib_h — LibMove handle (passed explicitly; see adaptation note 13)

    Higher is better.  The entropy term downweights the distance component
    and upweights the mobility component as entropy rises, encouraging
    exploration of less-greedy moves.
    """
    e = ns["entropy"]
    if e < 1:
        e = 1

    dst = ctx.arch_spec.lane_endpoints(lane)[1]

    # Distance delta: positive means the lane brings the qubit closer to its target.
    dd = (
        lib_h.blended_distance(q["current"], q["target"], PARAMS["w_t"])
        - lib_h.blended_distance(dst, q["target"], PARAMS["w_t"])
    )

    # Mobility delta: positive means the destination offers better future reach.
    dm = (
        lib_h.mobility(dst, ctx.targets)
        - lib_h.mobility(q["current"], ctx.targets)
    )

    return (PARAMS["w_d"] / e) * dd + PARAMS["w_m"] * e * dm

# ── Moveset scorer (called on each packed candidate) ─────────────────────────

def score_moveset(c, ns, ctx, lib_h):
    """
    Score a packed candidate.

    c     — StarlarkPackedCandidate: attributes move_set, new_config, score_sum
    ns    — per-node state dict
    ctx   — Ctx value
    lib_h — LibMove handle (passed explicitly)
    """
    delta_d = c.score_sum

    # Count qubits that have arrived at their target in the new config.
    # ctx.targets is a list of (qid, target_loc) 2-tuples.
    arrived = 0
    for t in ctx.targets:
        qid = t[0]
        target_loc = t[1]
        got = c.new_config.get(qid)
        if got != None and got == target_loc:
            arrived = arrived + 1

    # Mobility of each target qubit in the new config.
    delta_m = 0.0
    for t in ctx.targets:
        qid = t[0]
        got = c.new_config.get(qid)
        if got != None:
            delta_m = delta_m + lib_h.mobility(got, ctx.targets)

    return PARAMS["alpha"] * delta_d + PARAMS["beta"] * arrived + PARAMS["gamma"] * delta_m

def top_c_per_qubit(scored, c):
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

# ── init ─────────────────────────────────────────────────────────────────────

def init(root, ctx):
    """
    Called once before the step loop.  Returns the initial global state dict.
    """
    return {"current": root, "pending_parent": None}

# ── step ─────────────────────────────────────────────────────────────────────

def step(g, gs, ctx, lib_h):
    """
    Main policy step.  Called each kernel tick.

    g     — PolicyGraph
    gs    — global state dict {"current": int, "pending_parent": int|None}
    ctx   — Ctx (arch_spec, targets, blocked)
    lib_h — LibMove (distance, legal-lane, and AOD packing helpers)

    The kernel calls:  step(graph, gs, ctx, lib)
    with the per-solve bindings passed positionally.
    """

    # ── 1. Handle outcome of the previous insert_child ────────────────────
    outcome = g.last_insert()
    pending = gs["pending_parent"]

    if outcome != None and pending != None:
        # An insert was attempted last tick.
        if outcome["error"] != None or not outcome["is_new"]:
            # Insert failed or produced a duplicate: bump entropy on the
            # pending parent and clear pending.
            raw_ns = g.ns(pending)
            cur_entropy = raw_ns["entropy"] if "entropy" in raw_ns else 1
            return [
                update_node_state(pending, {"entropy": cur_entropy + PARAMS["delta_e"]}),
                update_global_state({"pending_parent": None}),
            ]
        # Insert succeeded: advance current to the new child.
        child_id = outcome["child_id"]
        return update_global_state({"current": child_id, "pending_parent": None})

    # ── 2. Check goal ─────────────────────────────────────────────────────
    node = gs["current"]
    if g.is_goal(node):
        return emit_solution(node)

    # ── 3. Read per-node state ────────────────────────────────────────────
    raw_ns = g.ns(node)
    entropy = raw_ns["entropy"] if "entropy" in raw_ns else 1
    tried = raw_ns["tried"] if "tried" in raw_ns else []
    ns = {"entropy": entropy, "tried": tried}

    # ── 4. Entropy ceiling: revert or fallback ────────────────────────────
    if ns["entropy"] >= PARAMS["e_max"]:
        anc = walk_up(g, node, PARAMS["reversion_steps"])
        if anc == None:
            anc = g.root

        root_ns = g.ns(g.root)
        root_entropy = root_ns["entropy"] if "entropy" in root_ns else 1

        if anc == g.root and root_entropy >= PARAMS["e_max"]:
            # Global entropy ceiling hit: fall back to sequential solver.
            return [
                invoke_builtin("sequential_fallback", {"from_config": g.config(g.root)}),
                halt("fallback"),
            ]

        anc_ns = g.ns(anc)
        anc_entropy = anc_ns["entropy"] if "entropy" in anc_ns else 1
        return [
            update_node_state(anc, {"entropy": anc_entropy + PARAMS["delta_e"]}),
            update_global_state({"current": anc}),
        ]

    # ── 5. Candidate construction ─────────────────────────────────────────
    cfg = g.config(node)

    # Defensive check for older kernels: query helpers require a real Config.
    if not _is_starlark_config(cfg):
        return update_node_state(node, {"entropy": ns["entropy"] + PARAMS["delta_e"]})

    scored = []
    for q in lib_h.unresolved_qubits(cfg):
        for lane in lib_h.legal_lanes(cfg, q["qid"]):
            score = score_lane(q, lane, ns, ctx, lib_h)
            scored = scored + [lib_h.scored_lane(q["qid"], lane, score)]

    topped = top_c_per_qubit(scored, PARAMS["top_c"])
    cands = lib_h.pack_aod_rectangles(topped, cfg, ctx)

    # Filter out candidates whose encoded move set has already been tried.
    fresh = []
    for c in cands:
        enc = c.move_set.encoded
        already = False
        for t in ns["tried"]:
            if t == enc:
                already = True
                break
        if not already:
            fresh = fresh + [c]

    if len(fresh) == 0:
        # All candidates exhausted: bump entropy to trigger reversion later.
        return update_node_state(node, {"entropy": ns["entropy"] + PARAMS["delta_e"]})

    # Pick the best candidate by moveset score.
    best = argmax(fresh, lambda c: score_moveset(c, ns, ctx, lib_h))

    return [
        insert_child(node, best.move_set.encoded),
        update_node_state(node, {"tried": ns["tried"] + [best.move_set.encoded]}),
        update_global_state({"pending_parent": node}),
    ]
