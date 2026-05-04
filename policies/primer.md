<!-- AUTOGEN: DO NOT EDIT BY HAND.
     Regenerate with `just generate-primer`. -->

# Move Policy & Target Generator DSL — Primer

<!-- BEGIN PROSE: intro -->
The Move Policy and Target Generator DSLs let you write Starlark policies that drive
bloqade-lanes' move synthesis and CZ-stage placement. Policies are loaded from `.star`
files, run inside a sandboxed Starlark evaluator, and operate on the same kernel data
that the Rust-native heuristics see. A Move policy controls how the search tree is
expanded step-by-step; a Target Generator policy controls which qubit placements are
offered as CZ-gate candidates at each stage.

From Python, pass `policy_path=` to `MoveSolver.solve(...)` or supply a custom
`PhysicalPlacementStrategy(target_generator=...)` to route placement through a `.star`
file. From the command line, use the `bloqade-bytecode` CLI: `eval-policy --policy
<file>.star --problem <fixture>.json` runs a policy end-to-end and prints a result
summary; `trace-policy --policy <file>.star --problem <fixture>.json` runs with a
verbose observer and emits one record per kernel event, useful for debugging step logic.
Both subcommands accept `--json` for structured output and `--params <path>` to override
`PARAMS_DEFAULTS` at load time.

For the full specification see
`docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md`. Working reference
policies live in `policies/reference/`. Problem fixture JSON files used by the snapshot
integration tests are in `policies/fixtures/{move,target}/`.
<!-- END PROSE: intro -->

## Move Policy surface

<!-- BEGIN AUTOGEN: actions -->
### `actions.* — kernel-driven verbs`

#### `insert_child`

Emit an `insert_child` action — create a child of `parent` with the given encoded lanes.

```rust
fn insert_child < 'v > (parent : i32 , move_set_encoded : & 'v ListRef < 'v > , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```

#### `update_node_state`

Emit an `update_node_state` action, merging `patch` dict into per-node DSL state for `node`.

```rust
fn update_node_state < 'v > (node : i32 , patch : Value < 'v > , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```

#### `update_global_state`

Emit an `update_global_state` action, merging `patch` into the solver-wide global state.

```rust
fn update_global_state < 'v > (patch : Value < 'v > , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```

#### `emit_solution`

Emit an `emit_solution` action, recording `node` as a solved output.

```rust
fn emit_solution < 'v > (node : i32 , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```

#### `halt`

Halt the search with `status` ∈ {"solved","unsolvable","fallback","error"}.

```rust
fn halt < 'v > (status : & str , # [starlark (default = "")] message : & str , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```

#### `invoke_builtin`

Invoke built-in `name` with `args` dict; result readable via `graph.last_builtin_result()`.

```rust
fn invoke_builtin < 'v > (name : & str , args : Value < 'v > , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```


<!-- END AUTOGEN: actions -->

<!-- BEGIN AUTOGEN: lib_move -->
### `lib_move.* — query primitives`

#### `hop_distance`

Lane-hop distance from `from_loc` to `target_loc`. `None` if unreachable.

```rust
fn hop_distance < 'v > (this : & LibMove , from_loc : & StarlarkLocation , target_loc : & StarlarkLocation , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `time_distance`

Time-distance (µs). `None` if no time table or unreachable.

```rust
fn time_distance < 'v > (this : & LibMove , from_loc : & StarlarkLocation , target_loc : & StarlarkLocation , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `blended_distance`

Convex blend of hop and time distance with weight `w_t` ∈ [0, 1].

```rust
fn blended_distance (this : & LibMove , from_loc : & StarlarkLocation , target_loc : & StarlarkLocation , w_t : StarlarkFloat ,) -> starlark :: Result < f64 >
```

#### `fastest_lane_us`

Fastest single-lane time across the architecture (µs).

```rust
fn fastest_lane_us (this : & LibMove) -> starlark :: Result < f64 >
```

#### `mobility`

Return the mobility score of `loc`: Σ 1/(1+hop_distance) over outgoing lanes × targets.

```rust
fn mobility < 'v > (this : & LibMove , loc : & StarlarkLocation , targets : & 'v ListRef < 'v > , heap : & 'v Heap ,) -> starlark :: Result < f64 >
```

#### `score_lanes`

Stage 1. Score every (qubit, lane) pair via `score_fn`; return a list of `ScoredLane`.

```rust
fn score_lanes < 'v > (this : & LibMove , config : & StarlarkConfig , ns : Value < 'v > , score_fn : Value < 'v > , ctx : Value < 'v > , eval : & mut starlark :: eval :: Evaluator < 'v , '_ , '_ > ,) -> starlark :: Result < Value < 'v > >
```

#### `top_c_per_qubit`

Stage 2. Retain the top-`c` lanes per qubit from `scored`, sorted by score descending.

```rust
fn top_c_per_qubit < 'v > (this : & LibMove , scored : & 'v ListRef < 'v > , c : i32 , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `group_by_triplet`

Stage 3. Group `scored` lanes by `(move_type, bus_id, direction)` triplet.

```rust
fn group_by_triplet < 'v > (this : & LibMove , scored : & 'v ListRef < 'v > , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `pack_aod_rectangles`

Stage 4. Pack triplet groups into AOD-compatible `PackedCandidate` rectangles.

```rust
fn pack_aod_rectangles < 'v > (this : & LibMove , groups : & 'v ListRef < 'v > , config : & StarlarkConfig , ctx : Value < 'v > , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```


<!-- END AUTOGEN: lib_move -->

<!-- BEGIN AUTOGEN: graph_handle -->
### `graph.* — read-only graph accessors`

#### `parent`

Parent of `node`, or None for the root.

```rust
fn parent < 'v > (this : & PolicyGraph , node : i32 , heap : & 'v Heap) -> starlark :: Result < Value < 'v > >
```

#### `depth`

Depth of `node` (root = 0).

```rust
fn depth (this : & PolicyGraph , node : i32) -> starlark :: Result < i32 >
```

#### `g_cost`

g-cost from root to `node`.

```rust
fn g_cost (this : & PolicyGraph , node : i32) -> starlark :: Result < f64 >
```

#### `children_of`

Insertion-order children of `node`.

```rust
fn children_of < 'v > (this : & PolicyGraph , node : i32 , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `path_from_root`

Return the path from root to `node` as a list of encoded-lane arrays.

```rust
fn path_from_root < 'v > (this : & PolicyGraph , node : i32 , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `ns`

Read the per-node DSL state. Returns a Starlark dict of fields.

```rust
fn ns < 'v > (this : & PolicyGraph , node : i32 , heap : & 'v Heap) -> starlark :: Result < Value < 'v > >
```

#### `config`

Return the qubit configuration at `node` as a `Config` value.

```rust
fn config < 'v > (this : & PolicyGraph , node : i32 , heap : & 'v Heap) -> starlark :: Result < Value < 'v > >
```

#### `is_goal`

Return `true` if `node`'s config matches the solve-start target config.

```rust
fn is_goal (this : & PolicyGraph , node : i32) -> starlark :: Result < bool >
```

#### `last_insert`

Outcome of the most recent `actions.insert_child(...)` action.

```rust
fn last_insert < 'v > (this : & PolicyGraph , heap : & 'v Heap) -> starlark :: Result < Value < 'v > >
```

#### `last_builtin_result`

Outcome of the most recent `actions.invoke_builtin(...)` call.

```rust
fn last_builtin_result < 'v > (this : & PolicyGraph , heap : & 'v Heap) -> starlark :: Result < Value < 'v > >
```


<!-- END AUTOGEN: graph_handle -->

<!-- BEGIN PROSE: move-tour -->
The four policies in `policies/reference/` cover the full range of Move policy
strategies, from a real production heuristic down to three educational examples
designed to expose specific parts of the API surface. Read them in the order listed
here.

### `entropy.star` — production heuristic (Plan A reference)

`entropy.star` is a Starlark reproduction of `Strategy::Entropy`, the Rust-native
heuristic that drives bloqade-lanes' default move synthesis. It is used as the
acid-test for parity between the Starlark DSL and the Rust heuristic: passing the
snapshot tests against `policies/fixtures/move/` at parity with `Strategy::Entropy`
is what "the DSL is feature-complete" means in practice.

The strategy is information-theoretic in spirit: each node in the search tree carries
an `entropy` integer that rises whenever a move attempt fails or a candidate is
exhausted. Low-entropy nodes prefer distance-reducing moves; high-entropy nodes
upweight mobility (future reach) to escape local minima. When entropy hits a ceiling
the policy walks up the ancestor chain and reverts to an earlier node, or falls back
to the sequential solver if the root itself is saturated.

API surface exercised:

- `graph.last_insert()` — reads the outcome of the previous `insert_child(...)` to decide whether to advance `current` or bump entropy on the parent
- `graph.ns(node)` — reads and updates per-node `{"entropy": int, "tried": list}` state
- `update_node_state(node, patch)` / `update_global_state(patch)` — mutates per-node and global state bags
- `lib.score_lanes(cfg, ns, score_fn, ctx)` → `lib.top_c_per_qubit(...)` → `lib.group_by_triplet(...)` → `lib.pack_aod_rectangles(...)` — the four-stage candidate pipeline
- `invoke_builtin("sequential_fallback", {...})` + `halt("fallback")` — graceful fallback path

Illustrative excerpt from `step(...)`:

`return update_global_state({"current": child_id, "pending_parent": None})`

Note that all action verbs (`insert_child`, `update_global_state`, `halt`, etc.) are
bare top-level globals in the Starlark namespace — there is no `actions.` prefix at
call sites, even though the AUTOGEN section above labels them `actions.*` for grouping
purposes.

### `dfs.star` — depth-first search (educational)

`dfs.star` implements the simplest possible tree traversal: always recurse on the most
recently inserted child. Each tick reads `graph.last_insert()` to discover whether the
previous `insert_child(...)` produced a new node, then immediately dives into that new
child by updating `current`. The branch index is tracked in global state so that when
a node has no new children to produce, the policy advances to the next sibling before
exhausting and halting with `"fallback"`.

API surface exercised:

- `graph.last_insert()` — the primary dive signal; `outcome["is_new"]` determines whether to recurse
- `graph.is_goal(node)` — checked after each successful insert and at the top of each tick
- `graph.config(node)` — passed to the four-stage pipeline to enumerate AOD-compatible move sets
- `insert_child(node, best.move_set.encoded)` — emits one expansion per tick
- `update_global_state({"current": child_id, "pending": None, "branch_index": 0})` — advances the frontier

Illustrative excerpt from `step(...)`:

`return update_global_state({"current": child_id, "pending": None, "branch_index": 0})`

`dfs.star` is the right starting point for understanding the tick-based execution
model. Read it before the other educational policies.

### `bfs.star` — breadth-first search (educational)

`bfs.star` visits nodes level-by-level by maintaining an explicit FIFO queue in policy
global state. It does not use `graph.last_insert()` to drive recursion; instead it
tracks a `phase` toggle between "insert" (emit one child of `expanding` per tick) and
"advance" (enqueue children via `graph.children_of(expanding)`, then pop the queue
head as the next parent).

There is a known v1 limitation: `insert_child(...)` actions are applied atomically
within a tick, so the child node-ids from a batch of inserts in one tick are not all
immediately available. `bfs.star` works around this by queuing *parent* node ids whose
children have already been inserted, then retrieving those children on the next tick
via `graph.children_of(parent)`. This approximates true BFS level-order: all nodes at
depth D are inserted before any node at depth D+1, but siblings within a depth level
are expanded in queue-pop order rather than strict insertion order.

API surface exercised:

- `graph.children_of(node)` — retrieves children after insertion to enqueue them for future expansion
- `graph.is_goal(node)` — checked on the expanding node and on each newly inserted child
- `update_global_state(patch)` — the only global-state mutation; no per-node state is used
- `insert_child(expanding, best.move_set.encoded)` — one child inserted per tick in the "insert" phase
- `halt("fallback", message)` — reached when the queue drains without finding the goal

Illustrative excerpt from `step(...)`:

`new_children = graph.children_of(expanding)`

`bfs.star` is the canonical demonstration that `update_global_state(...)` can carry
arbitrary list-structured state across ticks.

### `ids.star` — depth-bounded DFS (educational)

`ids.star` is documented in its header as "Iterative Deepening Search" but is in
practice **depth-bounded DFS**: a single solve runs DFS with a fixed depth cap
(`PARAMS["cap"]`, default 5). When the newly inserted child's depth meets or exceeds
the cap, the policy refuses to dive and instead tries the next sibling; when all
siblings are exhausted it backtracks via `graph.parent(node)`. If backtracking reaches
the root with no remaining candidates the policy halts with `"fallback"`.

True IDS would reset the frontier to the root and re-run with an incremented cap.
This is not implemented because `actions.reset_to_root()` does not exist in the v1
DSL surface. To approximate IDS in practice, run `eval-policy` in a loop with
increasing `--params` values for `cap`. This limitation is documented explicitly in
the policy's own docstring.

API surface exercised:

- `graph.depth(node)` — reads the depth of a newly inserted child to enforce the cap
- `graph.parent(node)` — drives backtracking when a node's candidates are exhausted
- `graph.last_insert()` — reads outcome to determine cap status before the dive decision
- `update_global_state({"current": parent, ...})` — backtracks the frontier
- `halt("fallback", message)` — reached when root is exhausted at the current cap

Illustrative excerpt from `step(...)`:

`child_depth = graph.depth(child_id)`

`ids.star` demonstrates that `graph.depth(...)` and `graph.parent(...)` are sufficient
to implement bounded backtracking without any additional kernel support. It also serves
as an honest record of what the v1 DSL surface cannot yet do.
<!-- END PROSE: move-tour -->

## Target Generator surface

<!-- BEGIN AUTOGEN: lib_target -->
### `lib_target.* — placement query primitives`

#### `cz_partner`

CZ blockade partner of `loc`, or `None`.

```rust
fn cz_partner < 'v > (this : & StarlarkLibTarget , loc : & StarlarkLocation , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```


<!-- END AUTOGEN: lib_target -->

<!-- BEGIN PROSE: target-tour -->
### `default_target.star` — CZ placement (Plan B reference)

`default_target.star` mirrors the in-tree `DefaultTargetGenerator` Rust heuristic.
It implements the `generate(ctx, lib)` contract: called once per CZ stage, it receives
the current qubit placement, the lists of control and target qubit ids, and any
look-ahead CZ layers, and it returns a list of candidate placement dicts (each dict
maps qubit ids to locations).

The strategy is simple: every qubit starts from its current placement, then for each
`(control, target)` pair the control qubit is moved to the CZ blockade partner of the
target qubit's location, looked up via `lib.cz_partner(loc)`. If any target qubit has
no CZ partner (returns `None`) the policy returns an empty list, deferring to the
fallback `DefaultTargetGenerator`.

```python
partner = lib.cz_partner(target[t])
if partner == None:
    return []
target[c] = partner
```

The Target surface is structurally simpler than the Move surface. There is no tick
loop, no transposition table, no `graph.last_insert()`, and no per-node state. The
policy is a single pure function: given the placement and the gate operands, produce
candidate placements. The kernel calls `generate(ctx, lib)` exactly once per CZ stage
and uses the returned list as the set of placements to evaluate.

`ctx` exposes `ctx.placement`, `ctx.controls`, `ctx.targets`,
`ctx.lookahead_cz_layers`, `ctx.cz_stage_index`, and `ctx.arch_spec`. `lib` exposes
`lib.cz_partner(loc)` and `lib.arch_spec` (an alias of `ctx.arch_spec`).

To write a new Target policy: copy `default_target.star`, change the candidate
generation logic inside `generate(ctx, lib)`, and return a list of placement dicts.
No `init(...)` function is needed — the Target surface has no persistent state across
stages. The `--problem <fixture>.json` for a Target policy uses `"kind": "target"`;
see the schema section below for the full fixture shape.
<!-- END PROSE: target-tour -->

## Problem fixture schema

<!-- BEGIN AUTOGEN: schema -->
### Problem fixture schema

Problem fixtures are JSON files with one of two top-level shapes, discriminated by a `"kind"` field.

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Problem",
  "oneOf": [
    {
      "type": "object",
      "required": [
        "arch",
        "initial",
        "kind",
        "target",
        "v"
      ],
      "properties": {
        "arch": {
          "type": "string"
        },
        "blocked": {
          "default": [],
          "type": "array",
          "items": {
            "type": "array",
            "items": {
              "type": "integer",
              "format": "int32"
            },
            "maxItems": 3,
            "minItems": 3
          }
        },
        "budget": {
          "default": null,
          "anyOf": [
            {
              "$ref": "#/definitions/Budget"
            },
            {
              "type": "null"
            }
          ]
        },
        "initial": {
          "type": "array",
          "items": {
            "type": "array",
            "items": [
              {
                "type": "integer",
                "format": "uint32",
                "minimum": 0.0
              },
              {
                "type": "array",
                "items": {
                  "type": "integer",
                  "format": "int32"
                },
                "maxItems": 3,
                "minItems": 3
              }
            ],
            "maxItems": 2,
            "minItems": 2
          }
        },
        "kind": {
          "type": "string",
          "enum": [
            "move"
          ]
        },
        "policy_params": {
          "default": null
        },
        "target": {
          "type": "array",
          "items": {
            "type": "array",
            "items": [
              {
                "type": "integer",
                "format": "uint32",
                "minimum": 0.0
              },
              {
                "type": "array",
                "items": {
                  "type": "integer",
                  "format": "int32"
                },
                "maxItems": 3,
                "minItems": 3
              }
            ],
            "maxItems": 2,
            "minItems": 2
          }
        },
        "v": {
          "type": "integer",
          "format": "uint32",
          "minimum": 0.0
        }
      }
    },
    {
      "type": "object",
      "required": [
        "arch",
        "controls",
        "current_placement",
        "kind",
        "targets",
        "v"
      ],
      "properties": {
        "arch": {
          "type": "string"
        },
        "controls": {
          "type": "array",
          "items": {
            "type": "integer",
            "format": "uint32",
            "minimum": 0.0
          }
        },
        "current_placement": {
          "type": "array",
          "items": {
            "type": "array",
            "items": [
              {
                "type": "integer",
                "format": "uint32",
                "minimum": 0.0
              },
              {
                "type": "array",
                "items": {
                  "type": "integer",
                  "format": "int32"
                },
                "maxItems": 3,
                "minItems": 3
              }
            ],
            "maxItems": 2,
            "minItems": 2
          }
        },
        "cz_stage_index": {
          "default": 0,
          "type": "integer",
          "format": "uint32",
          "minimum": 0.0
        },
        "kind": {
          "type": "string",
          "enum": [
            "target"
          ]
        },
        "lookahead_cz_layers": {
          "default": [],
          "type": "array",
          "items": {
            "type": "array",
            "items": [
              {
                "type": "array",
                "items": {
                  "type": "integer",
                  "format": "uint32",
                  "minimum": 0.0
                }
              },
              {
                "type": "array",
                "items": {
                  "type": "integer",
                  "format": "uint32",
                  "minimum": 0.0
                }
              }
            ],
            "maxItems": 2,
            "minItems": 2
          }
        },
        "policy_params": {
          "default": null
        },
        "targets": {
          "type": "array",
          "items": {
            "type": "integer",
            "format": "uint32",
            "minimum": 0.0
          }
        },
        "v": {
          "type": "integer",
          "format": "uint32",
          "minimum": 0.0
        }
      }
    }
  ],
  "definitions": {
    "Budget": {
      "type": "object",
      "required": [
        "max_expansions",
        "timeout_s"
      ],
      "properties": {
        "max_expansions": {
          "type": "integer",
          "format": "uint64",
          "minimum": 0.0
        },
        "timeout_s": {
          "type": "number",
          "format": "double"
        }
      }
    }
  }
}
```

<!-- END AUTOGEN: schema -->
