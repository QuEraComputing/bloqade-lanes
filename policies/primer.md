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

From Python, use `PolicyRunner(arch_json).solve(..., policy_path=...)` for one-shot
move synthesis, or wire `PolicyPlacementStrategy(traversal=PolicyTraversal(policy_path=...))`
(from `bloqade.lanes.heuristics.physical.policy_movement`) into a placement
pipeline. Target Generator DSLs plug in via the existing
`PhysicalPlacementStrategy(target_generator=...)` extension point.

From the command line, use the `bloqade-bytecode` CLI: `eval-policy --policy
<file>.star --problem <fixture>.json` runs a policy end-to-end and prints a result
summary; `trace-policy --policy <file>.star --problem <fixture>.json` runs with a
verbose observer and emits one record per kernel event, useful for debugging step logic.
Both subcommands accept `--json` for structured output and `--params <path>` to override
`PARAMS_DEFAULTS` at load time.

For the full specification see
`docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md`. The Move
Policy DSL surface below is the source of truth for strategy authors. Avoid
treating old strategy examples as templates: search ordering, pruning, and
frontier management are policy-owned choices, and premature branch caps can
prevent the solver from ever reaching a valid placement.
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

#### `unresolved_qubits`

Return unresolved qubits as dicts: {"qid", "current", "target"}.

```rust
fn unresolved_qubits < 'v > (this : & LibMove , config : & StarlarkConfig , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `legal_lanes`

Return outgoing legal lanes for `qid`, skipping occupied or blocked destinations.

```rust
fn legal_lanes < 'v > (this : & LibMove , config : & StarlarkConfig , qid : i32 , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `scored_lane`

Construct a `ScoredLane` record from policy-owned scoring logic.

```rust
fn scored_lane < 'v > (this : & LibMove , qid : i32 , lane : & StarlarkLane , score : Value < 'v > , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
```

#### `pack_aod_rectangles`

Pack scored lanes into AOD-compatible `PackedCandidate` rectangles.

```rust
fn pack_aod_rectangles < 'v > (this : & LibMove , scored : & 'v ListRef < 'v > , config : & StarlarkConfig , ctx : Value < 'v > , heap : & 'v Heap ,) -> starlark :: Result < Value < 'v > >
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
Move policy authors should use the API reference above directly rather than copying
old strategy examples. In particular, do not infer that per-qubit top-k pruning,
small branch caps, or fallback halts are part of the DSL contract. Those are policy
choices, not kernel requirements, and they can prevent physical placement from
reaching the exact target config needed to lower every CZ.

A successful Move policy must keep exploring until it emits a node whose config
matches the target placement, exhausts a real frontier, hits the kernel expansion
budget, or times out. If compilation leaves `place.CZ` statements behind, at least
one CZ pair was not placed on compatible neighboring physical sites; optimize search
completeness before optimizing move count or wall time.
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
