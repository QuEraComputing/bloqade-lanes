<!-- AUTOGEN: DO NOT EDIT BY HAND.
     Regenerate with `just generate-primer`. -->

# Move Policy & Target Generator DSL — Primer

<!-- BEGIN PROSE: intro -->
TODO: write prose for intro
<!-- END PROSE: intro -->

## Move Policy surface

<!-- BEGIN AUTOGEN: actions -->
### `actions.* — kernel-driven verbs`

#### `insert_child`

Stub-doc: insert child.

```rust
fn insert_child (parent : i32 , write : i32) -> anyhow :: Result < () >
```

#### `halt`

Stub-doc: halt with reason.

```rust
fn halt (reason : & str) -> anyhow :: Result < () >
```


<!-- END AUTOGEN: actions -->

<!-- BEGIN AUTOGEN: lib_move -->
### `lib_move.* — query primitives`

#### `hop_distance`

Stub-doc: hop distance.

```rust
fn hop_distance (qubit : u32) -> anyhow :: Result < u32 >
```


<!-- END AUTOGEN: lib_move -->

<!-- BEGIN AUTOGEN: graph_handle -->
### `graph.* — read-only graph accessors`

#### `depth`

Stub-doc: depth of node.

```rust
fn depth (node : i32) -> anyhow :: Result < u32 >
```


<!-- END AUTOGEN: graph_handle -->

<!-- BEGIN PROSE: move-tour -->
TODO: write prose for move-tour
<!-- END PROSE: move-tour -->

## Target Generator surface

<!-- BEGIN AUTOGEN: lib_target -->
### `lib_target.* — placement query primitives`

#### `cz_partner`

Stub-doc: cz partner.

```rust
fn cz_partner (qubit : u32) -> anyhow :: Result < u32 >
```


<!-- END AUTOGEN: lib_target -->

<!-- BEGIN PROSE: target-tour -->
TODO: write prose for target-tour
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
