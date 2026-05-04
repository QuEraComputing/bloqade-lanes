//! Move-DSL action vocabulary.
//!
//! Six verbs the policy issues per `step()`. The kernel applies these
//! in returned order, atomically per tick.
//!
//! Wire format: each action is a Starlark dict shaped
//! `{"$action": "<verb>", ...verb-specific fields}`. The kernel converts
//! these dicts to `MoveAction` values via `MoveAction::try_from_json`.

use bloqade_lanes_dsl_core::errors::DslError;
use starlark::starlark_module;
use starlark::values::Value;
use starlark::values::dict::AllocDict;
use starlark::values::list::{AllocList, ListRef};

use crate::graph::{MoveSet, NodeId};

/// One verb the policy issues per `step()`. The kernel applies these
/// in returned order, atomically per tick.
#[derive(Debug, Clone, serde::Serialize)]
pub enum MoveAction {
    InsertChild { parent: NodeId, move_set: MoveSet },
    UpdateNodeState { node: NodeId, patch: PatchValue },
    UpdateGlobalState { patch: PatchValue },
    EmitSolution { node: NodeId },
    Halt { status: String, message: String },
    InvokeBuiltin { name: String, args: PatchValue },
}

/// A serialized field-update bundle. Stored as JSON for simplicity:
/// the schema check (§5.10 row "update_node_state with a field not in
/// schema") is performed in the kernel against the declared NodeState
/// schema, not here.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PatchValue(pub serde_json::Value);

impl MoveAction {
    /// Convert a Starlark-emitted dict (round-tripped through serde_json)
    /// into a typed `MoveAction`. The dict shape is documented at the
    /// top of this module.
    pub fn try_from_json(v: &serde_json::Value) -> Result<Self, DslError> {
        let kind = v
            .get("$action")
            .and_then(|s| s.as_str())
            .ok_or_else(|| DslError::BadPolicy("action dict missing $action key".into()))?;
        match kind {
            "insert_child" => {
                let parent = v
                    .get("parent")
                    .and_then(|p| p.as_u64().or_else(|| p.as_i64().map(|i| i as u64)))
                    .ok_or_else(|| {
                        DslError::BadPolicy("insert_child: parent must be integer".into())
                    })?;
                let lanes: Vec<u64> = v
                    .get("move_set_encoded")
                    .and_then(|l| l.as_array())
                    .ok_or_else(|| {
                        DslError::BadPolicy("insert_child: move_set_encoded missing".into())
                    })?
                    .iter()
                    .map(|x| {
                        x.as_u64()
                            .or_else(|| x.as_i64().map(|i| i as u64))
                            .unwrap_or(0)
                    })
                    .collect();
                Ok(MoveAction::InsertChild {
                    parent: NodeId(parent as u32),
                    move_set: MoveSet::from_encoded(lanes),
                })
            }
            "update_node_state" => {
                let node = v
                    .get("node")
                    .and_then(|p| p.as_u64().or_else(|| p.as_i64().map(|i| i as u64)))
                    .ok_or_else(|| {
                        DslError::BadPolicy("update_node_state: node must be integer".into())
                    })?;
                let patch = v.get("patch").cloned().unwrap_or(serde_json::Value::Null);
                Ok(MoveAction::UpdateNodeState {
                    node: NodeId(node as u32),
                    patch: PatchValue(patch),
                })
            }
            "update_global_state" => {
                let patch = v.get("patch").cloned().unwrap_or(serde_json::Value::Null);
                Ok(MoveAction::UpdateGlobalState {
                    patch: PatchValue(patch),
                })
            }
            "emit_solution" => {
                let node = v
                    .get("node")
                    .and_then(|p| p.as_u64().or_else(|| p.as_i64().map(|i| i as u64)))
                    .ok_or_else(|| {
                        DslError::BadPolicy("emit_solution: node must be integer".into())
                    })?;
                Ok(MoveAction::EmitSolution {
                    node: NodeId(node as u32),
                })
            }
            "halt" => Ok(MoveAction::Halt {
                status: v
                    .get("status")
                    .and_then(|s| s.as_str())
                    .unwrap_or("error")
                    .into(),
                message: v
                    .get("message")
                    .and_then(|s| s.as_str())
                    .unwrap_or("")
                    .into(),
            }),
            "invoke_builtin" => Ok(MoveAction::InvokeBuiltin {
                name: v.get("name").and_then(|s| s.as_str()).unwrap_or("").into(),
                args: PatchValue(
                    v.get("args")
                        .cloned()
                        .unwrap_or(serde_json::Value::Object(Default::default())),
                ),
            }),
            other => Err(DslError::BadPolicy(format!("unknown action kind: {other}"))),
        }
    }
}

/// Register the six action verbs into a `GlobalsBuilder`.
///
/// **Namespacing decision:** in starlark-rust 0.13 there's no built-in
/// way to register a sub-namespace via `#[starlark_module]`. We therefore
/// register the verbs as TOP-LEVEL globals: `insert_child(...)`,
/// `update_node_state(...)`, etc., rather than `actions.insert_child(...)`.
///
/// The spec §5.5 documents the surface as `actions.X` for readability
/// in policy code; users who want that exact ergonomic can declare a
/// `actions = struct(insert_child=insert_child, ...)` line at the top
/// of their `.star` file. The reference `entropy.star` will do this.
///
/// We chose this path because the alternative (building a frozen struct
/// at global-register time) requires runtime evaluator access that the
/// `register_X(GlobalsBuilder)` macro signature doesn't provide.
#[starlark_module]
pub fn register_actions(builder: &mut starlark::environment::GlobalsBuilder) {
    /// Emit an `insert_child` action — create a child of `parent` with the given encoded lanes.
    fn insert_child<'v>(
        parent: i32,
        move_set_encoded: &'v ListRef<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let heap = eval.heap();
        // Convert the Starlark list of integers into a Vec<i64>, then
        // re-emit as a Starlark list inside the dict (so the dict
        // round-trips through serde_json cleanly).
        let lanes: Vec<i64> = move_set_encoded
            .iter()
            .map(|v| v.unpack_i32().map(|i| i as i64).unwrap_or(0))
            .collect();
        let dict = AllocDict([
            ("$action", heap.alloc("insert_child")),
            ("parent", heap.alloc(parent)),
            ("move_set_encoded", heap.alloc(AllocList(lanes.into_iter()))),
        ]);
        Ok(heap.alloc(dict))
    }

    /// Emit an `update_node_state` action, merging `patch` dict into per-node DSL state for `node`.
    fn update_node_state<'v>(
        node: i32,
        patch: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let heap = eval.heap();
        let dict = AllocDict([
            ("$action", heap.alloc("update_node_state")),
            ("node", heap.alloc(node)),
            ("patch", patch),
        ]);
        Ok(heap.alloc(dict))
    }

    /// Emit an `update_global_state` action, merging `patch` into the solver-wide global state.
    fn update_global_state<'v>(
        patch: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let heap = eval.heap();
        let dict = AllocDict([
            ("$action", heap.alloc("update_global_state")),
            ("patch", patch),
        ]);
        Ok(heap.alloc(dict))
    }

    /// Emit an `emit_solution` action, recording `node` as a solved output.
    fn emit_solution<'v>(
        node: i32,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let heap = eval.heap();
        let dict = AllocDict([
            ("$action", heap.alloc("emit_solution")),
            ("node", heap.alloc(node)),
        ]);
        Ok(heap.alloc(dict))
    }

    /// Halt the search with `status` ∈ {"solved","unsolvable","fallback","error"}.
    fn halt<'v>(
        status: &str,
        #[starlark(default = "")] message: &str,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let heap = eval.heap();
        let dict = AllocDict([
            ("$action", heap.alloc("halt")),
            ("status", heap.alloc(status)),
            ("message", heap.alloc(message)),
        ]);
        Ok(heap.alloc(dict))
    }

    /// Invoke built-in `name` with `args` dict; result readable via `graph.last_builtin_result()`.
    fn invoke_builtin<'v>(
        name: &str,
        args: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let heap = eval.heap();
        let dict = AllocDict([
            ("$action", heap.alloc("invoke_builtin")),
            ("name", heap.alloc(name)),
            ("args", args),
        ]);
        Ok(heap.alloc(dict))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn dict_to_action_insert_child() {
        let dict = json!({
            "$action": "insert_child",
            "parent": 0,
            "move_set_encoded": [42u64, 99u64],
        });
        let action = MoveAction::try_from_json(&dict).expect("convert");
        match action {
            MoveAction::InsertChild { parent, move_set } => {
                assert_eq!(parent.0, 0);
                assert_eq!(move_set.encoded_lanes(), &[42, 99]);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn dict_to_action_unknown_kind_errors() {
        let dict = json!({"$action": "frob"});
        let err = MoveAction::try_from_json(&dict).expect_err("must fail");
        let msg = format!("{err}");
        assert!(msg.contains("frob"));
    }
}
