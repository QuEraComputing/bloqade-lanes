//! Starlark-visible mediated view of `SearchGraph`, plus side channels
//! for `last_insert` / `last_builtin_result`.
//!
//! `PolicyGraph` exposes a small read API to policy code (`graph.root`,
//! `graph.config(node)`, `graph.parent(node)`, etc.). All mutation flows
//! through `actions`, never through the handle. The kernel owns the
//! backing `PolicyGraphInner` and shares it with the Starlark value via
//! `Arc<Mutex<...>>`. `starlark_simple_value!` requires `Send + Sync`;
//! `Arc<Mutex<...>>` satisfies that requirement. The evaluator and kernel
//! run on a single thread, so `Mutex::lock()` never contends in practice.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

use starlark::starlark_module;
use starlark::values::dict::AllocDict;
use starlark::values::list::AllocList;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

use crate::config::Config;
use crate::graph::{MoveSet, NodeId, SearchGraph};
use crate::move_policy_dsl::lib_move::StarlarkConfig;

/// Outcome of the most recent `actions.insert_child(...)`. Mirrors the
/// spec §5.6 contract.
#[derive(Debug, Clone)]
pub struct InsertOutcome {
    pub child_id: Option<NodeId>,
    pub is_new: bool,
    pub error: Option<String>,
}

/// Outcome of the most recent `actions.invoke_builtin(...)`.
#[derive(Debug, Clone)]
pub struct BuiltinOutcome {
    pub status: String,
    pub payload: serde_json::Value,
}

/// Per-node DSL state, owned by the kernel. Generic JSON for the v1
/// scope: a future refinement could promote a typed `record` schema.
#[derive(Debug, Clone, Default)]
pub struct NodeStateMap(pub HashMap<NodeId, serde_json::Value>);

/// Internal mutable state for a `PolicyGraph`. The kernel owns this
/// directly; the Starlark `PolicyGraph` value sees it through an
/// `Arc<Mutex<...>>`.
pub struct PolicyGraphInner {
    pub graph: SearchGraph,
    pub node_state: NodeStateMap,
    pub last_insert: Option<InsertOutcome>,
    pub last_builtin: Option<BuiltinOutcome>,
    /// Insertion-order children-of map. The base SearchGraph stores
    /// only parent pointers, so we maintain a parallel forward map.
    pub children: HashMap<NodeId, Vec<NodeId>>,
    /// Target config for `is_goal` checks. Set once at solve start.
    pub target: Config,
}

/// Starlark-visible read handle. Construct via `PolicyGraph::new`.
///
/// Uses `Arc<Mutex<...>>` so the type is `Send + Sync` as required by
/// `starlark_simple_value!`. The evaluator and kernel run on a single
/// thread, so `Mutex::lock()` never contends.
#[derive(Clone, ProvidesStaticType, NoSerialize)]
pub struct PolicyGraph {
    inner: Arc<Mutex<PolicyGraphInner>>,
}

impl allocative::Allocative for PolicyGraph {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let v = visitor.enter_self_sized::<Self>();
        v.exit();
    }
}

starlark::starlark_simple_value!(PolicyGraph);

impl std::fmt::Debug for PolicyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PolicyGraph").finish()
    }
}

impl std::fmt::Display for PolicyGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<PolicyGraph>")
    }
}

impl PolicyGraph {
    /// Construct from kernel-owned inner state. Used by the kernel
    /// (Task 17) and by tests.
    pub fn new(inner: PolicyGraphInner) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Lock and borrow the underlying inner state. Panics if the mutex
    /// is poisoned (which never happens in normal single-threaded use).
    pub fn inner_borrow(&self) -> MutexGuard<'_, PolicyGraphInner> {
        self.inner.lock().expect("PolicyGraphInner mutex poisoned")
    }

    /// Lock and borrow the underlying inner state mutably. Used by the
    /// kernel to apply actions. Panics if the mutex is poisoned.
    pub fn inner_borrow_mut(&self) -> MutexGuard<'_, PolicyGraphInner> {
        self.inner.lock().expect("PolicyGraphInner mutex poisoned")
    }

    /// Clone the `Arc` so the kernel can keep a handle to the inner
    /// state independently of the Starlark value.
    pub fn inner_arc(&self) -> Arc<Mutex<PolicyGraphInner>> {
        Arc::clone(&self.inner)
    }

    fn lock(&self) -> MutexGuard<'_, PolicyGraphInner> {
        self.inner.lock().expect("PolicyGraphInner mutex poisoned")
    }
}

#[starlark::values::starlark_value(type = "PolicyGraph")]
impl<'v> StarlarkValue<'v> for PolicyGraph {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        let inner = self.lock();
        match attr {
            "root" => Some(heap.alloc(inner.graph.root().0 as i32)),
            "count" => Some(heap.alloc(inner.graph.len() as i32)),
            _ => None,
        }
    }

    fn get_methods() -> Option<&'static starlark::environment::Methods> {
        static METHODS: starlark::environment::MethodsStatic =
            starlark::environment::MethodsStatic::new();
        METHODS.methods(register_graph_methods)
    }
}

#[starlark_module]
fn register_graph_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Parent of `node`, or None for the root.
    fn parent<'v>(this: &PolicyGraph, node: i32, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        match this.lock().graph.parent(NodeId(node as u32)) {
            Some(p) => Ok(heap.alloc(p.0 as i32)),
            None => Ok(Value::new_none()),
        }
    }

    /// Depth of `node` (root = 0).
    fn depth(this: &PolicyGraph, node: i32) -> starlark::Result<i32> {
        Ok(this.lock().graph.depth(NodeId(node as u32)) as i32)
    }

    /// g-cost from root to `node`.
    fn g_cost(this: &PolicyGraph, node: i32) -> starlark::Result<f64> {
        Ok(this.lock().graph.g_score(NodeId(node as u32)))
    }

    /// Insertion-order children of `node`.
    fn children_of<'v>(
        this: &PolicyGraph,
        node: i32,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let inner = this.lock();
        let kids: Vec<i32> = inner
            .children
            .get(&NodeId(node as u32))
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|n| n.0 as i32)
            .collect();
        Ok(heap.alloc(AllocList(kids.into_iter())))
    }

    /// Path from root to `node` as a list of `move_set_encoded` lane
    /// arrays. (We expose the encoded lane vec rather than a typed
    /// `MoveSet` value to keep the wrapper type-set small for v1; the
    /// reference policies and acid test consume the encoded form.)
    fn path_from_root<'v>(
        this: &PolicyGraph,
        node: i32,
        heap: &'v Heap,
    ) -> starlark::Result<Value<'v>> {
        let path: Vec<MoveSet> = this.lock().graph.reconstruct_path(NodeId(node as u32));
        let lists: Vec<Value<'v>> = path
            .into_iter()
            .map(|ms: MoveSet| {
                let lanes: Vec<i64> = ms.encoded_lanes().iter().map(|&l| l as i64).collect();
                heap.alloc(AllocList(lanes.into_iter()))
            })
            .collect();
        Ok(heap.alloc(AllocList(lists.into_iter())))
    }

    /// Read the per-node DSL state. Returns a Starlark dict of fields.
    fn ns<'v>(this: &PolicyGraph, node: i32, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let inner = this.lock();
        let state = inner
            .node_state
            .0
            .get(&NodeId(node as u32))
            .cloned()
            .unwrap_or(serde_json::Value::Object(Default::default()));
        Ok(json_to_starlark(state, heap))
    }

    /// Configuration of `node`. Returns a real [`StarlarkConfig`] so that
    /// policy code can call `lib.score_lanes(graph.config(node), ...)` and
    /// related pipeline methods.
    fn config<'v>(this: &PolicyGraph, node: i32, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let inner = this.lock();
        let cfg = inner.graph.config(NodeId(node as u32)).clone();
        Ok(heap.alloc(StarlarkConfig(cfg)))
    }

    /// Goal predicate: returns `true` when the node's config equals the
    /// target config that was set at solve start.
    fn is_goal(this: &PolicyGraph, node: i32) -> starlark::Result<bool> {
        let inner = this.lock();
        let cfg = inner.graph.config(NodeId(node as u32));
        Ok(*cfg == inner.target)
    }

    /// Outcome of the most recent `actions.insert_child(...)` action.
    /// Returns a dict {child_id: int|None, is_new: bool, error: str|None}
    /// or None if no insert has been issued yet.
    fn last_insert<'v>(this: &PolicyGraph, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let inner = this.inner.lock().expect("PolicyGraphInner mutex poisoned");
        match &inner.last_insert {
            None => Ok(Value::new_none()),
            Some(o) => {
                let child_id_v: Value<'v> = match o.child_id {
                    Some(id) => heap.alloc(id.0 as i32),
                    None => Value::new_none(),
                };
                let error_v: Value<'v> = match &o.error {
                    Some(s) => heap.alloc(s.as_str()),
                    None => Value::new_none(),
                };
                let dict = AllocDict([
                    ("child_id", child_id_v),
                    ("is_new", heap.alloc(o.is_new)),
                    ("error", error_v),
                ]);
                Ok(heap.alloc(dict))
            }
        }
    }

    /// Outcome of the most recent `actions.invoke_builtin(...)` call.
    /// Returns a dict {status: str, payload: dict|list|primitive} or None.
    fn last_builtin_result<'v>(this: &PolicyGraph, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let inner = this.inner.lock().expect("PolicyGraphInner mutex poisoned");
        match &inner.last_builtin {
            None => Ok(Value::new_none()),
            Some(o) => {
                let payload_v = json_to_starlark(o.payload.clone(), heap);
                let dict = AllocDict([
                    ("status", heap.alloc(o.status.as_str())),
                    ("payload", payload_v),
                ]);
                Ok(heap.alloc(dict))
            }
        }
    }
}

/// Recursively convert a `serde_json::Value` to a Starlark `Value`.
/// Used by `graph.ns()` to expose the per-node JSON state to policies.
fn json_to_starlark<'v>(value: serde_json::Value, heap: &'v Heap) -> Value<'v> {
    use serde_json::Value as J;
    match value {
        J::Null => Value::new_none(),
        J::Bool(b) => Value::new_bool(b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                heap.alloc(i as i32)
            } else if let Some(f) = n.as_f64() {
                heap.alloc(f)
            } else {
                Value::new_none()
            }
        }
        J::String(s) => heap.alloc(s),
        J::Array(arr) => {
            let items: Vec<Value<'v>> =
                arr.into_iter().map(|v| json_to_starlark(v, heap)).collect();
            heap.alloc(AllocList(items.into_iter()))
        }
        J::Object(map) => {
            let pairs: Vec<(String, Value<'v>)> = map
                .into_iter()
                .map(|(k, v)| (k, json_to_starlark(v, heap)))
                .collect();
            heap.alloc(AllocDict(pairs))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

    #[test]
    fn root_attrs_via_inner() {
        let cfg = Config::new([(
            0u32,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        )])
        .unwrap();
        let inner = PolicyGraphInner {
            graph: SearchGraph::new(cfg.clone()),
            node_state: NodeStateMap::default(),
            last_insert: None,
            last_builtin: None,
            children: HashMap::new(),
            target: cfg,
        };
        let pg = PolicyGraph::new(inner);
        let inner_ref = pg.inner_borrow();
        assert_eq!(inner_ref.graph.root().0, 0);
        assert_eq!(inner_ref.graph.depth(inner_ref.graph.root()), 0);
        assert_eq!(inner_ref.graph.len(), 1);
    }
}
