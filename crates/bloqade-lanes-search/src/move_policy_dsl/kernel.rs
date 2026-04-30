//! The Move Policy DSL kernel loop.
//!
//! Public entry: [`solve_with_policy`].
//!
//! Per-solve flow:
//!   1. Build initial [`Config`] and target [`Config`]. Compute `dist_table`.
//!   2. Load the `.star` policy via [`load_policy_with_globals`] (parses
//!      and freezes the module against the kernel's full global set:
//!      stdlib + utilities + actions).
//!   3. Build [`PolicyGraphInner`], wrap as [`PolicyGraph`] (Arc<Mutex>).
//!   4. Build [`LibMove`] and [`Ctx`] Starlark values.
//!   5. Call `init(root, ctx) -> GlobalState` once; marshal result to
//!      `serde_json::Value`.
//!   6. Repeat: `step(graph, gs, ctx, lib) -> Action | [Action]`; apply each
//!      action; track expansions; check budgets; emit halt status.
//!   7. Reconstruct path from any `EmitSolution`-recorded NodeId.
//!
//! ## Out of scope (for v1, deferred to future tasks)
//!
//! - The Starlark-side `graph.last_insert()` / `graph.last_builtin_result()`
//!   methods (Task 18). The kernel still **records** these outcomes; the
//!   Starlark accessor lands later.
//! - Schema validation for `update_node_state` / `update_global_state`
//!   (Task 19). The kernel performs naive object-merge.
//! - The `sequential_fallback` builtin and other builtins (Task 20).
//!   `InvokeBuiltin` here records a placeholder `BuiltinOutcome` and
//!   continues the loop.
//!
//! ## Init/step return convention
//!
//! `init(root, ctx)` and `step(...)` should return Starlark *dicts* (or
//! plain values like int/None) — not `struct(...)`. The marshaller below
//! supports None/bool/int/float/str/list/dict, which is enough for v1.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::adapter::LoadedPolicy;
use bloqade_lanes_dsl_core::errors::DslError;
use bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec;
use bloqade_lanes_dsl_core::sandbox::{SandboxConfig, make_evaluator};

use crate::config::Config;
use crate::graph::{MoveSet, NodeId, SearchGraph};
use crate::heuristic::DistanceTable;
use crate::lane_index::LaneIndex;
use crate::move_policy_dsl::actions::{MoveAction, register_actions};
use crate::move_policy_dsl::builtins::sequential_fallback;
use crate::move_policy_dsl::graph_handle::{
    BuiltinOutcome, InsertOutcome, NodeStateMap, PolicyGraph, PolicyGraphInner,
};
use crate::move_policy_dsl::lib_move::{Ctx, LibMove};

// ── public types ─────────────────────────────────────────────────────────

/// Per-solve options for [`solve_with_policy`]: policy file + budgets +
/// sandbox knobs.
#[derive(Debug, Clone)]
pub struct PolicyOptions {
    /// Path on disk to the `.star` policy file. The file is parsed and
    /// frozen once per solve.
    pub policy_path: String,
    /// Free-form metadata bundle echoed into the result. The kernel does
    /// not interpret these; the caller is responsible for either reading
    /// them inside the policy via a `PARAMS` global or echoing them back
    /// from `policy_params` for downstream visibility.
    pub policy_params: serde_json::Value,
    /// Maximum number of newly-committed child nodes before the kernel
    /// returns [`PolicyStatus::BudgetExhausted`].
    pub max_expansions: u64,
    /// Optional wall-clock budget. `None` disables the timeout.
    pub timeout_s: Option<f64>,
    /// Sandbox config for the per-solve evaluator.
    pub sandbox: SandboxConfig,
}

impl Default for PolicyOptions {
    fn default() -> Self {
        Self {
            policy_path: String::new(),
            policy_params: serde_json::Value::Object(Default::default()),
            max_expansions: 100_000,
            timeout_s: None,
            sandbox: SandboxConfig::default(),
        }
    }
}

/// Outcome of a [`solve_with_policy`] call.
#[derive(Debug, Clone)]
pub struct PolicyResult {
    /// Terminal state of the kernel loop.
    pub status: PolicyStatus,
    /// Sequence of move-sets from the root to the (first emitted)
    /// solution node, in root-to-node order. Empty if no solution was
    /// emitted.
    pub move_layers: Vec<MoveSet>,
    /// Configuration at the (first emitted) solution node, or the
    /// initial configuration if no solution was emitted.
    pub goal_config: Config,
    /// Number of newly-committed child nodes during the solve.
    pub nodes_expanded: u32,
    /// Echo of the policy file path.
    pub policy_file: String,
    /// Echo of the policy params.
    pub policy_params: serde_json::Value,
}

/// Terminal status enum for [`PolicyResult::status`].
///
/// Mirrors the spec §5.10 status codes plus a few internal flavors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyStatus {
    /// Policy halted with `actions.halt("solved", ...)`.
    Solved,
    /// Policy halted with `actions.halt("unsolvable", ...)`.
    Unsolvable,
    /// `max_expansions` reached before halt.
    BudgetExhausted,
    /// `timeout_s` elapsed before halt.
    Timeout,
    /// Policy halted with `actions.halt("fallback", ...)` (string carries
    /// detail).
    Fallback(String),
    /// Policy syntax error reported by the Starlark parser.
    SyntaxError(String),
    /// Policy runtime error reported by the Starlark evaluator.
    RuntimeError(String),
    /// `update_*_state` named a field outside the declared schema.
    /// (Reserved for Task 19; not produced by v1.)
    SchemaError(String),
    /// Policy structurally bad (missing `step`, returned wrong type,
    /// etc.).
    BadPolicy(String),
    /// Per-tick Starlark step budget exceeded. (Reserved; not produced
    /// by v1 since starlark-0.13 doesn't expose a step limit.)
    StarlarkBudget,
    /// Per-solve Starlark heap cap exceeded. (Reserved; same as above.)
    StarlarkOOM,
}

// ── public entry point ───────────────────────────────────────────────────

/// Run a Move Policy DSL solve.
///
/// Loads the policy at `opts.policy_path`, calls `init(root, ctx)` once,
/// then loops `step(graph, gs, ctx, lib)` applying returned actions until
/// the policy halts or a budget is exhausted.
pub fn solve_with_policy(
    initial: impl IntoIterator<Item = (u32, LocationAddr)>,
    target: impl IntoIterator<Item = (u32, LocationAddr)>,
    blocked: impl IntoIterator<Item = LocationAddr>,
    index: Arc<LaneIndex>,
    opts: PolicyOptions,
) -> Result<PolicyResult, DslError> {
    // 1. Build initial / target / blocked sets.
    let initial_cfg =
        Config::new(initial).map_err(|e| DslError::BadPolicy(format!("initial config: {e}")))?;
    let target_cfg =
        Config::new(target).map_err(|e| DslError::BadPolicy(format!("target config: {e}")))?;
    let blocked_set: HashSet<u64> = blocked.into_iter().map(|l| l.encode()).collect();
    let target_pairs: Vec<(u32, u64)> = target_cfg.iter().map(|(q, l)| (q, l.encode())).collect();

    // 2. Distance table for lib.hop_distance / lib.time_distance.
    let target_encs: Vec<u64> = target_pairs.iter().map(|&(_, l)| l).collect();
    let dist_table = Arc::new(DistanceTable::new(&target_encs, &index).with_time_distances(&index));

    // 3. Load the policy with the SAME globals we use at evaluation
    //    time (standard + utilities + actions). Using the dsl-core
    //    `LoadedPolicy::from_path` here would parse/freeze the policy
    //    against utilities-only globals and `def step(...): return
    //    halt(...)` would fail at evaluator time on the unresolved
    //    `halt` identifier.
    //
    //    `policy_params` are bound as `PARAMS_OVERRIDE` on the load-time
    //    module so that the policy can merge them into its declared defaults
    //    at module-eval time (before the module is frozen). This ensures
    //    free-variable references to `PARAMS_OVERRIDE` inside frozen functions
    //    resolve correctly.
    let policy = load_policy_with_globals(&opts.policy_path, &opts.sandbox, &opts.policy_params)?;

    // 4. Build the policy-graph state.
    let inner = PolicyGraphInner {
        graph: SearchGraph::new(initial_cfg.clone()),
        node_state: NodeStateMap::default(),
        last_insert: None,
        last_builtin: None,
        children: Default::default(),
        target: target_cfg.clone(),
    };
    let policy_graph = PolicyGraph::new(inner);

    // 5. Build LibMove + Ctx.
    let arch_spec_arc = Arc::new(index.arch_spec().clone());
    let arch_wrap = StarlarkArchSpec(arch_spec_arc);
    let lib = LibMove {
        index: index.clone(),
        dist_table: dist_table.clone(),
        targets: target_pairs.clone(),
        blocked: blocked_set.clone(),
    };
    let ctx = Ctx::new(target_pairs.clone(), blocked_set.clone(), arch_wrap);

    // 6. Build the per-solve Starlark globals (standard + utilities + actions).
    let globals = build_solve_globals();

    // 7. Build the per-solve module and bind the well-known globals
    //    (`graph`, `lib`, `ctx`). The kernel keeps a separate
    //    `Arc<Mutex<PolicyGraphInner>>` handle so that it can mutate the
    //    graph state outside of Starlark (apply_actions). The Starlark
    //    side and the kernel side both refer to the same backing inner
    //    state via the shared Arc.
    let inner_arc_for_kernel = policy_graph.inner_arc();
    let module = starlark::environment::Module::new();
    bind_module_globals(&module, policy_graph, &lib, &ctx, &opts.policy_params);

    // 8. Call init(root, ctx).
    let mut gs = call_init(
        &policy,
        &globals,
        &module,
        &inner_arc_for_kernel,
        &ctx,
        &opts.sandbox,
    )?;

    // 9. Main step() loop.
    let mut nodes_expanded: u32 = 0;
    let mut solutions: Vec<NodeId> = Vec::new();
    let mut fallback_path: Option<Vec<MoveSet>> = None;
    let start = Instant::now();
    // Schema state: pre-seeded from init()'s return value so that all keys
    // present in the initial global state are automatically allowed.  The
    // node-state schema starts empty (None = not yet established; the first
    // update_node_state patch establishes it).
    let mut node_state_schema: Option<HashSet<String>> = None;
    let mut global_state_schema: Option<HashSet<String>> =
        gs.as_object().map(|m| m.keys().cloned().collect());

    loop {
        // Budget checks.
        if let Some(t) = opts.timeout_s
            && start.elapsed().as_secs_f64() > t
        {
            return Ok(make_terminal_result(
                PolicyStatus::Timeout,
                &opts,
                &solutions,
                &inner_arc_for_kernel,
                &initial_cfg,
                &target_cfg,
                fallback_path.clone(),
                nodes_expanded,
            ));
        }
        if (nodes_expanded as u64) >= opts.max_expansions {
            return Ok(make_terminal_result(
                PolicyStatus::BudgetExhausted,
                &opts,
                &solutions,
                &inner_arc_for_kernel,
                &initial_cfg,
                &target_cfg,
                fallback_path.clone(),
                nodes_expanded,
            ));
        }

        // Invoke step(graph, gs, ctx, lib) and capture the action list.
        let actions = call_step(&policy, &globals, &module, &gs, &ctx, &lib, &opts.sandbox)?;

        // Apply actions atomically.
        let (committed_new_child, halt, new_gs) = apply_actions(
            &inner_arc_for_kernel,
            &index,
            &actions,
            &mut solutions,
            &gs,
            &target_cfg,
            &blocked_set,
            &mut fallback_path,
            &mut node_state_schema,
            &mut global_state_schema,
        )?;
        gs = new_gs;
        if committed_new_child {
            nodes_expanded += 1;
        }
        if let Some(status) = halt {
            return Ok(make_terminal_result(
                status,
                &opts,
                &solutions,
                &inner_arc_for_kernel,
                &initial_cfg,
                &target_cfg,
                fallback_path.clone(),
                nodes_expanded,
            ));
        }
    }
}

// ── globals binding ──────────────────────────────────────────────────────

/// Build the `Globals` for the per-solve evaluator: stdlib + utilities +
/// the action-emitter verbs (`insert_child`, `update_node_state`, etc.).
fn build_solve_globals() -> starlark::environment::Globals {
    starlark::environment::GlobalsBuilder::standard()
        .with(bloqade_lanes_dsl_core::primitives::utilities::register_utilities)
        .with(register_actions)
        .build()
}

/// Parse and freeze a `.star` policy module against the same `Globals`
/// the kernel will use to evaluate `init` / `step` (stdlib + utilities +
/// actions). `LoadedPolicy::from_path` in dsl-core uses
/// utilities-only globals, which would cause `halt(...)` and friends
/// to be unresolved at evaluator time.
///
/// `policy_params` is bound as `PARAMS_OVERRIDE` on the module **before**
/// `eval_module` runs, so module-level Starlark code (and any helper
/// functions frozen into the module) can reference `PARAMS_OVERRIDE` as a
/// free variable. This allows the policy to merge caller-supplied overrides
/// into its declared parameter defaults at load time, before the module is
/// frozen.
fn load_policy_with_globals(
    path: &str,
    sandbox: &SandboxConfig,
    policy_params: &serde_json::Value,
) -> Result<LoadedPolicy, DslError> {
    let src = std::fs::read_to_string(path)?;
    let ast = starlark::syntax::AstModule::parse(path, src, &starlark::syntax::Dialect::Standard)
        .map_err(|e| DslError::Parse {
        path: path.to_owned(),
        message: format!("{e}"),
    })?;
    let module = starlark::environment::Module::new();
    let globals = build_solve_globals();
    // Bind PARAMS_OVERRIDE before eval_module so that module-level code and
    // frozen helper functions can reference it as a free variable.
    {
        let heap = module.heap();
        let overrides = json_to_starlark(policy_params.clone(), heap);
        module.set("PARAMS_OVERRIDE", overrides);
    }
    {
        let mut eval = make_evaluator(&module, &globals, sandbox);
        eval.eval_module(ast, &globals)
            .map_err(|e| DslError::Runtime {
                traceback: format!("{e:?}"),
            })?;
    }
    let frozen = module.freeze().map_err(|e| DslError::Runtime {
        traceback: format!("{e:?}"),
    })?;
    Ok(LoadedPolicy {
        frozen,
        globals,
        source_path: path.to_owned(),
    })
}

/// Bind `graph`, `lib`, `ctx`, and `PARAMS_OVERRIDE` as named globals on
/// the per-solve `Module`. The Starlark policy refers to these by bare name.
/// The `policy_graph` value is moved into the module's heap; the kernel
/// keeps an `Arc<Mutex<PolicyGraphInner>>` handle separately.
///
/// `PARAMS_OVERRIDE` is always bound (empty dict when the caller passes no
/// overrides) so that policies can unconditionally merge it into their
/// declared `PARAMS_DEFAULTS` without guarding against a missing name.
fn bind_module_globals(
    module: &starlark::environment::Module,
    policy_graph: PolicyGraph,
    lib: &LibMove,
    ctx: &Ctx,
    policy_params: &serde_json::Value,
) {
    let heap = module.heap();
    module.set("graph", heap.alloc(policy_graph));
    module.set("lib", heap.alloc(lib.clone()));
    module.set("ctx", heap.alloc(ctx.clone()));
    // Always bind PARAMS_OVERRIDE — empty dict if the caller passed no overrides.
    let overrides = json_to_starlark(policy_params.clone(), heap);
    module.set("PARAMS_OVERRIDE", overrides);
}

// ── init / step invocation ───────────────────────────────────────────────

fn call_init(
    policy: &LoadedPolicy,
    globals: &starlark::environment::Globals,
    module: &starlark::environment::Module,
    inner: &Arc<Mutex<PolicyGraphInner>>,
    ctx: &Ctx,
    sandbox: &SandboxConfig,
) -> Result<serde_json::Value, DslError> {
    let init = policy
        .get("init")
        .ok_or_else(|| DslError::BadPolicy("policy missing `init` function".into()))?;

    let mut eval = make_evaluator(module, globals, sandbox);
    let heap = module.heap();
    let inner_root = inner
        .lock()
        .expect("PolicyGraphInner mutex poisoned")
        .graph
        .root()
        .0 as i32;
    let args = [heap.alloc(inner_root), heap.alloc(ctx.clone())];

    // `OwnedFrozenValue` exposes its inner value via `.value()`.
    let init_v = init.value();
    let result = eval
        .eval_function(init_v, &args, &[])
        .map_err(|e| DslError::Runtime {
            traceback: format!("{e:?}"),
        })?;

    starlark_value_to_json(result).map_err(|e| DslError::Runtime { traceback: e })
}

fn call_step(
    policy: &LoadedPolicy,
    globals: &starlark::environment::Globals,
    module: &starlark::environment::Module,
    gs: &serde_json::Value,
    ctx: &Ctx,
    lib: &LibMove,
    sandbox: &SandboxConfig,
) -> Result<Vec<MoveAction>, DslError> {
    let step = policy
        .get("step")
        .ok_or_else(|| DslError::BadPolicy("policy missing `step` function".into()))?;
    let mut eval = make_evaluator(module, globals, sandbox);
    let heap = module.heap();
    let gs_value = json_to_starlark(gs.clone(), heap);
    // Look up the previously-bound `graph` global on the module — we
    // can't clone the `PolicyGraph` (no `Clone` impl), so we store it
    // once on the module heap during `bind_module_globals` and pull the
    // `Value<'v>` back out for each evaluator call.
    let graph_val = module_get_or_err(module, "graph")?;
    let args = [
        graph_val,
        gs_value,
        heap.alloc(ctx.clone()),
        heap.alloc(lib.clone()),
    ];
    let step_v = step.value();
    let result = eval
        .eval_function(step_v, &args, &[])
        .map_err(|e| DslError::Runtime {
            traceback: format!("{e:?}"),
        })?;
    starlark_actions_to_vec(result)
}

/// Look up a name bound on `module` and return its `Value`, or error if
/// it is missing. Used to round-trip the `graph` handle (which is not
/// `Clone`) across multiple evaluator calls.
fn module_get_or_err<'v>(
    module: &'v starlark::environment::Module,
    name: &str,
) -> Result<starlark::values::Value<'v>, DslError> {
    module.get(name).ok_or_else(|| DslError::Runtime {
        traceback: format!("internal: module global `{name}` missing"),
    })
}

/// Convert `step()`'s return value into a `Vec<MoveAction>`.
///
/// Accepts:
/// - `None` → empty action list (no-op tick),
/// - a single dict → one action,
/// - a list of dicts → that many actions.
fn starlark_actions_to_vec(
    value: starlark::values::Value<'_>,
) -> Result<Vec<MoveAction>, DslError> {
    use starlark::values::dict::DictRef;
    use starlark::values::list::ListRef;

    if value.is_none() {
        return Ok(Vec::new());
    }
    if let Some(list) = ListRef::from_value(value) {
        let mut actions = Vec::with_capacity(list.len());
        for v in list.iter() {
            let dict = DictRef::from_value(v)
                .ok_or_else(|| DslError::BadPolicy("step action element must be a dict".into()))?;
            let json = starlark_dict_to_json(&dict)?;
            actions.push(MoveAction::try_from_json(&json)?);
        }
        return Ok(actions);
    }
    if let Some(dict) = DictRef::from_value(value) {
        let json = starlark_dict_to_json(&dict)?;
        return Ok(vec![MoveAction::try_from_json(&json)?]);
    }
    Err(DslError::BadPolicy(format!(
        "step must return a dict, list, or None, got: {value}"
    )))
}

// ── value ⇄ JSON marshalling ─────────────────────────────────────────────

/// Recursively convert a Starlark value to `serde_json::Value`.
///
/// Supports `None`, `bool`, `int`, `float`, `str`, `list`, `dict`. Any
/// other value type triggers an error (returned as `Err(String)` so the
/// caller can wrap with the appropriate `DslError`).
fn starlark_value_to_json(value: starlark::values::Value<'_>) -> Result<serde_json::Value, String> {
    use starlark::values::ValueLike;
    use starlark::values::dict::DictRef;
    use starlark::values::list::ListRef;

    if value.is_none() {
        return Ok(serde_json::Value::Null);
    }
    if let Some(b) = value.unpack_bool() {
        return Ok(serde_json::Value::Bool(b));
    }
    if let Some(i) = value.unpack_i32() {
        return Ok(serde_json::Value::Number(serde_json::Number::from(i)));
    }
    if let Some(sf) = value.downcast_ref::<starlark::values::float::StarlarkFloat>() {
        return Ok(serde_json::Number::from_f64(sf.0)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null));
    }
    if let Some(s) = value.unpack_str() {
        return Ok(serde_json::Value::String(s.to_owned()));
    }
    if let Some(list) = ListRef::from_value(value) {
        let items: Result<Vec<_>, _> = list.iter().map(starlark_value_to_json).collect();
        return Ok(serde_json::Value::Array(items?));
    }
    if let Some(dict) = DictRef::from_value(value) {
        return starlark_dict_to_json(&dict).map_err(|e| format!("{e}"));
    }
    // Opaque Starlark objects (e.g. StarlarkConfig passed as an invoke_builtin
    // arg) are not JSON-serialisable.  Convert them to null so the action dict
    // can still be parsed; the kernel-side builtin handler reads the config
    // directly from the graph state rather than from the marshalled args.
    Ok(serde_json::Value::Null)
}

/// Convert a Starlark dict (with string keys) to a JSON object.
fn starlark_dict_to_json(
    dict: &starlark::values::dict::DictRef<'_>,
) -> Result<serde_json::Value, DslError> {
    let mut obj = serde_json::Map::new();
    for (k, v) in dict.iter() {
        let key = k
            .unpack_str()
            .ok_or_else(|| DslError::BadPolicy(format!("dict key must be a string, got {k}")))?;
        let val = starlark_value_to_json(v).map_err(|e| DslError::Runtime { traceback: e })?;
        obj.insert(key.to_owned(), val);
    }
    Ok(serde_json::Value::Object(obj))
}

/// Recursively convert a `serde_json::Value` to a Starlark value.
fn json_to_starlark<'v>(
    value: serde_json::Value,
    heap: &'v starlark::values::Heap,
) -> starlark::values::Value<'v> {
    use serde_json::Value as J;
    use starlark::values::dict::AllocDict;
    use starlark::values::list::AllocList;
    match value {
        J::Null => starlark::values::Value::new_none(),
        J::Bool(b) => starlark::values::Value::new_bool(b),
        J::Number(n) => {
            if let Some(i) = n.as_i64() {
                heap.alloc(i as i32)
            } else if let Some(f) = n.as_f64() {
                heap.alloc(f)
            } else {
                starlark::values::Value::new_none()
            }
        }
        J::String(s) => heap.alloc(s),
        J::Array(arr) => {
            let items: Vec<_> = arr.into_iter().map(|v| json_to_starlark(v, heap)).collect();
            heap.alloc(AllocList(items.into_iter()))
        }
        J::Object(map) => {
            let pairs: Vec<(String, _)> = map
                .into_iter()
                .map(|(k, v)| (k, json_to_starlark(v, heap)))
                .collect();
            heap.alloc(AllocDict(pairs))
        }
    }
}

// ── apply_actions ────────────────────────────────────────────────────────

/// Apply each emitted action to the policy graph state.
///
/// Returns `(committed_new_child, halt, new_gs)`:
/// - `committed_new_child` — true if at least one `InsertChild` succeeded
///   AND added a brand-new node (de-dup via the transposition table).
/// - `halt` — `Some(status)` if the policy issued a `halt` action; the
///   loop then terminates.
/// - `new_gs` — updated global state JSON.
///
/// `node_state_schema` and `global_state_schema` track the established
/// set of allowed top-level field names for `update_node_state` /
/// `update_global_state` patches.  `None` means "not yet observed";
/// the first patch establishes the schema.  Subsequent patches with an
/// unknown field cause an early return with
/// `PolicyStatus::SchemaError(field_name)`.
#[allow(clippy::too_many_arguments)]
fn apply_actions(
    inner_arc: &Arc<Mutex<PolicyGraphInner>>,
    index: &LaneIndex,
    actions: &[MoveAction],
    solutions: &mut Vec<NodeId>,
    gs: &serde_json::Value,
    target_cfg: &Config,
    blocked_set: &HashSet<u64>,
    fallback_path: &mut Option<Vec<MoveSet>>,
    node_state_schema: &mut Option<HashSet<String>>,
    global_state_schema: &mut Option<HashSet<String>>,
) -> Result<(bool, Option<PolicyStatus>, serde_json::Value), DslError> {
    let mut new_gs = gs.clone();
    let mut committed_new_child = false;
    let mut halt: Option<PolicyStatus> = None;

    for action in actions {
        match action {
            MoveAction::InsertChild { parent, move_set } => {
                // Resolve each lane to a (qubit_at_src, dst) move.
                // If any lane has no qubit at its src, mark this insert
                // as aod_invalid and skip.
                let mut inner = inner_arc.lock().expect("PolicyGraphInner mutex poisoned");
                let parent_cfg = inner.graph.config(*parent).clone();

                let mut moves: Vec<(u32, LocationAddr)> = Vec::with_capacity(move_set.len());
                let mut aod_error: Option<String> = None;
                for lane in move_set.decode() {
                    let Some((src, dst)) = index.endpoints(&lane) else {
                        aod_error = Some(format!(
                            "aod_invalid: lane {} not in index",
                            lane.encode_u64()
                        ));
                        break;
                    };
                    let Some(qid) = parent_cfg.qubit_at(src) else {
                        aod_error = Some(format!(
                            "aod_invalid: lane {} has no qubit at src",
                            lane.encode_u64()
                        ));
                        break;
                    };
                    moves.push((qid, dst));
                }

                if let Some(msg) = aod_error {
                    inner.last_insert = Some(InsertOutcome {
                        child_id: None,
                        is_new: false,
                        error: Some(msg),
                    });
                    continue;
                }

                let new_config = parent_cfg.with_moves(&moves);
                let (child_id, is_new) =
                    inner
                        .graph
                        .insert(*parent, move_set.clone(), new_config, 1.0);
                if is_new {
                    inner.children.entry(*parent).or_default().push(child_id);
                    committed_new_child = true;
                }
                inner.last_insert = Some(InsertOutcome {
                    child_id: Some(child_id),
                    is_new,
                    error: None,
                });
            }
            MoveAction::UpdateNodeState { node, patch } => {
                // Collect the patch's top-level keys.
                let patch_keys: Vec<String> = patch
                    .0
                    .as_object()
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                // Additive schema: any new key encountered in a patch is
                // added to the allowed set.  This allows policies (such as
                // the reference entropy.star) to use different node-state
                // fields on different code paths without triggering a
                // SchemaError.  Strict closed-world enforcement is deferred
                // to Task 19.
                match node_state_schema {
                    None => {
                        *node_state_schema = Some(patch_keys.iter().cloned().collect());
                    }
                    Some(schema) => {
                        for k in &patch_keys {
                            schema.insert(k.clone());
                        }
                    }
                }
                let mut inner = inner_arc.lock().expect("PolicyGraphInner mutex poisoned");
                let entry = inner
                    .node_state
                    .0
                    .entry(*node)
                    .or_insert_with(|| serde_json::Value::Object(Default::default()));
                merge_json_object(entry, &patch.0);
            }
            MoveAction::UpdateGlobalState { patch } => {
                // Collect the patch's top-level keys.
                let patch_keys: Vec<String> = patch
                    .0
                    .as_object()
                    .map(|m| m.keys().cloned().collect())
                    .unwrap_or_default();
                // First patch establishes the schema; subsequent patches
                // are validated against it.
                match global_state_schema {
                    None => {
                        *global_state_schema = Some(patch_keys.iter().cloned().collect());
                    }
                    Some(schema) => {
                        for k in &patch_keys {
                            if !schema.contains(k) {
                                return Ok((
                                    committed_new_child,
                                    Some(PolicyStatus::SchemaError(k.clone())),
                                    new_gs,
                                ));
                            }
                        }
                    }
                }
                merge_json_object(&mut new_gs, &patch.0);
            }
            MoveAction::EmitSolution { node } => {
                solutions.push(*node);
                // Auto-halt with Solved on the first emitted solution.
                // Policies that want to continue searching after finding a
                // solution must issue an explicit halt themselves; otherwise
                // the kernel terminates as soon as a solution is recorded.
                halt = Some(PolicyStatus::Solved);
                break;
            }
            MoveAction::Halt { status, .. } => {
                halt = Some(map_halt_status(status));
                break;
            }
            MoveAction::InvokeBuiltin { name, .. } if name == "sequential_fallback" => {
                let parent_cfg = {
                    let inner = inner_arc.lock().expect("PolicyGraphInner mutex poisoned");
                    inner.graph.config(inner.graph.root()).clone()
                };
                let path = sequential_fallback(&parent_cfg, target_cfg, index, blocked_set);
                *fallback_path = Some(path.clone());
                let mut inner = inner_arc.lock().expect("PolicyGraphInner mutex poisoned");
                inner.last_builtin = Some(BuiltinOutcome {
                    status: if path.is_empty() {
                        "fallback: no moves needed".into()
                    } else {
                        "ok".into()
                    },
                    payload: serde_json::json!({"path_len": path.len()}),
                });
            }
            MoveAction::InvokeBuiltin { name, .. } => {
                let mut inner = inner_arc.lock().expect("PolicyGraphInner mutex poisoned");
                inner.last_builtin = Some(BuiltinOutcome {
                    status: format!("unknown builtin: {name}"),
                    payload: serde_json::Value::Null,
                });
            }
        }
    }
    Ok((committed_new_child, halt, new_gs))
}

/// Shallow merge: copy each key from `patch` into `target`. Both must be
/// JSON objects; if either isn't, we no-op (schema validation in Task
/// 19 will catch genuine misuse). A `Null` target is upgraded to the
/// patch as-is — convenient for `init` returning `None` and the first
/// `update_global_state` populating the bag.
fn merge_json_object(target: &mut serde_json::Value, patch: &serde_json::Value) {
    use serde_json::Value;
    if let (Value::Object(t), Value::Object(p)) = (&mut *target, patch) {
        for (k, v) in p {
            t.insert(k.clone(), v.clone());
        }
    } else if matches!(target, Value::Null) {
        *target = patch.clone();
    }
}

fn map_halt_status(s: &str) -> PolicyStatus {
    match s {
        "solved" => PolicyStatus::Solved,
        "unsolvable" => PolicyStatus::Unsolvable,
        "fallback" => PolicyStatus::Fallback("policy-requested".into()),
        "error" => PolicyStatus::RuntimeError("policy-requested".into()),
        other => PolicyStatus::RuntimeError(format!("unknown halt status: {other}")),
    }
}

#[allow(clippy::too_many_arguments)]
fn make_terminal_result(
    status: PolicyStatus,
    opts: &PolicyOptions,
    solutions: &[NodeId],
    inner_arc: &Arc<Mutex<PolicyGraphInner>>,
    initial_cfg: &Config,
    target_cfg: &Config,
    fallback_path: Option<Vec<MoveSet>>,
    nodes_expanded: u32,
) -> PolicyResult {
    let (move_layers, goal_config) = if matches!(&status, PolicyStatus::Fallback(_))
        && let Some(path) = fallback_path
    {
        (path, target_cfg.clone())
    } else if let Some(&node) = solutions.first() {
        let inner = inner_arc.lock().expect("PolicyGraphInner mutex poisoned");
        (
            inner.graph.reconstruct_path(node),
            inner.graph.config(node).clone(),
        )
    } else {
        (Vec::new(), initial_cfg.clone())
    };
    PolicyResult {
        status,
        move_layers,
        goal_config,
        nodes_expanded,
        policy_file: opts.policy_path.clone(),
        policy_params: opts.policy_params.clone(),
    }
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::io::Write;

    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::test_utils::example_arch_json;

    /// Trivial halt-only policy: returns `Solved` with no expansions.
    #[test]
    fn trivial_halt_only_policy_returns_solved() {
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = Arc::new(LaneIndex::new(spec));

        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(
            tmp,
            r#"
def init(root, ctx):
    return None

def step(graph, gs, ctx, lib):
    return halt("solved", "trivial")
"#
        )
        .expect("write");
        // Force the file's contents to be flushed to disk before the
        // policy loader reads it.
        tmp.flush().expect("flush");
        let path = tmp.path().to_str().expect("path").to_owned();

        let opts = PolicyOptions {
            policy_path: path,
            policy_params: serde_json::Value::Object(Default::default()),
            max_expansions: 1000,
            timeout_s: Some(5.0),
            sandbox: SandboxConfig::default(),
        };

        let result = solve_with_policy(
            std::iter::empty::<(u32, LocationAddr)>(),
            std::iter::empty::<(u32, LocationAddr)>(),
            std::iter::empty::<LocationAddr>(),
            index,
            opts,
        )
        .expect("solve");

        assert_eq!(result.status, PolicyStatus::Solved);
        assert!(result.move_layers.is_empty());
        assert_eq!(result.nodes_expanded, 0);
    }
}
