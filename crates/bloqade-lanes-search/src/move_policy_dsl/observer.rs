//! Observer trait for the Move Policy DSL kernel.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §5.

use std::io::Write;

use serde::Serialize;

use crate::move_policy_dsl::actions::MoveAction;
use crate::move_policy_dsl::kernel::PolicyStatus;

/// Snapshot of the policy graph at the moment `on_init` is called.
/// Data-only mirror of relevant fields on `PolicyGraphInner`.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PolicyGraphSnapshot {
    pub root_qubits: Vec<u32>,
    pub target_qubits: Vec<u32>,
    pub blocked_count: usize,
}

/// Kernel side-channel state visible after the *current tick* of
/// `solve_with_policy` finishes applying its action batch.
///
/// **Semantics:** A single Starlark `step(...)` call can return a list of
/// actions, all of which are applied in one tick before the next tick begins.
/// `GraphDelta` reflects the kernel's `last_insert` / `last_builtin` *at the
/// end of the tick that contained the action being reported* — not the
/// per-action delta. When `step` returns multiple `InsertChild`s in one tick,
/// each `on_step` for that tick sees the same `delta.last_insert` (the id of
/// the last child inserted in the batch). Observers that need per-action
/// granularity should record actions themselves and not rely on `delta`
/// fields being action-scoped.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct GraphDelta {
    pub last_insert: Option<u64>,
    pub last_builtin: Option<String>,
}

/// Kernel observer for the Move Policy DSL `solve_with_policy` loop.
///
/// All hooks have default empty bodies. Implementors override only what
/// they need.
pub trait MoveKernelObserver {
    fn on_init(&mut self, _root: &PolicyGraphSnapshot) {}
    /// Called once per applied action. `delta` reflects tick-end kernel
    /// side-channel state (see [`GraphDelta`] for the per-tick vs per-action
    /// distinction).
    fn on_step(&mut self, _step: u64, _depth: u32, _action: &MoveAction, _delta: &GraphDelta) {}
    fn on_builtin(&mut self, _step: u64, _name: &str, _ok: bool) {}
    fn on_halt(&mut self, _status: &PolicyStatus) {}
}

/// No-op observer; the default for non-tracing callers.
pub struct NoOpMoveObserver;
impl MoveKernelObserver for NoOpMoveObserver {}

/// JSON record envelope. Schema version `v` is bumped only on incompatible
/// changes (field removals / semantic shifts); additive changes are
/// non-breaking.
#[derive(Serialize)]
struct EnvInit<'a> {
    v: u32,
    kind: &'static str,
    root: &'a PolicyGraphSnapshot,
}
#[derive(Serialize)]
struct EnvStep<'a> {
    v: u32,
    kind: &'static str,
    step: u64,
    depth: u32,
    action: &'a MoveAction,
    delta: &'a GraphDelta,
}
#[derive(Serialize)]
struct EnvBuiltin<'a> {
    v: u32,
    kind: &'static str,
    step: u64,
    name: &'a str,
    ok: bool,
}
#[derive(Serialize)]
struct EnvHalt<'a> {
    v: u32,
    kind: &'static str,
    status: &'a PolicyStatus,
}

const SCHEMA_VERSION: u32 = 1;

/// Streaming NDJSON trace observer. One record per kernel event; one line
/// per record; no trailing comma; flushes after every record so a partial
/// run still produces a parseable transcript.
pub struct JsonMoveTraceObserver<W: Write> {
    writer: W,
}

impl<W: Write> JsonMoveTraceObserver<W> {
    pub fn new(writer: W) -> Self {
        Self { writer }
    }

    fn emit<T: Serialize>(&mut self, env: &T) {
        // Best-effort emission. If the writer fails (e.g., broken pipe to
        // `head`), drop the record silently; we don't want trace I/O to
        // poison the policy run itself.
        let line = serde_json::to_string(env).expect("trace serialization");
        let _ = writeln!(self.writer, "{line}");
        let _ = self.writer.flush();
    }
}

impl<W: Write> MoveKernelObserver for JsonMoveTraceObserver<W> {
    fn on_init(&mut self, root: &PolicyGraphSnapshot) {
        self.emit(&EnvInit {
            v: SCHEMA_VERSION,
            kind: "init",
            root,
        });
    }
    fn on_step(&mut self, step: u64, depth: u32, action: &MoveAction, delta: &GraphDelta) {
        self.emit(&EnvStep {
            v: SCHEMA_VERSION,
            kind: "step",
            step,
            depth,
            action,
            delta,
        });
    }
    fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {
        self.emit(&EnvBuiltin {
            v: SCHEMA_VERSION,
            kind: "builtin",
            step,
            name,
            ok,
        });
    }
    fn on_halt(&mut self, status: &PolicyStatus) {
        self.emit(&EnvHalt {
            v: SCHEMA_VERSION,
            kind: "halt",
            status,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test double that records every call in order, used by the kernel
    /// hookup test in Step 5.
    #[derive(Default)]
    pub(crate) struct RecordingObserver {
        pub calls: Vec<String>,
    }
    impl MoveKernelObserver for RecordingObserver {
        fn on_init(&mut self, _root: &PolicyGraphSnapshot) {
            self.calls.push("init".into());
        }
        fn on_step(&mut self, step: u64, depth: u32, _a: &MoveAction, _d: &GraphDelta) {
            self.calls.push(format!("step:{step}:{depth}"));
        }
        fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {
            self.calls.push(format!("builtin:{step}:{name}:{ok}"));
        }
        fn on_halt(&mut self, _status: &PolicyStatus) {
            self.calls.push("halt".into());
        }
    }

    #[test]
    fn json_trace_observer_emits_init_step_halt_records() {
        let mut buf: Vec<u8> = Vec::new();
        {
            let mut obs = JsonMoveTraceObserver::new(&mut buf);
            obs.on_init(&PolicyGraphSnapshot {
                root_qubits: vec![0, 1],
                target_qubits: vec![0, 1],
                blocked_count: 0,
            });
            let action = MoveAction::Halt {
                status: "done".into(),
                message: String::new(),
            };
            obs.on_step(0, 1, &action, &GraphDelta::default());
            obs.on_halt(&PolicyStatus::Solved);
        }
        let s = std::str::from_utf8(&buf).unwrap();
        let lines: Vec<_> = s.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains(r#""kind":"init""#));
        assert!(lines[1].contains(r#""kind":"step""#));
        assert!(lines[1].contains(r#""depth":1"#));
        assert!(lines[2].contains(r#""kind":"halt""#));
        for ln in &lines {
            assert!(!ln.is_empty());
        }
    }

    #[test]
    fn no_op_observer_compiles_against_trait() {
        // Pure type-level check that the no-op satisfies the trait.
        fn _accept(_o: &mut dyn MoveKernelObserver) {}
        let mut obs = NoOpMoveObserver;
        _accept(&mut obs);
    }

    #[test]
    fn kernel_calls_observer_in_order_for_a_trivial_policy() {
        use std::io::Write;
        use std::sync::Arc;

        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
        use bloqade_lanes_dsl_core::sandbox::SandboxConfig;

        use crate::lane_index::LaneIndex;
        use crate::move_policy_dsl::kernel::{PolicyOptions, solve_with_policy};
        use crate::test_utils::example_arch_json;

        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = Arc::new(LaneIndex::new(spec));

        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(
            tmp,
            "def init(root, ctx): return None\n\
             def step(graph, gs, ctx, lib): return halt(\"solved\", \"done\")\n"
        )
        .expect("write");
        tmp.flush().expect("flush");
        let path = tmp.path().to_str().expect("path").to_owned();

        let opts = PolicyOptions {
            policy_path: path,
            policy_params: serde_json::Value::Null,
            max_expansions: 32,
            timeout_s: Some(1.0),
            sandbox: SandboxConfig::default(),
        };

        let mut obs = RecordingObserver::default();
        let _ = solve_with_policy(
            std::iter::empty::<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)>(),
            std::iter::empty::<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)>(),
            std::iter::empty::<bloqade_lanes_bytecode_core::arch::addr::LocationAddr>(),
            index,
            opts,
            &mut obs,
        )
        .unwrap();

        // Trivial policy: one init, then halt.
        assert_eq!(obs.calls.first().unwrap(), "init");
        assert_eq!(obs.calls.last().unwrap(), "halt");
    }

    /// Regression test: `on_step` must receive the actual search-tree depth of
    /// the just-applied node, not a hard-coded `0`.
    ///
    /// A policy is authored that:
    ///   - Stage 0: inserts one child via `insert_child(graph.root, [<lane>])`,
    ///     where `<lane>` is a valid forward site-bus lane whose src holds qubit 0.
    ///   - Stage 1: halts.
    ///
    /// The `InsertChild` action produces a genuinely new config (qubit moved from
    /// site 0 → site 5), so the child lands at depth 1.  The recording observer
    /// captures `"step:N:D"` entries; we assert D == 1 for the insert step.
    ///
    /// If a future change hard-codes depth=0 again, the `D` field will be 0 and
    /// this assertion will catch it.
    #[test]
    fn on_step_receives_real_depth_for_inserted_child() {
        use std::io::Write;
        use std::sync::Arc;

        use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
        use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
        use bloqade_lanes_dsl_core::sandbox::SandboxConfig;

        use crate::lane_index::LaneIndex;
        use crate::move_policy_dsl::kernel::{PolicyOptions, solve_with_policy};
        use crate::test_utils::{example_arch_json, lane, loc};

        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let index = Arc::new(LaneIndex::new(spec));

        // Compute the u64 encoding of the forward site-bus lane for zone 0,
        // word 0, site 0, bus 0 — this is the first lane in the example arch's
        // site bus (src=[0..4], dst=[5..9]).
        let lane_enc: u64 = lane(0, 0, 0).encode_u64();

        // Initial config: qubit 0 at site 0 of word 0.
        let initial = [(0u32, loc(0, 0))];

        let mut tmp = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(
            tmp,
            "def init(root, ctx):\n\
             \x20\x20return {{\"stage\": 0}}\n\
             \n\
             def step(graph, gs, ctx, lib):\n\
             \x20\x20if gs[\"stage\"] == 0:\n\
             \x20\x20\x20\x20return [\n\
             \x20\x20\x20\x20\x20\x20insert_child(graph.root, [{lane_enc}]),\n\
             \x20\x20\x20\x20\x20\x20update_global_state({{\"stage\": 1}}),\n\
             \x20\x20\x20\x20]\n\
             \x20\x20return halt(\"solved\", \"done\")\n",
            lane_enc = lane_enc,
        )
        .expect("write");
        tmp.flush().expect("flush");
        let path = tmp.path().to_str().expect("path").to_owned();

        let opts = PolicyOptions {
            policy_path: path,
            policy_params: serde_json::Value::Null,
            max_expansions: 32,
            timeout_s: Some(2.0),
            sandbox: SandboxConfig::default(),
        };

        let mut obs = RecordingObserver::default();
        let _ = solve_with_policy(
            initial.iter().cloned(),
            std::iter::empty::<(u32, LocationAddr)>(),
            std::iter::empty::<LocationAddr>(),
            index,
            opts,
            &mut obs,
        )
        .unwrap();

        assert!(
            obs.calls.iter().any(|c| c == "init"),
            "expected init call; got: {:?}",
            obs.calls
        );
        assert!(
            obs.calls.iter().any(|c| c == "halt"),
            "expected halt call; got: {:?}",
            obs.calls
        );

        // Find step events and verify at least one has depth > 0.
        // The insert_child in stage 0 creates a new child at depth 1, so its
        // "step:N:D" entry must have D == 1.  A hard-coded depth=0 bug would
        // produce D == 0 here and fail this assertion.
        let step_calls: Vec<&String> = obs
            .calls
            .iter()
            .filter(|c| c.starts_with("step:"))
            .collect();
        assert!(
            !step_calls.is_empty(),
            "expected at least one step event; got calls: {:?}",
            obs.calls
        );
        let depths: Vec<u32> = step_calls
            .iter()
            .map(|c| c.rsplit(':').next().unwrap().parse().unwrap())
            .collect();
        assert!(
            depths.iter().any(|&d| d > 0),
            "expected at least one step event with depth > 0 (from InsertChild at depth 1); \
             depths were {:?}; all calls: {:?}",
            depths,
            obs.calls
        );
    }
}
