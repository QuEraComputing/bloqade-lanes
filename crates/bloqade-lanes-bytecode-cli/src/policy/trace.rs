//! `trace-policy` subcommand: per-event verbose trace output.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::fixture::{self, Problem};
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::move_policy_dsl::{
    GraphDelta, JsonMoveTraceObserver, MoveKernelObserver, PolicyGraphSnapshot, PolicyOptions,
    PolicyStatus, solve_with_policy,
};
use bloqade_lanes_search::target_generator_dsl::{
    CandidateSummary, JsonTargetTraceObserver, TargetContextSnapshot, TargetKernelObserver,
    run_target_policy,
};

#[allow(clippy::too_many_arguments)]
pub fn run_trace_policy(
    policy: &Path,
    problem: &Path,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
    _seed: Option<u64>,
    out: Option<&Path>,
) -> Result<(), String> {
    let (parsed, arch_path) = fixture::load(problem).map_err(|e| format!("error: {e}"))?;
    let arch_json = std::fs::read_to_string(&arch_path)
        .map_err(|e| format!("error: reading arch {}: {e}", arch_path.display()))?;
    let arch = bloqade_lanes_bytecode_core::arch::ArchSpec::from_json(&arch_json)
        .map_err(|e| format!("error: parsing arch {}: {e}", arch_path.display()))?;

    let writer: Box<dyn Write> = match out {
        Some(p) => Box::new(BufWriter::new(
            File::create(p).map_err(|e| format!("error: writing {}: {e}", p.display()))?,
        )),
        None => Box::new(io::stdout()),
    };

    match parsed {
        Problem::Move(mp) => trace_move(
            policy,
            mp,
            arch,
            params,
            max_expansions,
            timeout_s,
            json,
            writer,
        ),
        Problem::Target(tp) => trace_target(policy, tp, arch, params, json, writer),
    }
}

fn loc_from_triple(t: &[i32; 3]) -> LocationAddr {
    LocationAddr {
        zone_id: t[0] as u32,
        word_id: t[1] as u32,
        site_id: t[2] as u32,
    }
}

#[allow(clippy::too_many_arguments)]
fn trace_move(
    policy: &Path,
    mp: bloqade_lanes_search::fixture::MoveProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    max_expansions: Option<u64>,
    timeout_s: Option<f64>,
    json: bool,
    writer: Box<dyn Write>,
) -> Result<(), String> {
    let index = Arc::new(LaneIndex::new(arch));
    let initial: Vec<_> = mp
        .initial
        .iter()
        .map(|(q, t)| (*q, loc_from_triple(t)))
        .collect();
    let target: Vec<_> = mp
        .target
        .iter()
        .map(|(q, t)| (*q, loc_from_triple(t)))
        .collect();
    let blocked: Vec<_> = mp.blocked.iter().map(loc_from_triple).collect();

    let opts = PolicyOptions {
        policy_path: policy.display().to_string(),
        sandbox: SandboxConfig::default(),
        policy_params: super::eval::load_params(params, &mp.policy_params)?,
        max_expansions: max_expansions
            .or(mp.budget.as_ref().map(|b| b.max_expansions))
            .unwrap_or(5_000),
        timeout_s: Some(
            timeout_s
                .or(mp.budget.as_ref().map(|b| b.timeout_s))
                .unwrap_or(10.0),
        ),
    };

    if json {
        let mut obs = JsonMoveTraceObserver::new(writer);
        let _ = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
            .map_err(|e| format!("error: {e}"))?;
    } else {
        let mut obs = HumanMoveTraceObserver::new(writer);
        let _ = solve_with_policy(initial, target, blocked, index, opts, &mut obs)
            .map_err(|e| format!("error: {e}"))?;
    }
    Ok(())
}

fn trace_target(
    policy: &Path,
    tp: bloqade_lanes_search::fixture::TargetProblem,
    arch: bloqade_lanes_bytecode_core::arch::ArchSpec,
    params: Option<&Path>,
    json: bool,
    writer: Box<dyn Write>,
) -> Result<(), String> {
    let index = Arc::new(LaneIndex::new(arch));
    let placement: Vec<_> = tp
        .current_placement
        .iter()
        .map(|(q, t)| (*q, loc_from_triple(t)))
        .collect();
    let cfg = SandboxConfig::default();
    let params_value = super::eval::load_params(params, &tp.policy_params)?;
    let controls = tp.controls.clone();
    let targets = tp.targets.clone();
    let lookahead = tp.lookahead_cz_layers.clone();
    let stage_idx = tp.cz_stage_index;

    if json {
        let mut obs = JsonTargetTraceObserver::new(writer);
        let _ = run_target_policy(
            policy,
            index,
            placement,
            controls,
            targets,
            lookahead,
            stage_idx,
            params_value,
            &cfg,
            &mut obs,
        );
    } else {
        let mut obs = HumanTargetTraceObserver::new(writer);
        let _ = run_target_policy(
            policy,
            index,
            placement,
            controls,
            targets,
            lookahead,
            stage_idx,
            params_value,
            &cfg,
            &mut obs,
        );
    }
    Ok(())
}

/// Human-readable observer for Move policies. One line per event,
/// terse fields, designed for interactive debugging.
struct HumanMoveTraceObserver<W: Write> {
    w: W,
}

impl<W: Write> HumanMoveTraceObserver<W> {
    fn new(w: W) -> Self {
        Self { w }
    }
}

impl<W: Write> MoveKernelObserver for HumanMoveTraceObserver<W> {
    fn on_init(&mut self, root: &PolicyGraphSnapshot) {
        let _ = writeln!(
            self.w,
            "init   qubits={} target={} blocked={}",
            root.root_qubits.len(),
            root.target_qubits.len(),
            root.blocked_count
        );
    }

    fn on_step(
        &mut self,
        step: u64,
        depth: u32,
        action: &bloqade_lanes_search::move_policy_dsl::actions::MoveAction,
        _delta: &GraphDelta,
    ) {
        let _ = writeln!(self.w, "step   #{step:04} depth={depth} action={action:?}");
    }

    fn on_builtin(&mut self, step: u64, name: &str, ok: bool) {
        let _ = writeln!(self.w, "builtin #{step:04} {name} ok={ok}");
    }

    fn on_halt(&mut self, status: &PolicyStatus) {
        let _ = writeln!(self.w, "halt   status={status:?}");
    }
}

/// Human-readable observer for Target policies.
struct HumanTargetTraceObserver<W: Write> {
    w: W,
}

impl<W: Write> HumanTargetTraceObserver<W> {
    fn new(w: W) -> Self {
        Self { w }
    }
}

impl<W: Write> TargetKernelObserver for HumanTargetTraceObserver<W> {
    fn on_invoke(&mut self, stage: u64, ctx: &TargetContextSnapshot) {
        let _ = writeln!(
            self.w,
            "invoke stage={stage} qubits={} controls={} targets={} lookahead={}",
            ctx.current_qubit_count, ctx.controls_len, ctx.targets_len, ctx.lookahead_layers
        );
    }

    fn on_result(&mut self, stage: u64, s: &CandidateSummary, ok: bool) {
        let _ = writeln!(
            self.w,
            "result stage={stage} ok={ok} candidates={} first_size={}",
            s.num_candidates, s.first_candidate_size
        );
    }
}
