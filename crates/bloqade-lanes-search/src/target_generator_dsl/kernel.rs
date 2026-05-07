//! Target Generator DSL kernel.
//!
//! Loads a `.star` policy via `dsl-core`, invokes its top-level
//! `generate(ctx, lib)` function once per CZ stage, parses the returned
//! `list[dict[int, Location]]`, and validates each candidate against
//! [`crate::target_generator::validate_candidate`].
//!
//! Unlike the Move Policy DSL kernel, there is no search graph, no step
//! loop, and no transposition table — target generation is a single pure
//! function call. The runner caches the parsed/frozen policy across
//! invocations so a single `.star` file can be reused over many CZ stages.

use std::path::Path;
use std::sync::Arc;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_dsl_core::DslError;
use bloqade_lanes_dsl_core::adapter::LoadedPolicy;
use bloqade_lanes_dsl_core::primitives::StarlarkPlacement;
use bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec;
use bloqade_lanes_dsl_core::primitives::types::StarlarkLocation;
use bloqade_lanes_dsl_core::sandbox::{SandboxConfig, build_globals, make_evaluator};
use starlark::environment::Module;
use starlark::values::ValueLike;
use starlark::values::dict::DictRef;
use starlark::values::list::ListRef;
use thiserror::Error;

use crate::lane_index::LaneIndex;
use crate::target_generator::{CandidateError, validate_candidate};
use crate::target_generator_dsl::ctx_handle::StarlarkTargetContext;
use crate::target_generator_dsl::lib_target::StarlarkLibTarget;

#[derive(Debug, Error)]
pub enum TargetPolicyError {
    #[error("dsl error: {0}")]
    Dsl(#[from] DslError),
    #[error("policy missing required `generate` function: {reason}")]
    BadPolicy { reason: String },
    #[error("policy returned malformed shape: {0}")]
    ShapeError(String),
    #[error("invalid candidate from policy: {error}")]
    InvalidCandidate { error: CandidateError },
}

/// Parsed-and-frozen policy plus its sandbox config. Reusable across
/// many `generate(...)` calls.
pub struct TargetPolicyRunner {
    loaded: LoadedPolicy,
    cfg: SandboxConfig,
}

impl TargetPolicyRunner {
    /// Parse and freeze a policy file. The `cfg` is cloned and reused on
    /// every subsequent `generate(...)` call.
    pub fn from_path(
        path: impl AsRef<Path>,
        cfg: &SandboxConfig,
    ) -> Result<Self, TargetPolicyError> {
        let loaded = LoadedPolicy::from_path(path, cfg)?;
        Ok(Self {
            loaded,
            cfg: cfg.clone(),
        })
    }

    /// Parse and freeze a policy from in-memory source. Used in tests.
    pub fn from_source(
        name: String,
        src: String,
        cfg: &SandboxConfig,
    ) -> Result<Self, TargetPolicyError> {
        let loaded = LoadedPolicy::from_source(name, src, cfg)?;
        Ok(Self {
            loaded,
            cfg: cfg.clone(),
        })
    }

    /// Run the policy's `generate(ctx, lib)` and validate each candidate.
    ///
    /// Returns the candidates exactly as the policy ordered them. An empty
    /// list signals "defer entirely to fallback" — the caller (e.g.
    /// `PhysicalPlacementStrategy._build_candidates`) handles fallback.
    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        &self,
        index: Arc<LaneIndex>,
        placement: Vec<(u32, LocationAddr)>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
        cz_stage_index: u32,
        _policy_params: serde_json::Value,
        observer: &mut dyn crate::target_generator_dsl::observer::TargetKernelObserver,
    ) -> Result<Vec<Vec<(u32, LocationAddr)>>, TargetPolicyError> {
        use crate::target_generator_dsl::observer::{CandidateSummary, TargetContextSnapshot};

        let snap = TargetContextSnapshot {
            current_qubit_count: placement.len(),
            controls_len: controls.len(),
            targets_len: targets.len(),
            lookahead_layers: lookahead_cz_layers.len(),
            cz_stage_index,
        };
        observer.on_invoke(cz_stage_index as u64, &snap);

        let generate_fn =
            self.loaded
                .get("generate")
                .ok_or_else(|| TargetPolicyError::BadPolicy {
                    reason: "policy must define `def generate(ctx, lib)`".to_string(),
                })?;

        // Build per-call ctx and lib. ArchSpec is cloned once and shared
        // (cheap: it's wrapped in an Arc) between ctx and lib.
        let arch_spec_arc = Arc::new(index.arch_spec().clone());
        let arch_wrap = StarlarkArchSpec(arch_spec_arc);
        let ctx = StarlarkTargetContext::new(
            arch_wrap.clone(),
            StarlarkPlacement::from_pairs(placement),
            controls.clone(),
            targets.clone(),
            lookahead_cz_layers,
            cz_stage_index,
        );
        let lib = StarlarkLibTarget::new(arch_wrap);

        // Per-call evaluator with the same globals the policy was frozen
        // against (standard + utilities, via dsl-core's build_globals).
        let module = Module::new();
        let globals = build_globals(&self.cfg);
        let mut eval = make_evaluator(&module, &globals, &self.cfg);

        let ctx_v = module.heap().alloc(ctx);
        let lib_v = module.heap().alloc(lib);

        let result = eval
            .eval_function(generate_fn.value(), &[ctx_v, lib_v], &[])
            .map_err(|e| DslError::Runtime {
                traceback: format!("{e:?}"),
            })?;

        // Parse `list[dict[int, Location]]`.
        let outer = ListRef::from_value(result).ok_or_else(|| {
            TargetPolicyError::ShapeError(format!(
                "generate must return list[dict[int, Location]], got: {result}"
            ))
        })?;

        let mut candidates: Vec<Vec<(u32, LocationAddr)>> = Vec::with_capacity(outer.len());
        for cand_v in outer.iter() {
            let dict = DictRef::from_value(cand_v).ok_or_else(|| {
                TargetPolicyError::ShapeError(format!(
                    "candidate must be a dict[int, Location], got: {cand_v}"
                ))
            })?;
            let mut pairs: Vec<(u32, LocationAddr)> = Vec::with_capacity(dict.len());
            for (k, v) in dict.iter() {
                let qid = k.unpack_i32().ok_or_else(|| {
                    TargetPolicyError::ShapeError(format!(
                        "candidate dict key must be int, got: {k}"
                    ))
                })?;
                if qid < 0 {
                    return Err(TargetPolicyError::ShapeError(format!(
                        "candidate qid must be >= 0, got: {qid}"
                    )));
                }
                let loc_ref = v.downcast_ref::<StarlarkLocation>().ok_or_else(|| {
                    TargetPolicyError::ShapeError(format!(
                        "candidate dict value must be Location, got: {v}"
                    ))
                })?;
                pairs.push((qid as u32, loc_ref.0));
            }
            candidates.push(pairs);
        }

        // Validate each candidate against architecture-level invariants.
        let result: Result<Vec<Vec<(u32, LocationAddr)>>, TargetPolicyError> = {
            let mut validated = Vec::with_capacity(candidates.len());
            for cand in candidates {
                validate_candidate(&cand, &controls, &targets, &index)
                    .map_err(|error| TargetPolicyError::InvalidCandidate { error })?;
                validated.push(cand);
            }
            Ok(validated)
        };

        let summary = match &result {
            Ok(cands) => CandidateSummary {
                num_candidates: cands.len(),
                first_candidate_size: cands.first().map_or(0, |c| c.len()),
            },
            Err(_) => CandidateSummary {
                num_candidates: 0,
                first_candidate_size: 0,
            },
        };
        observer.on_result(cz_stage_index as u64, &summary, result.is_ok());
        result
    }
}

/// One-shot helper for callers that don't need to reuse a runner.
#[allow(clippy::too_many_arguments)]
pub fn run_target_policy(
    policy_path: impl AsRef<Path>,
    index: Arc<LaneIndex>,
    placement: Vec<(u32, LocationAddr)>,
    controls: Vec<u32>,
    targets: Vec<u32>,
    lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
    cz_stage_index: u32,
    policy_params: serde_json::Value,
    cfg: &SandboxConfig,
    observer: &mut dyn crate::target_generator_dsl::observer::TargetKernelObserver,
) -> Result<Vec<Vec<(u32, LocationAddr)>>, TargetPolicyError> {
    let runner = TargetPolicyRunner::from_path(policy_path, cfg)?;
    runner.generate(
        index,
        placement,
        controls,
        targets,
        lookahead_cz_layers,
        cz_stage_index,
        policy_params,
        observer,
    )
}

// ── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::test_utils::example_arch_json;

    fn loc(word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            zone_id: 0,
            word_id: word,
            site_id: site,
        }
    }

    fn make_index() -> Arc<LaneIndex> {
        let arch: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        Arc::new(LaneIndex::new(arch))
    }

    const DEFAULT_POLICY: &str = r#"
def generate(ctx, lib):
    target = {}
    for q in ctx.placement.qubits():
        target[q] = ctx.placement.get(q)
    for i in range(len(ctx.controls)):
        c = ctx.controls[i]
        t = ctx.targets[i]
        target[c] = lib.cz_partner(target[t])
    return [target]
"#;

    #[test]
    fn returns_one_validated_candidate_for_default_policy() {
        let cfg = SandboxConfig::default();
        let runner =
            TargetPolicyRunner::from_source("default.star".into(), DEFAULT_POLICY.into(), &cfg)
                .expect("load");
        // Word 0 ↔ word 1 are CZ partners (per example_arch_json).
        // Place qubit 0 at (0,0,0), qubit 1 at (0,1,0).
        let placement = vec![(0u32, loc(0, 0)), (1u32, loc(1, 0))];
        let candidates = runner
            .generate(
                make_index(),
                placement,
                vec![0],
                vec![1],
                vec![],
                0,
                serde_json::Value::Object(Default::default()),
                &mut crate::target_generator_dsl::NoOpTargetObserver,
            )
            .expect("generate");
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].len(), 2);
    }

    #[test]
    fn invalid_candidate_returns_validation_error() {
        // Returns the placement unchanged — qubit 0 is NOT moved to the
        // CZ partner of qubit 1's location, so the (control=0, target=1)
        // pair fails the CZ invariant.
        const BAD: &str = r#"
def generate(ctx, lib):
    bad = {}
    for q in ctx.placement.qubits():
        bad[q] = ctx.placement.get(q)
    return [bad]
"#;
        let cfg = SandboxConfig::default();
        let runner =
            TargetPolicyRunner::from_source("bad.star".into(), BAD.into(), &cfg).expect("load");
        let placement = vec![(0u32, loc(0, 0)), (1u32, loc(0, 1))];
        let err = runner
            .generate(
                make_index(),
                placement,
                vec![0],
                vec![1],
                vec![],
                0,
                serde_json::Value::Object(Default::default()),
                &mut crate::target_generator_dsl::NoOpTargetObserver,
            )
            .expect_err("must reject");
        assert!(
            matches!(err, TargetPolicyError::InvalidCandidate { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn missing_generate_function_is_bad_policy() {
        let cfg = SandboxConfig::default();
        let runner =
            TargetPolicyRunner::from_source("no_gen.star".into(), "PARAMS = {}\n".into(), &cfg)
                .expect("load");
        let err = runner
            .generate(
                make_index(),
                vec![(0u32, loc(0, 0))],
                vec![],
                vec![],
                vec![],
                0,
                serde_json::Value::Object(Default::default()),
                &mut crate::target_generator_dsl::NoOpTargetObserver,
            )
            .expect_err("must reject");
        assert!(
            matches!(err, TargetPolicyError::BadPolicy { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn empty_candidate_list_returns_no_candidates() {
        const EMPTY_POLICY: &str = r#"
def generate(ctx, lib):
    return []
"#;
        let cfg = SandboxConfig::default();
        let runner =
            TargetPolicyRunner::from_source("empty.star".into(), EMPTY_POLICY.into(), &cfg)
                .expect("load");
        let placement = vec![(0u32, loc(0, 0)), (1u32, loc(1, 0))];
        let candidates = runner
            .generate(
                make_index(),
                placement,
                vec![0],
                vec![1],
                vec![],
                0,
                serde_json::Value::Object(Default::default()),
                &mut crate::target_generator_dsl::NoOpTargetObserver,
            )
            .expect("generate");
        assert!(
            candidates.is_empty(),
            "expected empty list, got {candidates:?}"
        );
    }
}
