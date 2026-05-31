//! PyO3 bindings for the Move Policy DSL kernel.
//!
//! Sidecar to [`crate::search_python`]. Exposes [`PyPolicyRunner`] and
//! [`PyPolicySolveResult`] so Move Policy DSL solves never touch the
//! `PyMoveSolver`/`PySolveResult` surface — keeping the existing strategy
//! API stable and merge-friendly.
//!
//! Mirrors [`crate::target_generator_dsl_python`] for the Target Generator
//! DSL.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
use bloqade_lanes_search::LaneIndex;
use bloqade_lanes_search::dsl::move_policy_dsl::{
    NoOpMoveObserver, PolicyOptions, PolicyResult, PolicyStatus, solve_with_policy,
};

use crate::arch_python::{PyArchSpec, PyLocationAddr};

// ── JSON helpers for policy_params conversion ──

pub(crate) fn pydict_to_json(d: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let mut obj = serde_json::Map::new();
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let val = pyany_to_json(&v)?;
        obj.insert(key, val);
    }
    Ok(serde_json::Value::Object(obj))
}

fn pyany_to_json(v: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    if v.is_none() {
        Ok(serde_json::Value::Null)
    } else if let Ok(b) = v.extract::<bool>() {
        Ok(serde_json::Value::Bool(b))
    } else if let Ok(i) = v.extract::<i64>() {
        Ok(serde_json::Value::Number(serde_json::Number::from(i)))
    } else if let Ok(f) = v.extract::<f64>() {
        Ok(serde_json::json!(f))
    } else if let Ok(s) = v.extract::<String>() {
        Ok(serde_json::Value::String(s))
    } else if let Ok(list) = v.downcast::<PyList>() {
        let items: PyResult<Vec<_>> = list.iter().map(|x| pyany_to_json(&x)).collect();
        Ok(serde_json::Value::Array(items?))
    } else if let Ok(dict) = v.downcast::<PyDict>() {
        pydict_to_json(dict)
    } else {
        Err(PyValueError::new_err(format!(
            "policy_params: unsupported value type {}",
            v.get_type().name()?
        )))
    }
}

fn policy_status_str(s: &PolicyStatus) -> String {
    match s {
        PolicyStatus::Solved => "solved".to_string(),
        PolicyStatus::Unsolvable => "unsolvable".to_string(),
        PolicyStatus::BudgetExhausted => "budget_exhausted".to_string(),
        PolicyStatus::Timeout => "timeout".to_string(),
        PolicyStatus::Fallback(m) => format!("fallback: {m}"),
        PolicyStatus::SyntaxError(m) => format!("syntax_error: {m}"),
        PolicyStatus::RuntimeError(m) => format!("runtime_error: {m}"),
        PolicyStatus::SchemaError(f) => format!("schema_error: {f}"),
        PolicyStatus::BadPolicy(m) => format!("bad_policy: {m}"),
        PolicyStatus::StarlarkBudget => "starlark_budget".to_string(),
        PolicyStatus::StarlarkOOM => "starlark_oom".to_string(),
    }
}

// ── Result ──

/// Result of a Move Policy DSL solve.
///
/// Sidecar to [`crate::search_python::PySolveResult`] — keeps DSL-specific
/// fields off the strategy-based result type. `move_layers` are emitted as
/// `(direction, move_type, zone_id, word_id, site_id, bus_id)` tuples,
/// matching the merge-base lane wire format.
type LaneTuple = (u8, u8, u32, u32, u32, u32);
type MoveLayers = Vec<Vec<LaneTuple>>;

#[pyclass(
    name = "PolicySolveResult",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyPolicySolveResult {
    policy_file: String,
    policy_params_json: String,
    policy_status: String,
    is_solved: bool,
    move_layers: MoveLayers,
    goal_config: BTreeMap<u32, LocationAddr>,
    nodes_expanded: u32,
}

#[pymethods]
impl PyPolicySolveResult {
    /// Native status mirror: ``"solved"``, ``"unsolvable"``, or ``"budget_exceeded"``.
    ///
    /// Provided for compatibility with the strategy-based result; prefer
    /// ``policy_status`` to inspect the full DSL terminal state.
    #[getter]
    fn status(&self) -> &'static str {
        if self.is_solved {
            "solved"
        } else if self.policy_status == "unsolvable" {
            "unsolvable"
        } else {
            "budget_exceeded"
        }
    }

    /// Echo of the ``.star`` policy file path.
    #[getter]
    fn policy_file(&self) -> &str {
        &self.policy_file
    }

    /// JSON-encoded echo of ``policy_params`` dict.
    /// Use ``json.loads(result.policy_params)`` to recover the dict.
    #[getter]
    fn policy_params(&self) -> &str {
        &self.policy_params_json
    }

    /// Full DSL terminal status string (e.g. ``"solved"``,
    /// ``"fallback: <detail>"``, ``"runtime_error: <detail>"``).
    #[getter]
    fn policy_status(&self) -> &str {
        &self.policy_status
    }

    /// Move layers as 6-tuples: ``(direction, move_type, zone_id, word_id, site_id, bus_id)``.
    #[getter]
    fn move_layers(&self) -> MoveLayers {
        self.move_layers.clone()
    }

    /// Goal configuration: mapping of qubit_id to LocationAddress.
    #[getter]
    fn goal_config(&self) -> HashMap<u32, PyLocationAddr> {
        self.goal_config
            .iter()
            .map(|(qid, loc)| (*qid, PyLocationAddr { inner: *loc }))
            .collect()
    }

    /// Number of nodes expanded by the DSL kernel during the solve.
    #[getter]
    fn nodes_expanded(&self) -> u32 {
        self.nodes_expanded
    }

    fn __repr__(&self) -> String {
        format!(
            "PolicySolveResult(policy_status='{}', steps={}, expanded={})",
            self.policy_status,
            self.move_layers.len(),
            self.nodes_expanded,
        )
    }
}

impl PyPolicySolveResult {
    pub fn from_policy_result(p: PolicyResult) -> Self {
        let is_solved = matches!(p.status, PolicyStatus::Solved);
        let policy_status = policy_status_str(&p.status);
        let move_layers = p
            .move_layers
            .iter()
            .map(|ms| {
                ms.decode()
                    .into_iter()
                    .map(|lane| {
                        (
                            lane.direction as u8,
                            lane.move_type as u8,
                            lane.zone_id,
                            lane.word_id,
                            lane.site_id,
                            lane.bus_id,
                        )
                    })
                    .collect()
            })
            .collect();
        let goal_config: BTreeMap<u32, LocationAddr> = p.goal_config.iter().collect();
        Self {
            policy_file: p.policy_file,
            policy_params_json: serde_json::to_string(&p.policy_params).unwrap_or_default(),
            policy_status,
            is_solved,
            move_layers,
            goal_config,
            nodes_expanded: p.nodes_expanded,
        }
    }
}

// ── Runner ──

/// Reusable Move Policy DSL runner.
///
/// Sidecar to [`crate::search_python::PyMoveSolver`]. Constructed once from
/// an architecture spec (JSON string or `ArchSpec`), then call
/// [`PyPolicyRunner::solve`] for each move synthesis problem with a `.star`
/// policy.
#[pyclass(
    name = "PolicyRunner",
    frozen,
    module = "bloqade.lanes.bytecode._native"
)]
pub struct PyPolicyRunner {
    arch_spec: ArchSpec,
    index: Arc<LaneIndex>,
}

#[pymethods]
impl PyPolicyRunner {
    /// Create a runner from an ArchSpec JSON string.
    #[new]
    fn new(arch_spec_json: &str) -> PyResult<Self> {
        let arch_spec: ArchSpec = serde_json::from_str(arch_spec_json)
            .map_err(|e| PyValueError::new_err(format!("invalid arch spec JSON: {e}")))?;
        let index = Arc::new(LaneIndex::new(arch_spec.clone()));
        Ok(Self { arch_spec, index })
    }

    /// Create a runner from a native ArchSpec object.
    #[staticmethod]
    fn from_arch_spec(arch: &PyArchSpec) -> PyResult<Self> {
        let json = serde_json::to_string(&arch.inner)
            .map_err(|e| PyValueError::new_err(format!("failed to serialize arch spec: {e}")))?;
        Self::new(&json)
    }

    /// Solve a move synthesis problem using a `.star` policy.
    ///
    /// Args:
    ///     initial: Mapping of qubit_id to LocationAddress for starting positions.
    ///     target: Mapping of qubit_id to LocationAddress for desired positions.
    ///     blocked: List of LocationAddress for immovable obstacle locations.
    ///     policy_path: Path to a `.star` Move Policy DSL file.
    ///     policy_params: Optional free-form dict echoed into the result's
    ///         `policy_params` field (JSON-encoded string).
    ///     max_expansions: Optional limit on kernel node expansions
    ///         (defaults to 100_000).
    ///     timeout_s: Optional wall-clock time limit (seconds) for the kernel.
    ///
    /// Returns:
    ///     PolicySolveResult with the policy terminal state, move layers,
    ///     goal configuration, and expansion count.
    #[pyo3(signature = (initial, target, blocked, policy_path, *, policy_params=None, max_expansions=None, timeout_s=None))]
    #[allow(clippy::too_many_arguments)]
    fn solve(
        &self,
        py: Python<'_>,
        initial: BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        target: BTreeMap<u32, PyRef<'_, PyLocationAddr>>,
        blocked: Vec<PyRef<'_, PyLocationAddr>>,
        policy_path: &str,
        policy_params: Option<&Bound<'_, PyDict>>,
        max_expansions: Option<u64>,
        timeout_s: Option<f64>,
    ) -> PyResult<PyPolicySolveResult> {
        let initial_pairs: Vec<(u32, LocationAddr)> =
            initial.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let target_pairs: Vec<(u32, LocationAddr)> =
            target.iter().map(|(&qid, loc)| (qid, loc.inner)).collect();
        let blocked_locs: Vec<LocationAddr> = blocked.iter().map(|loc| loc.inner).collect();

        let params_json: serde_json::Value = match policy_params {
            Some(d) => pydict_to_json(d)?,
            None => serde_json::Value::Object(Default::default()),
        };
        let policy_opts = PolicyOptions {
            policy_path: policy_path.to_string(),
            policy_params: params_json,
            max_expansions: max_expansions.unwrap_or(100_000),
            timeout_s,
            sandbox: bloqade_lanes_dsl_core::sandbox::SandboxConfig::default(),
        };

        let index = Arc::clone(&self.index);
        let result = py
            .allow_threads(|| {
                solve_with_policy(
                    initial_pairs,
                    target_pairs,
                    blocked_locs,
                    index,
                    policy_opts,
                    &mut NoOpMoveObserver,
                )
            })
            .map_err(|e| PyValueError::new_err(format!("{e}")))?;

        Ok(PyPolicySolveResult::from_policy_result(result))
    }

    fn __repr__(&self) -> String {
        format!(
            "PolicyRunner(zones={}, words={})",
            self.arch_spec.zones.len(),
            self.arch_spec.words.len(),
        )
    }
}
