//! PyO3 binding for the Target Generator DSL (Plan B of #597).
//!
//! Exposes a single `TargetPolicyRunner` class to Python. The class is
//! constructed once per (policy file, arch_spec) pair and reused across
//! many CZ stages by the higher-level Python adapter.

use std::sync::Arc;

use bloqade_lanes_dsl_core::sandbox::SandboxConfig;
use bloqade_lanes_search::lane_index::LaneIndex;
use bloqade_lanes_search::target_generator_dsl::{TargetPolicyError, TargetPolicyRunner};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::arch_python::{PyArchSpec, PyLocationAddr};

/// Reusable runner that wraps a parsed `.star` target-generator policy
/// alongside an architecture spec.
#[pyclass(name = "TargetPolicyRunner", module = "bloqade.lanes.bytecode._native")]
pub struct PyTargetPolicyRunner {
    inner: TargetPolicyRunner,
    index: Arc<LaneIndex>,
}

#[pymethods]
impl PyTargetPolicyRunner {
    #[new]
    #[pyo3(signature = (policy_path, arch_spec))]
    fn new(policy_path: &str, arch_spec: PyRef<'_, PyArchSpec>) -> PyResult<Self> {
        let cfg = SandboxConfig::default();
        let inner = TargetPolicyRunner::from_path(policy_path, &cfg).map_err(map_err)?;
        Ok(Self {
            inner,
            index: Arc::new(LaneIndex::new(arch_spec.inner.clone())),
        })
    }

    /// Run the policy's `generate(ctx, lib)` for one CZ stage.
    #[pyo3(signature = (placement, controls, targets, lookahead_cz_layers, cz_stage_index, policy_params=None))]
    fn generate<'py>(
        &self,
        py: Python<'py>,
        placement: &Bound<'py, PyDict>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
        cz_stage_index: u32,
        policy_params: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyList>> {
        let placement_pairs = pydict_to_placement(placement)?;
        let params_json = match policy_params {
            Some(d) => crate::search_python::pydict_to_json(d)?,
            None => serde_json::Value::Object(Default::default()),
        };

        let result = py
            .allow_threads(|| {
                let mut observer = bloqade_lanes_search::target_generator_dsl::NoOpTargetObserver;
                self.inner.generate(
                    self.index.clone(),
                    placement_pairs,
                    controls,
                    targets,
                    lookahead_cz_layers,
                    cz_stage_index,
                    params_json,
                    &mut observer,
                )
            })
            .map_err(map_err)?;

        let out = PyList::empty(py);
        for cand in result {
            let dict = PyDict::new(py);
            for (qid, addr) in cand {
                let py_loc = Py::new(py, PyLocationAddr { inner: addr })?;
                dict.set_item(qid, py_loc)?;
            }
            out.append(dict)?;
        }
        Ok(out)
    }
}

fn pydict_to_placement(
    d: &Bound<'_, PyDict>,
) -> PyResult<Vec<(u32, bloqade_lanes_bytecode_core::arch::addr::LocationAddr)>> {
    let mut out = Vec::with_capacity(d.len());
    for (k, v) in d.iter() {
        let qid: u32 = k.extract()?;
        let py_loc: PyRef<PyLocationAddr> = v.extract()?;
        out.push((qid, py_loc.inner));
    }
    Ok(out)
}

fn map_err(e: TargetPolicyError) -> PyErr {
    PyValueError::new_err(format!("{e}"))
}
