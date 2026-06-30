use pyo3::prelude::*;
use pyo3::types::PyBytes;

use bloqade_lanes_bytecode_core::version::Version;
use bloqade_lanes_bytecode_core::vihaco_isa::program as rs_prog;
use bloqade_lanes_bytecode_core::vihaco_isa::validate as rs_val;

use crate::arch_python::PyArchSpec;
use crate::instruction_python::PyInstruction;

#[pyclass(name = "Program", frozen, module = "bloqade.lanes.bytecode._native")]
#[derive(Clone)]
pub struct PyProgram {
    pub(crate) inner: rs_prog::Program,
}

#[pymethods]
impl PyProgram {
    #[new]
    fn new(version: (u16, u16), instructions: Vec<PyRef<'_, PyInstruction>>) -> Self {
        Self {
            inner: rs_prog::Program {
                version: Version::new(version.0, version.1),
                instructions: instructions.iter().map(|i| i.inner.clone()).collect(),
            },
        }
    }

    #[staticmethod]
    fn from_text(source: &str, py: Python<'_>) -> PyResult<Self> {
        let program = rs_prog::Program::parse_text(source)
            .map_err(|e| crate::errors::text_error_to_py(py, &e))?;
        Ok(Self { inner: program })
    }

    fn to_text(&self) -> String {
        self.inner.to_text()
    }

    #[staticmethod]
    fn from_binary(data: &Bound<'_, PyBytes>, py: Python<'_>) -> PyResult<Self> {
        let bytes = data.as_bytes();
        let program = rs_prog::Program::from_binary(bytes)
            .map_err(|e| crate::errors::program_error_to_py(py, &e))?;
        Ok(Self { inner: program })
    }

    fn to_binary<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = self.inner.to_binary();
        PyBytes::new(py, &bytes)
    }

    /// Validate the program.
    ///
    /// With no arguments, runs structural validation only.
    /// With `arch=spec`, also validates addresses and device-capability
    /// constraints against the architecture.
    ///
    /// `stack=True` is accepted for API compatibility but is currently a
    /// no-op: stack-type simulation has not been ported to the vihaco backend
    /// (tracked in bloqade-lanes#769).
    #[pyo3(signature = (arch=None, stack=false))]
    fn validate(&self, py: Python<'_>, arch: Option<&PyArchSpec>, stack: bool) -> PyResult<()> {
        let _ = stack; // stack-type simulation not yet ported (see #769)

        let arch_ref = arch.map(|a| &a.inner);
        let all_errors = rs_val::validate_structure(&self.inner)
            .into_iter()
            .chain(rs_val::validate(&self.inner, arch_ref))
            .collect::<Vec<_>>();

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(crate::errors::validation_errors_to_py(py, all_errors))
        }
    }

    #[getter]
    fn version(&self) -> (u16, u16) {
        (self.inner.version.major, self.inner.version.minor)
    }

    #[getter]
    fn instructions(&self) -> Vec<PyInstruction> {
        self.inner
            .instructions
            .iter()
            .map(|i| PyInstruction { inner: i.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Program(version=({}, {}), instructions={})",
            self.inner.version.major,
            self.inner.version.minor,
            self.inner.instructions.len()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.instructions.len()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
