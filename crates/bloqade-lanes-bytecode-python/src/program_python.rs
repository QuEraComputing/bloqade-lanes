use pyo3::prelude::*;
use pyo3::types::PyBytes;

use bloqade_lanes_bytecode_core::isa::program as rs_prog;
use bloqade_lanes_bytecode_core::isa::validate as rs_val;
use bloqade_lanes_bytecode_core::version::Version;

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
            inner: rs_prog::from_code(
                Version::new(version.0, version.1),
                instructions.iter().map(|i| i.inner.clone()).collect(),
            ),
        }
    }

    #[staticmethod]
    fn from_text(source: &str, py: Python<'_>) -> PyResult<Self> {
        let program =
            rs_prog::parse_text(source).map_err(|e| crate::errors::text_error_to_py(py, &e))?;
        Ok(Self { inner: program })
    }

    fn to_text(&self) -> String {
        rs_prog::to_text(&self.inner)
    }

    #[staticmethod]
    fn from_binary(data: &Bound<'_, PyBytes>, py: Python<'_>) -> PyResult<Self> {
        let bytes = data.as_bytes();
        let program =
            rs_prog::from_binary(bytes).map_err(|e| crate::errors::program_error_to_py(py, &e))?;
        Ok(Self { inner: program })
    }

    fn to_binary<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let bytes = rs_prog::to_binary(&self.inner);
        PyBytes::new(py, &bytes)
    }

    /// Validate the program.
    ///
    /// With no arguments, runs structural validation only.
    /// With `arch=spec`, also validates addresses and device-capability
    /// constraints against the architecture.
    /// With `stack=True`, also runs stack-type simulation (underflow, type
    /// mismatches, and lane/location group checks).
    #[pyo3(signature = (arch=None, stack=false))]
    fn validate(&self, py: Python<'_>, arch: Option<&PyArchSpec>, stack: bool) -> PyResult<()> {
        let arch_ref = arch.map(|a| &a.inner);
        let mut all_errors = rs_val::validate_structure(&self.inner)
            .into_iter()
            .chain(rs_val::validate(&self.inner, arch_ref))
            .collect::<Vec<_>>();

        if stack {
            all_errors.extend(rs_val::simulate_stack(&self.inner, arch_ref));
        }

        if all_errors.is_empty() {
            Ok(())
        } else {
            Err(crate::errors::validation_errors_to_py(py, all_errors))
        }
    }

    #[getter]
    fn version(&self) -> (u16, u16) {
        (
            self.inner.extra.version.major,
            self.inner.extra.version.minor,
        )
    }

    #[getter]
    fn instructions(&self) -> Vec<PyInstruction> {
        self.inner
            .code
            .iter()
            .map(|i| PyInstruction { inner: i.clone() })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Program(version=({}, {}), instructions={})",
            self.inner.extra.version.major,
            self.inner.extra.version.minor,
            self.inner.code.len()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.code.len()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}
