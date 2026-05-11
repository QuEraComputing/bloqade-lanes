use pyo3::prelude::*;
use pyo3::types::PyBytes;

use bloqade_lanes_bytecode_core::bytecode::program as rs_prog;
use bloqade_lanes_bytecode_core::bytecode::text as rs_text;
use bloqade_lanes_bytecode_core::bytecode::validate as rs_val;
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
            inner: rs_prog::Program {
                version: Version::new(version.0, version.1),
                instructions: instructions.iter().map(|i| i.inner).collect(),
            },
        }
    }

    #[staticmethod]
    fn from_text(source: &str, py: Python<'_>) -> PyResult<Self> {
        let program =
            rs_text::parse(source).map_err(|e| crate::errors::parse_error_to_py(py, &e))?;
        Ok(Self { inner: program })
    }

    fn to_text(&self) -> String {
        rs_text::print(&self.inner)
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
    /// With `arch=spec`, also validates addresses against the architecture.
    /// With `arch=spec, stack=True`, also runs stack type simulation.
    #[pyo3(signature = (arch=None, stack=false))]
    fn validate(&self, py: Python<'_>, arch: Option<&PyArchSpec>, stack: bool) -> PyResult<()> {
        let mut all_errors = Vec::new();

        // Structural validation always runs
        all_errors.extend(rs_val::validate_structure(&self.inner));

        // Address and capability validation if arch provided
        if let Some(arch) = arch {
            all_errors.extend(rs_val::validate_arch_constraints(&self.inner, &arch.inner));
        }

        // Stack simulation if requested
        if stack {
            let arch_ref = arch.map(|a| &a.inner);
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
        (self.inner.version.major, self.inner.version.minor)
    }

    #[getter]
    fn instructions(&self) -> Vec<PyInstruction> {
        self.inner
            .instructions
            .iter()
            .map(|i| PyInstruction { inner: *i })
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
