use pyo3::prelude::*;

mod arch_python;
pub(crate) mod errors;
mod instruction_python;
mod program_python;
pub(crate) mod validation;

#[pymodule]
#[pyo3(name = "_native")]
fn bloqade_lanes_bytecode(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Arch spec types
    m.add_class::<arch_python::PyArchSpec>()?;
    m.add_class::<arch_python::PyGeometry>()?;
    m.add_class::<arch_python::PyWord>()?;
    m.add_class::<arch_python::PyGrid>()?;
    m.add_class::<arch_python::PyBuses>()?;
    m.add_class::<arch_python::PyBus>()?;
    m.add_class::<arch_python::PyZone>()?;
    m.add_class::<arch_python::PyTransportPath>()?;

    // Address types and enums
    m.add_class::<arch_python::PyDirection>()?;
    m.add_class::<arch_python::PyMoveType>()?;
    m.add_class::<arch_python::PyLocationAddr>()?;
    m.add_class::<arch_python::PyLaneAddr>()?;
    m.add_class::<arch_python::PyZoneAddr>()?;

    // Instruction and Program
    m.add_class::<instruction_python::PyInstruction>()?;
    m.add_class::<program_python::PyProgram>()?;

    Ok(())
}
