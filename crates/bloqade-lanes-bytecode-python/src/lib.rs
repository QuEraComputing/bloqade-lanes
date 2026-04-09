//! PyO3 bindings for the Bloqade lanes bytecode library.
//!
//! This crate exposes the core bytecode types to Python as the
//! `bloqade.lanes.bytecode._native` extension module. It wraps
//! types from [`bloqade_lanes_bytecode_core`] with Python-friendly
//! constructors, input validation, and error conversion.

use pyo3::prelude::*;

mod arch_python;
mod atom_state_python;
pub(crate) mod errors;
mod instruction_python;
mod program_python;
mod search_python;
pub(crate) mod validation;

#[pymodule]
#[pyo3(name = "_native")]
fn bloqade_lanes_bytecode(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Arch spec types
    m.add_class::<arch_python::PyArchSpec>()?;
    m.add_class::<arch_python::PyWord>()?;
    m.add_class::<arch_python::PyGrid>()?;
    m.add_class::<arch_python::PySiteBus>()?;
    m.add_class::<arch_python::PyWordBus>()?;
    m.add_class::<arch_python::PyZoneBus>()?;
    m.add_class::<arch_python::PyZone>()?;
    m.add_class::<arch_python::PyMode>()?;
    m.add_class::<arch_python::PyTransportPath>()?;

    // Address types and enums
    m.add_class::<arch_python::PyDirection>()?;
    m.add_class::<arch_python::PyMoveType>()?;
    m.add_class::<arch_python::PyLocationAddr>()?;
    m.add_class::<arch_python::PyLaneAddr>()?;
    m.add_class::<arch_python::PyZoneAddr>()?;

    // Atom state
    m.add_class::<atom_state_python::PyAtomStateData>()?;

    // Instruction and Program
    m.add_class::<instruction_python::PyInstruction>()?;
    m.add_class::<program_python::PyProgram>()?;

    // Search / move synthesis
    m.add_class::<search_python::PyMoveSolver>()?;
    m.add_class::<search_python::PySolveResult>()?;

    Ok(())
}
