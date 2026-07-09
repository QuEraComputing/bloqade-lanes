use std::ffi::CString;

use bloqade_lanes_bytecode_core::arch::ArchSpec;
use bloqade_lanes_bytecode_core::isa::Program;
use bloqade_lanes_bytecode_core::isa::validate::ValidationError;

/// Opaque handle wrapping a `Program`.
pub struct LANESProgram {
    pub(crate) inner: Program,
}

/// Opaque handle wrapping an `ArchSpec`.
pub struct LANESArchSpec {
    pub(crate) inner: ArchSpec,
}

/// Opaque handle wrapping a list of validation errors with cached CString messages.
pub struct LANESValidationErrors {
    pub(crate) errors: Vec<ValidationError>,
    pub(crate) messages: Vec<CString>,
}

impl LANESValidationErrors {
    pub(crate) fn from_errors(errors: Vec<ValidationError>) -> Self {
        let messages = errors
            .iter()
            .map(|e| CString::new(e.to_string()).unwrap_or_default())
            .collect();
        Self { errors, messages }
    }
}
