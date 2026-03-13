use std::ffi::CString;

use bloqade_lanes_bytecode_core::arch::ArchSpec;
use bloqade_lanes_bytecode_core::bytecode::program::Program;
use bloqade_lanes_bytecode_core::bytecode::validate::ValidationError;

/// Opaque handle wrapping a `Program`.
pub struct BLQDProgram {
    pub(crate) inner: Program,
}

/// Opaque handle wrapping an `ArchSpec`.
pub struct BLQDArchSpec {
    pub(crate) inner: ArchSpec,
}

/// Opaque handle wrapping a list of validation errors with cached CString messages.
pub struct BLQDValidationErrors {
    pub(crate) errors: Vec<ValidationError>,
    pub(crate) messages: Vec<CString>,
}

impl BLQDValidationErrors {
    pub(crate) fn from_errors(errors: Vec<ValidationError>) -> Self {
        let messages = errors
            .iter()
            .map(|e| CString::new(e.to_string()).unwrap_or_default())
            .collect();
        Self { errors, messages }
    }
}
