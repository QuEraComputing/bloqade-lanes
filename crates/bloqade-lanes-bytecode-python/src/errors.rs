use pyo3::prelude::*;
use pyo3::types::PyList;

use bloqade_lanes_bytecode_core::arch::query::{LaneGroupError, LocationGroupError};
use bloqade_lanes_bytecode_core::arch::validate::ArchSpecError;
use bloqade_lanes_bytecode_core::isa::INSTRUCTION_WIDTH;
use bloqade_lanes_bytecode_core::isa::program::BinaryError;
use bloqade_lanes_bytecode_core::isa::text::TextError;
use bloqade_lanes_bytecode_core::isa::validate::ValidationError;

const EXCEPTIONS_MODULE: &str = "bloqade.lanes.bytecode.exceptions";

/// Convert a single ArchSpecError to a Python exception instance.
fn arch_spec_error_to_py(py: Python<'_>, error: &ArchSpecError) -> PyResult<PyObject> {
    let module = py.import(EXCEPTIONS_MODULE)?;

    // All ArchSpecError variants contain a String message. Map each to the
    // most appropriate existing Python exception class.
    let (cls_name, message) = match error {
        ArchSpecError::Structure(msg) => ("ArchSpecGeometryError", msg.as_str()),
        ArchSpecError::ZoneBus(msg) => ("ArchSpecBusError", msg.as_str()),
        ArchSpecError::InterZoneBus(msg) => ("ArchSpecBusError", msg.as_str()),
        ArchSpecError::GridInvariant(msg) => ("ArchSpecGeometryError", msg.as_str()),
        ArchSpecError::EntanglingPair(msg) => ("ArchSpecZoneError", msg.as_str()),
        ArchSpecError::Mode(msg) => ("ArchSpecZoneError", msg.as_str()),
        ArchSpecError::Path(msg) => ("ArchSpecPathError", msg.as_str()),
    };

    let cls = module.getattr(cls_name)?;
    let obj = cls.call1((message,))?;
    Ok(obj.into())
}

/// Convert a Vec<ArchSpecError> to a single Python ArchSpecError with an errors list.
pub fn arch_spec_errors_to_py(py: Python<'_>, errors: Vec<ArchSpecError>) -> PyErr {
    let module = match py.import(EXCEPTIONS_MODULE) {
        Ok(m) => m,
        Err(e) => return e,
    };

    let py_errors: Vec<PyObject> = match errors
        .iter()
        .map(|e| arch_spec_error_to_py(py, e))
        .collect::<PyResult<Vec<_>>>()
    {
        Ok(v) => v,
        Err(e) => return e,
    };

    let msg = errors
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    let cls = match module.getattr("ArchSpecError") {
        Ok(c) => c,
        Err(e) => return e,
    };

    let py_errors_list = match PyList::new(py, &py_errors) {
        Ok(l) => l,
        Err(e) => return e,
    };

    match cls.call1((&msg, py_errors_list)) {
        Ok(instance) => PyErr::from_value(instance),
        Err(e) => e,
    }
}

// --- Location / Lane group error conversion ---

pub fn location_group_error_to_py(
    py: Python<'_>,
    error: &LocationGroupError,
) -> PyResult<PyObject> {
    let module = py.import(EXCEPTIONS_MODULE)?;

    let obj = match error {
        LocationGroupError::DuplicateAddress { address } => {
            let cls = module.getattr("DuplicateLocationAddressError")?;
            cls.call1((*address,))?
        }
        LocationGroupError::InvalidAddress {
            zone_id,
            word_id,
            site_id,
        } => {
            let cls = module.getattr("InvalidLocationAddressError")?;
            cls.call1((*zone_id, *word_id, *site_id))?
        }
    };

    Ok(obj.into())
}

pub fn lane_group_error_to_py(py: Python<'_>, error: &LaneGroupError) -> PyResult<PyObject> {
    let module = py.import(EXCEPTIONS_MODULE)?;

    let obj = match error {
        LaneGroupError::DuplicateAddress { address } => {
            let cls = module.getattr("DuplicateLaneAddressError")?;
            let combined = (address.0 as u64) | ((address.1 as u64) << 32);
            cls.call1((combined,))?
        }
        LaneGroupError::InvalidLane { message } => {
            let cls = module.getattr("InvalidLaneAddressError")?;
            cls.call1((message.as_str(),))?
        }
        LaneGroupError::Inconsistent { message } => {
            let cls = module.getattr("LaneGroupInconsistentError")?;
            cls.call1((message.as_str(),))?
        }
        LaneGroupError::WordNotInSiteBusList {
            zone_id: _,
            word_id,
        } => {
            let cls = module.getattr("LaneWordNotInSiteBusListError")?;
            cls.call1((*word_id,))?
        }
        LaneGroupError::SiteNotInWordBusList {
            zone_id: _,
            site_id,
        } => {
            let cls = module.getattr("LaneSiteNotInWordBusListError")?;
            cls.call1((*site_id,))?
        }
        LaneGroupError::AODConstraintViolation { message } => {
            let cls = module.getattr("LaneGroupAODConstraintViolationError")?;
            cls.call1((message.as_str(),))?
        }
    };

    Ok(obj.into())
}

/// Convert a single ValidationError to a Python exception instance.
fn validation_error_to_py(py: Python<'_>, error: &ValidationError) -> PyResult<PyObject> {
    let module = py.import(EXCEPTIONS_MODULE)?;

    let obj = match error {
        // Capability checks
        ValidationError::ControlFlowRequiresFeedForward { pc, .. }
        | ValidationError::MultipleMeasuresRequireFeedForward { pc } => {
            let cls = module.getattr("FeedForwardNotSupportedError")?;
            cls.call1((*pc,))?
        }
        ValidationError::FillRequiresAtomReloading { pc } => {
            let cls = module.getattr("AtomReloadingNotSupportedError")?;
            cls.call1((*pc,))?
        }
        // Address checks (carry the arch layer's own message)
        ValidationError::InvalidLocation { pc, message }
        | ValidationError::InvalidLane { pc, message }
        | ValidationError::InvalidZone { pc, message } => {
            let cls = module.getattr("AddressValidationError")?;
            cls.call1((*pc, message.as_str()))?
        }
        // Structural checks
        ValidationError::NewArrayZeroDim0 { pc } => {
            let cls = module.getattr("NewArrayZeroDim0Error")?;
            cls.call1((*pc,))?
        }
        ValidationError::NewArrayInvalidTypeTag { pc, type_tag } => {
            let cls = module.getattr("NewArrayInvalidTypeTagError")?;
            cls.call1((*pc, *type_tag))?
        }
        ValidationError::InitialFillNotFirst { pc } => {
            let cls = module.getattr("InitialFillNotFirstError")?;
            cls.call1((*pc,))?
        }
        ValidationError::EmptyProgram => {
            let cls = module.getattr("EmptyProgramError")?;
            cls.call0()?
        }
        ValidationError::MissingTerminator { pc } => {
            let cls = module.getattr("MissingTerminatorError")?;
            cls.call1((*pc,))?
        }
        ValidationError::UnreachableInstruction { pc } => {
            let cls = module.getattr("UnreachableInstructionError")?;
            cls.call1((*pc,))?
        }
        // Stack-type simulation
        ValidationError::StackUnderflow { pc } => {
            let cls = module.getattr("StackUnderflowError")?;
            cls.call1((*pc,))?
        }
        ValidationError::TypeMismatch { pc, expected, got } => {
            let cls = module.getattr("TypeMismatchError")?;
            cls.call1((*pc, *expected, *got))?
        }
        ValidationError::LocationGroupValidation { pc, error } => {
            let inner = location_group_error_to_py(py, error)?;
            let cls = module.getattr("LocationValidationError")?;
            cls.call1((*pc, inner))?
        }
        ValidationError::LaneGroupValidation { pc, error } => {
            let inner = lane_group_error_to_py(py, error)?;
            let cls = module.getattr("LaneValidationError")?;
            cls.call1((*pc, inner))?
        }
    };

    Ok(obj.into())
}

/// Convert a Vec<ValidationError> to a single Python ValidationError with an errors list.
pub fn validation_errors_to_py(py: Python<'_>, errors: Vec<ValidationError>) -> PyErr {
    let module = match py.import(EXCEPTIONS_MODULE) {
        Ok(m) => m,
        Err(e) => return e,
    };

    let py_errors: Vec<PyObject> = match errors
        .iter()
        .map(|e| validation_error_to_py(py, e))
        .collect::<PyResult<Vec<_>>>()
    {
        Ok(v) => v,
        Err(e) => return e,
    };

    let msg = errors
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("\n");

    let cls = match module.getattr("ValidationError") {
        Ok(c) => c,
        Err(e) => return e,
    };

    let py_errors_list = match PyList::new(py, &py_errors) {
        Ok(l) => l,
        Err(e) => return e,
    };

    match cls.call1((&msg, py_errors_list)) {
        Ok(instance) => PyErr::from_value(instance),
        Err(e) => e,
    }
}

/// Convert a TextError (`.sst` parse failure) to a Python exception.
pub fn text_error_to_py(py: Python<'_>, error: &TextError) -> PyErr {
    let module = match py.import(EXCEPTIONS_MODULE) {
        Ok(m) => m,
        Err(e) => return e,
    };

    let result: PyResult<PyObject> = (|| {
        let obj = match error {
            TextError::MissingVersion => {
                let cls = module.getattr("MissingVersionError")?;
                cls.call0()?
            }
            TextError::InvalidVersion { line, value } => {
                let cls = module.getattr("InvalidVersionError")?;
                cls.call1((format!("line {line}: '{value}'"),))?
            }
            TextError::BadInstruction { line, text } => {
                let cls = module.getattr("BadInstructionError")?;
                cls.call1((*line, text.as_str()))?
            }
        };
        Ok(obj.into())
    })();

    match result {
        Ok(obj) => PyErr::from_value(obj.into_bound(py)),
        Err(e) => e,
    }
}

/// Convert a BinaryError (program (de)serialization failure) to a Python exception.
pub fn program_error_to_py(py: Python<'_>, error: &BinaryError) -> PyErr {
    let module = match py.import(EXCEPTIONS_MODULE) {
        Ok(m) => m,
        Err(e) => return e,
    };

    let result: PyResult<PyObject> = (|| {
        let obj = match error {
            BinaryError::BadMagic => {
                let cls = module.getattr("BadMagicError")?;
                cls.call0()?
            }
            BinaryError::Truncated { expected, got } => {
                let cls = module.getattr("TruncatedError")?;
                cls.call1((*expected, *got))?
            }
            BinaryError::UnalignedCode { len } => {
                let cls = module.getattr("UnalignedCodeError")?;
                cls.call1((*len, INSTRUCTION_WIDTH as usize))?
            }
            BinaryError::Decode { pc, message } => {
                let cls = module.getattr("DecodeErrorInProgram")?;
                cls.call1((format!("pc {pc}: {message}"),))?
            }
        };
        Ok(obj.into())
    })();

    match result {
        Ok(obj) => PyErr::from_value(obj.into_bound(py)),
        Err(e) => e,
    }
}

/// Convert an ArchSpecLoadError to a Python exception.
pub fn arch_spec_load_error_to_py(
    py: Python<'_>,
    error: &bloqade_lanes_bytecode_core::arch::query::ArchSpecLoadError,
) -> PyErr {
    use bloqade_lanes_bytecode_core::arch::query::ArchSpecLoadError;
    match error {
        ArchSpecLoadError::Json(e) => {
            pyo3::exceptions::PyValueError::new_err(format!("JSON parse error: {}", e))
        }
        ArchSpecLoadError::Validation(errors) => arch_spec_errors_to_py(py, errors.clone()),
    }
}
