use pyo3::prelude::*;
use pyo3::types::PyList;

use bloqade_lanes_bytecode_core::arch::query::{LaneGroupError, LocationGroupError};
use bloqade_lanes_bytecode_core::arch::validate::ArchSpecError;
use bloqade_lanes_bytecode_core::bytecode::program::ProgramError;
use bloqade_lanes_bytecode_core::bytecode::text::ParseError;
use bloqade_lanes_bytecode_core::bytecode::validate::ValidationError;

const EXCEPTIONS_MODULE: &str = "bloqade.lanes.bytecode.exceptions";

/// Convert a single ArchSpecError to a Python exception instance.
fn arch_spec_error_to_py(py: Python<'_>, error: &ArchSpecError) -> PyResult<PyObject> {
    let module = py.import(EXCEPTIONS_MODULE)?;

    let obj = match error {
        ArchSpecError::Zone { message } => {
            let cls = module.getattr("ArchSpecZoneError")?;
            cls.call1((message.as_str(),))?
        }
        ArchSpecError::Geometry { message } => {
            let cls = module.getattr("ArchSpecGeometryError")?;
            cls.call1((message.as_str(),))?
        }
        ArchSpecError::Bus { message } => {
            let cls = module.getattr("ArchSpecBusError")?;
            cls.call1((message.as_str(),))?
        }
        ArchSpecError::Path { message } => {
            let cls = module.getattr("ArchSpecPathError")?;
            cls.call1((message.as_str(),))?
        }
    };

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
        Ok(instance) => PyErr::from_value(instance.into()),
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
        LocationGroupError::InvalidAddress { word_id, site_id } => {
            let cls = module.getattr("InvalidLocationAddressError")?;
            cls.call1((*word_id, *site_id))?
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
        LaneGroupError::WordNotInSiteBusList { word_id } => {
            let cls = module.getattr("LaneWordNotInSiteBusListError")?;
            cls.call1((*word_id,))?
        }
        LaneGroupError::SiteNotInWordBusList { site_id } => {
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
        ValidationError::StackUnderflow { pc } => {
            let cls = module.getattr("StackUnderflowError")?;
            cls.call1((*pc,))?
        }
        ValidationError::TypeMismatch { pc, expected, got } => {
            let cls = module.getattr("TypeMismatchError")?;
            cls.call1((*pc, *expected, *got))?
        }
        ValidationError::InvalidZone { pc, zone_id } => {
            let cls = module.getattr("InvalidZoneError")?;
            cls.call1((*pc, *zone_id))?
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
        ValidationError::FeedForwardNotSupported { pc } => {
            let cls = module.getattr("FeedForwardNotSupportedError")?;
            cls.call1((*pc,))?
        }
        ValidationError::AtomReloadingNotSupported { pc } => {
            let cls = module.getattr("AtomReloadingNotSupportedError")?;
            cls.call1((*pc,))?
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
        Ok(instance) => PyErr::from_value(instance.into()),
        Err(e) => e,
    }
}

/// Convert a ParseError to a Python exception.
pub fn parse_error_to_py(py: Python<'_>, error: &ParseError) -> PyErr {
    let module = match py.import(EXCEPTIONS_MODULE) {
        Ok(m) => m,
        Err(e) => return e,
    };

    let result: PyResult<PyObject> = (|| {
        let obj = match error {
            ParseError::MissingVersion => {
                let cls = module.getattr("MissingVersionError")?;
                cls.call0()?
            }
            ParseError::InvalidVersion(msg) => {
                let cls = module.getattr("InvalidVersionError")?;
                cls.call1((msg.as_str(),))?
            }
            ParseError::UnknownMnemonic { line, mnemonic } => {
                let cls = module.getattr("UnknownMnemonicError")?;
                cls.call1((*line, mnemonic.as_str()))?
            }
            ParseError::MissingOperand { line, mnemonic } => {
                let cls = module.getattr("MissingOperandError")?;
                cls.call1((*line, mnemonic.as_str()))?
            }
            ParseError::InvalidOperand { line, message } => {
                let cls = module.getattr("InvalidOperandError")?;
                cls.call1((*line, message.as_str()))?
            }
        };
        Ok(obj.into())
    })();

    match result {
        Ok(obj) => PyErr::from_value(obj.into_bound(py)),
        Err(e) => e,
    }
}

/// Convert a ProgramError to a Python exception.
pub fn program_error_to_py(py: Python<'_>, error: &ProgramError) -> PyErr {
    let module = match py.import(EXCEPTIONS_MODULE) {
        Ok(m) => m,
        Err(e) => return e,
    };

    let result: PyResult<PyObject> = (|| {
        let obj = match error {
            ProgramError::BadMagic => {
                let cls = module.getattr("BadMagicError")?;
                cls.call0()?
            }
            ProgramError::Truncated { expected, got } => {
                let cls = module.getattr("TruncatedError")?;
                cls.call1((*expected, *got))?
            }
            ProgramError::UnknownSectionType(t) => {
                let cls = module.getattr("UnknownSectionTypeError")?;
                cls.call1((*t,))?
            }
            ProgramError::InvalidCodeSectionLength(len) => {
                let cls = module.getattr("InvalidCodeSectionLengthError")?;
                cls.call1((*len,))?
            }
            ProgramError::MissingMetadataSection => {
                let cls = module.getattr("MissingMetadataSectionError")?;
                cls.call0()?
            }
            ProgramError::MissingCodeSection => {
                let cls = module.getattr("MissingCodeSectionError")?;
                cls.call0()?
            }
            ProgramError::Decode(decode_err) => {
                let cls = module.getattr("DecodeErrorInProgram")?;
                cls.call1((decode_err.to_string(),))?
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
