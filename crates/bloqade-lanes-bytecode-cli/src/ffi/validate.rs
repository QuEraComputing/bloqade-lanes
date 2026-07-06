use std::os::raw::c_char;

use bloqade_lanes_bytecode_core::isa::validate;

use super::error::{LanesStatus, clear_last_error, set_last_error};
use super::handles::{LANESArchSpec, LANESProgram, LANESValidationErrors};

/// Structural validation (arity bounds, initial_fill ordering, etc.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_validate_structure(
    prog: *const LANESProgram,
    out: *mut *mut LANESValidationErrors,
) -> LanesStatus {
    clear_last_error();

    if prog.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let errors = validate::validate_structure(&prog.inner);
    let status = if errors.is_empty() {
        LanesStatus::Ok
    } else {
        LanesStatus::ErrValidation
    };
    let handle = Box::new(LANESValidationErrors::from_errors(errors));
    unsafe { *out = Box::into_raw(handle) };
    status
}

/// Architecture-dependent validation (addresses + capability constraints).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_validate_addresses(
    prog: *const LANESProgram,
    arch: *const LANESArchSpec,
    out: *mut *mut LANESValidationErrors,
) -> LanesStatus {
    clear_last_error();

    if prog.is_null() || arch.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let arch = unsafe { &*arch };
    let errors = validate::validate(&prog.inner, Some(&arch.inner));
    let status = if errors.is_empty() {
        LanesStatus::Ok
    } else {
        LanesStatus::ErrValidation
    };
    let handle = Box::new(LANESValidationErrors::from_errors(errors));
    unsafe { *out = Box::into_raw(handle) };
    status
}

/// Stack type simulation (underflow, type mismatches, and lane/location group
/// checks). Runs without an arch spec (duplicate-only group checks).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_simulate_stack(
    prog: *const LANESProgram,
    out: *mut *mut LANESValidationErrors,
) -> LanesStatus {
    clear_last_error();

    if prog.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let errors = validate::simulate_stack(&prog.inner, None);
    let status = if errors.is_empty() {
        LanesStatus::Ok
    } else {
        LanesStatus::ErrValidation
    };
    let handle = Box::new(LANESValidationErrors::from_errors(errors));
    unsafe { *out = Box::into_raw(handle) };
    status
}

/// Number of errors in the handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_validation_errors_count(errs: *const LANESValidationErrors) -> u32 {
    if errs.is_null() {
        return 0;
    }
    let errs = unsafe { &*errs };
    errs.errors.len() as u32
}

/// Error message at index. Returns NULL if index is out of range.
/// Pointer is valid until the handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_validation_error_message(
    errs: *const LANESValidationErrors,
    index: u32,
) -> *const c_char {
    if errs.is_null() {
        return std::ptr::null();
    }
    let errs = unsafe { &*errs };
    match errs.messages.get(index as usize) {
        Some(cstr) => cstr.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Free a validation errors handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_validation_errors_free(errs: *mut LANESValidationErrors) {
    if !errs.is_null() {
        drop(unsafe { Box::from_raw(errs) });
    }
}
