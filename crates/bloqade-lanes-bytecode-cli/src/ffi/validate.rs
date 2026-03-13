use std::os::raw::c_char;

use bloqade_lanes_bytecode_core::bytecode::validate;

use super::error::{BlqdStatus, clear_last_error, set_last_error};
use super::handles::{BLQDArchSpec, BLQDProgram, BLQDValidationErrors};

/// Structural validation (arity bounds, initial_fill ordering, etc.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_validate_structure(
    prog: *const BLQDProgram,
    out: *mut *mut BLQDValidationErrors,
) -> BlqdStatus {
    clear_last_error();

    if prog.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let errors = validate::validate_structure(&prog.inner);
    let status = if errors.is_empty() {
        BlqdStatus::Ok
    } else {
        BlqdStatus::ErrValidation
    };
    let handle = Box::new(BLQDValidationErrors::from_errors(errors));
    unsafe { *out = Box::into_raw(handle) };
    status
}

/// Address validation against an architecture spec.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_validate_addresses(
    prog: *const BLQDProgram,
    arch: *const BLQDArchSpec,
    out: *mut *mut BLQDValidationErrors,
) -> BlqdStatus {
    clear_last_error();

    if prog.is_null() || arch.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let arch = unsafe { &*arch };
    let errors = validate::validate_addresses(&prog.inner, &arch.inner);
    let status = if errors.is_empty() {
        BlqdStatus::Ok
    } else {
        BlqdStatus::ErrValidation
    };
    let handle = Box::new(BLQDValidationErrors::from_errors(errors));
    unsafe { *out = Box::into_raw(handle) };
    status
}

/// Stack type simulation.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_simulate_stack(
    prog: *const BLQDProgram,
    out: *mut *mut BLQDValidationErrors,
) -> BlqdStatus {
    clear_last_error();

    if prog.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let errors = validate::simulate_stack(&prog.inner, None);
    let status = if errors.is_empty() {
        BlqdStatus::Ok
    } else {
        BlqdStatus::ErrValidation
    };
    let handle = Box::new(BLQDValidationErrors::from_errors(errors));
    unsafe { *out = Box::into_raw(handle) };
    status
}

/// Number of errors in the handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_validation_errors_count(errs: *const BLQDValidationErrors) -> u32 {
    if errs.is_null() {
        return 0;
    }
    let errs = unsafe { &*errs };
    errs.errors.len() as u32
}

/// Error message at index. Returns NULL if index is out of range.
/// Pointer is valid until the handle is freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_validation_error_message(
    errs: *const BLQDValidationErrors,
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
pub unsafe extern "C" fn blqd_validation_errors_free(errs: *mut BLQDValidationErrors) {
    if !errs.is_null() {
        drop(unsafe { Box::from_raw(errs) });
    }
}
