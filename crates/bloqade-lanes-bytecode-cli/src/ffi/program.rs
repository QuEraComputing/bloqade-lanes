use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;

use bloqade_lanes_bytecode_core::isa::program::{from_binary, to_binary};
use bloqade_lanes_bytecode_core::isa::{parse_text, to_text};

use super::error::{LanesStatus, clear_last_error, set_last_error};
use super::handles::LANESProgram;

/// Parse a native LANES binary buffer into a Program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_from_binary(
    data: *const u8,
    len: usize,
    out: *mut *mut LANESProgram,
) -> LanesStatus {
    clear_last_error();

    if data.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let bytes = unsafe { slice::from_raw_parts(data, len) };

    match from_binary(bytes) {
        Ok(program) => {
            let handle = Box::new(LANESProgram { inner: program });
            unsafe { *out = Box::into_raw(handle) };
            LanesStatus::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            LanesStatus::ErrDecode
        }
    }
}

/// Serialize a Program to native LANES binary format.
/// Caller must free the returned buffer with `lanes_free_bytes(out_data, out_len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_to_binary(
    prog: *const LANESProgram,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> LanesStatus {
    clear_last_error();

    if prog.is_null() || out_data.is_null() || out_len.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let bytes = to_binary(&prog.inner);
    let len = bytes.len();
    let boxed = bytes.into_boxed_slice();
    let ptr = Box::into_raw(boxed) as *mut u8;

    unsafe {
        *out_data = ptr;
        *out_len = len;
    }
    LanesStatus::Ok
}

/// Parse assembly text (null-terminated UTF-8) into a Program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_from_text(
    text_ptr: *const c_char,
    out: *mut *mut LANESProgram,
) -> LanesStatus {
    clear_last_error();

    if text_ptr.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let c_str = unsafe { CStr::from_ptr(text_ptr) };
    let source = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("invalid UTF-8: {}", e));
            return LanesStatus::ErrIo;
        }
    };

    match parse_text(source) {
        Ok(program) => {
            let handle = Box::new(LANESProgram { inner: program });
            unsafe { *out = Box::into_raw(handle) };
            LanesStatus::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            LanesStatus::ErrParse
        }
    }
}

/// Print a Program as assembly text.
/// Caller must free the returned string with `lanes_free_string()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_to_text(
    prog: *const LANESProgram,
    out_text: *mut *mut c_char,
) -> LanesStatus {
    clear_last_error();

    if prog.is_null() || out_text.is_null() {
        set_last_error("null pointer argument");
        return LanesStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let text_out = to_text(&prog.inner);

    match CString::new(text_out) {
        Ok(cstr) => {
            unsafe { *out_text = cstr.into_raw() };
            LanesStatus::Ok
        }
        Err(e) => {
            set_last_error(format!("text contains null byte: {}", e));
            LanesStatus::ErrIo
        }
    }
}

/// Query instruction count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_instruction_count(prog: *const LANESProgram) -> u32 {
    if prog.is_null() {
        return 0;
    }
    let prog = unsafe { &*prog };
    prog.inner.code.len() as u32
}

/// Query program version.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_version(
    prog: *const LANESProgram,
    major: *mut u16,
    minor: *mut u16,
) {
    if prog.is_null() || major.is_null() || minor.is_null() {
        return;
    }
    let prog = unsafe { &*prog };
    unsafe {
        *major = prog.inner.extra.version.major;
        *minor = prog.inner.extra.version.minor;
    }
}

/// Free a Program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn lanes_program_free(prog: *mut LANESProgram) {
    if !prog.is_null() {
        drop(unsafe { Box::from_raw(prog) });
    }
}
