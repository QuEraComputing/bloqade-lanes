use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::slice;

use bloqade_lanes_bytecode_core::bytecode::program::Program;
use bloqade_lanes_bytecode_core::bytecode::text;

use super::error::{BlqdStatus, clear_last_error, set_last_error};
use super::handles::BLQDProgram;

/// Parse a BLQD binary buffer into a Program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_from_binary(
    data: *const u8,
    len: usize,
    out: *mut *mut BLQDProgram,
) -> BlqdStatus {
    clear_last_error();

    if data.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let bytes = unsafe { slice::from_raw_parts(data, len) };

    match Program::from_binary(bytes) {
        Ok(program) => {
            let handle = Box::new(BLQDProgram { inner: program });
            unsafe { *out = Box::into_raw(handle) };
            BlqdStatus::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            BlqdStatus::ErrDecode
        }
    }
}

/// Serialize a Program to BLQD binary format.
/// Caller must free the returned buffer with `blqd_free_bytes(out_data, out_len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_to_binary(
    prog: *const BLQDProgram,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> BlqdStatus {
    clear_last_error();

    if prog.is_null() || out_data.is_null() || out_len.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let bytes = prog.inner.to_binary();
    let len = bytes.len();
    let boxed = bytes.into_boxed_slice();
    let ptr = Box::into_raw(boxed) as *mut u8;

    unsafe {
        *out_data = ptr;
        *out_len = len;
    }
    BlqdStatus::Ok
}

/// Parse assembly text (null-terminated UTF-8) into a Program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_from_text(
    text_ptr: *const c_char,
    out: *mut *mut BLQDProgram,
) -> BlqdStatus {
    clear_last_error();

    if text_ptr.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let c_str = unsafe { CStr::from_ptr(text_ptr) };
    let source = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("invalid UTF-8: {}", e));
            return BlqdStatus::ErrIo;
        }
    };

    match text::parse(source) {
        Ok(program) => {
            let handle = Box::new(BLQDProgram { inner: program });
            unsafe { *out = Box::into_raw(handle) };
            BlqdStatus::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            BlqdStatus::ErrParse
        }
    }
}

/// Print a Program as assembly text.
/// Caller must free the returned string with `blqd_free_string()`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_to_text(
    prog: *const BLQDProgram,
    out_text: *mut *mut c_char,
) -> BlqdStatus {
    clear_last_error();

    if prog.is_null() || out_text.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let prog = unsafe { &*prog };
    let text_out = text::print(&prog.inner);

    match CString::new(text_out) {
        Ok(cstr) => {
            unsafe { *out_text = cstr.into_raw() };
            BlqdStatus::Ok
        }
        Err(e) => {
            set_last_error(format!("text contains null byte: {}", e));
            BlqdStatus::ErrIo
        }
    }
}

/// Query instruction count.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_instruction_count(prog: *const BLQDProgram) -> u32 {
    if prog.is_null() {
        return 0;
    }
    let prog = unsafe { &*prog };
    prog.inner.instructions.len() as u32
}

/// Query program version.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_version(
    prog: *const BLQDProgram,
    major: *mut u16,
    minor: *mut u16,
) {
    if prog.is_null() || major.is_null() || minor.is_null() {
        return;
    }
    let prog = unsafe { &*prog };
    unsafe {
        *major = prog.inner.version.major;
        *minor = prog.inner.version.minor;
    }
}

/// Free a Program handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_program_free(prog: *mut BLQDProgram) {
    if !prog.is_null() {
        drop(unsafe { Box::from_raw(prog) });
    }
}
