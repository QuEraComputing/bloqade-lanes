use std::ffi::CString;
use std::os::raw::c_char;

/// Free a Rust-allocated C string (from `blqd_program_to_text`, etc.)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}

/// Free a Rust-allocated byte buffer (from `blqd_program_to_binary`).
/// `len` must be the value returned by the producing function.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_free_bytes(ptr: *mut u8, len: usize) {
    if !ptr.is_null() {
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, len)) });
    }
}
