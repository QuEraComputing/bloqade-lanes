use std::ffi::CStr;
use std::os::raw::c_char;

use bloqade_lanes_bytecode_core::arch::ArchSpec;

use super::error::{BlqdStatus, clear_last_error, set_last_error};
use super::handles::BLQDArchSpec;

/// Parse and validate an architecture spec from JSON (null-terminated UTF-8).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_arch_from_json(
    json: *const c_char,
    out: *mut *mut BLQDArchSpec,
) -> BlqdStatus {
    clear_last_error();

    if json.is_null() || out.is_null() {
        set_last_error("null pointer argument");
        return BlqdStatus::ErrNullPtr;
    }

    let c_str = unsafe { CStr::from_ptr(json) };
    let json_str = match c_str.to_str() {
        Ok(s) => s,
        Err(e) => {
            set_last_error(format!("invalid UTF-8: {}", e));
            return BlqdStatus::ErrIo;
        }
    };

    match ArchSpec::from_json(json_str) {
        Ok(spec) => {
            let handle = Box::new(BLQDArchSpec { inner: spec });
            unsafe { *out = Box::into_raw(handle) };
            BlqdStatus::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            BlqdStatus::ErrJson
        }
    }
}

/// Free an ArchSpec handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn blqd_arch_free(arch: *mut BLQDArchSpec) {
    if !arch.is_null() {
        drop(unsafe { Box::from_raw(arch) });
    }
}
