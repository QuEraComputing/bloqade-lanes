use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

/// Status codes returned by all fallible C-API functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlqdStatus {
    Ok = 0,
    ErrParse = 1,
    ErrDecode = 2,
    ErrValidation = 3,
    ErrIo = 4,
    ErrNullPtr = 5,
    ErrJson = 6,
}

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

pub(crate) fn set_last_error(msg: impl Into<Vec<u8>>) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg.into()).ok();
    });
}

pub(crate) fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Returns a pointer to the last error message, or NULL if no error.
/// Valid until the next C-API call on the same thread.
#[unsafe(no_mangle)]
pub extern "C" fn blqd_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match &*borrow {
            Some(cstr) => cstr.as_ptr(),
            None => std::ptr::null(),
        }
    })
}
