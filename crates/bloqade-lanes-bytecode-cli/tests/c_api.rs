use std::ffi::{CStr, CString};
use std::ptr;

use bloqade_lanes_bytecode::ffi::arch::*;
use bloqade_lanes_bytecode::ffi::error::*;
use bloqade_lanes_bytecode::ffi::handles::*;
use bloqade_lanes_bytecode::ffi::memory::*;
use bloqade_lanes_bytecode::ffi::program::*;
use bloqade_lanes_bytecode::ffi::validate::*;

// --- Program round-trip tests ---

#[test]
fn text_to_binary_to_text_round_trip() {
    let source = CString::new(".version 1.0\nconst_int 42\nhalt\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();

    // Parse text
    let status = unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };
    assert_eq!(status, BlqdStatus::Ok);
    assert!(!prog.is_null());

    // Check instruction count
    let count = unsafe { blqd_program_instruction_count(prog) };
    assert_eq!(count, 2);

    // Check version
    let mut major: u16 = 0;
    let mut minor: u16 = 0;
    unsafe { blqd_program_version(prog, &mut major, &mut minor) };
    assert_eq!(major, 1);
    assert_eq!(minor, 0);

    // Serialize to binary
    let mut bin_data: *mut u8 = ptr::null_mut();
    let mut bin_len: usize = 0;
    let status = unsafe { blqd_program_to_binary(prog, &mut bin_data, &mut bin_len) };
    assert_eq!(status, BlqdStatus::Ok);
    assert!(!bin_data.is_null());
    assert!(bin_len > 0);

    // Parse binary back
    let mut prog2: *mut BLQDProgram = ptr::null_mut();
    let status = unsafe { blqd_program_from_binary(bin_data, bin_len, &mut prog2) };
    assert_eq!(status, BlqdStatus::Ok);

    // Convert back to text
    let mut text_out: *mut std::os::raw::c_char = ptr::null_mut();
    let status = unsafe { blqd_program_to_text(prog2, &mut text_out) };
    assert_eq!(status, BlqdStatus::Ok);
    assert!(!text_out.is_null());

    let text_str = unsafe { CStr::from_ptr(text_out) }.to_str().unwrap();
    assert!(text_str.contains("const_int 42"));
    assert!(text_str.contains("halt"));

    // Cleanup
    unsafe {
        blqd_free_string(text_out);
        blqd_free_bytes(bin_data, bin_len);
        blqd_program_free(prog2);
        blqd_program_free(prog);
    }
}

#[test]
fn binary_decode_known_good() {
    // Build a known binary via text parse, then decode it
    let source = CString::new(".version 1.0\nhalt\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    let status = unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };
    assert_eq!(status, BlqdStatus::Ok);

    let mut bin_data: *mut u8 = ptr::null_mut();
    let mut bin_len: usize = 0;
    unsafe { blqd_program_to_binary(prog, &mut bin_data, &mut bin_len) };

    let mut prog2: *mut BLQDProgram = ptr::null_mut();
    let status = unsafe { blqd_program_from_binary(bin_data, bin_len, &mut prog2) };
    assert_eq!(status, BlqdStatus::Ok);
    assert_eq!(unsafe { blqd_program_instruction_count(prog2) }, 1);

    unsafe {
        blqd_free_bytes(bin_data, bin_len);
        blqd_program_free(prog2);
        blqd_program_free(prog);
    }
}

// --- Null pointer tests ---

#[test]
fn null_pointer_returns_err_null_ptr() {
    let mut prog: *mut BLQDProgram = ptr::null_mut();

    // from_binary with null data
    let status = unsafe { blqd_program_from_binary(ptr::null(), 0, &mut prog) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // from_binary with null out
    let data: [u8; 1] = [0];
    let status = unsafe { blqd_program_from_binary(data.as_ptr(), 1, ptr::null_mut()) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // from_text with null text
    let status = unsafe { blqd_program_from_text(ptr::null(), &mut prog) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // to_binary with null prog
    let mut out_data: *mut u8 = ptr::null_mut();
    let mut out_len: usize = 0;
    let status = unsafe { blqd_program_to_binary(ptr::null(), &mut out_data, &mut out_len) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // to_text with null prog
    let mut text_out: *mut std::os::raw::c_char = ptr::null_mut();
    let status = unsafe { blqd_program_to_text(ptr::null(), &mut text_out) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // arch_from_json with null json
    let mut arch: *mut BLQDArchSpec = ptr::null_mut();
    let status = unsafe { blqd_arch_from_json(ptr::null(), &mut arch) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // validate_structure with null prog
    let mut errs: *mut BLQDValidationErrors = ptr::null_mut();
    let status = unsafe { blqd_validate_structure(ptr::null(), &mut errs) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // validate_addresses with null prog
    let status = unsafe { blqd_validate_addresses(ptr::null(), ptr::null(), &mut errs) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);

    // simulate_stack with null prog
    let status = unsafe { blqd_simulate_stack(ptr::null(), &mut errs) };
    assert_eq!(status, BlqdStatus::ErrNullPtr);
}

#[test]
fn instruction_count_null_returns_zero() {
    assert_eq!(unsafe { blqd_program_instruction_count(ptr::null()) }, 0);
}

#[test]
fn validation_errors_count_null_returns_zero() {
    assert_eq!(unsafe { blqd_validation_errors_count(ptr::null()) }, 0);
}

#[test]
fn validation_error_message_null_returns_null() {
    assert!(unsafe { blqd_validation_error_message(ptr::null(), 0) }.is_null());
}

// --- Error message tests ---

#[test]
fn invalid_input_sets_last_error() {
    let bad_source = CString::new("no version directive\nhalt\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();

    let status = unsafe { blqd_program_from_text(bad_source.as_ptr(), &mut prog) };
    assert_eq!(status, BlqdStatus::ErrParse);
    assert!(prog.is_null());

    let err_ptr = blqd_last_error();
    assert!(!err_ptr.is_null());
    let err_msg = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap();
    assert!(!err_msg.is_empty());
}

#[test]
fn invalid_binary_sets_last_error() {
    let bad_data: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
    let mut prog: *mut BLQDProgram = ptr::null_mut();

    let status = unsafe { blqd_program_from_binary(bad_data.as_ptr(), bad_data.len(), &mut prog) };
    assert_eq!(status, BlqdStatus::ErrDecode);

    let err_ptr = blqd_last_error();
    assert!(!err_ptr.is_null());
}

#[test]
fn successful_call_clears_last_error() {
    // First, trigger an error
    let bad_source = CString::new("invalid").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    unsafe { blqd_program_from_text(bad_source.as_ptr(), &mut prog) };
    assert!(!blqd_last_error().is_null());

    // Now make a successful call
    let good_source = CString::new(".version 1.0\nhalt\n").unwrap();
    let status = unsafe { blqd_program_from_text(good_source.as_ptr(), &mut prog) };
    assert_eq!(status, BlqdStatus::Ok);

    let err_ptr = blqd_last_error();
    assert!(err_ptr.is_null());

    unsafe { blqd_program_free(prog) };
}

// --- Validation tests ---

#[test]
fn validate_structure_valid_program() {
    let source =
        CString::new(".version 1.0\nconst_loc 0x00000000\ninitial_fill 1\nhalt\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };

    let mut errs: *mut BLQDValidationErrors = ptr::null_mut();
    let status = unsafe { blqd_validate_structure(prog, &mut errs) };
    assert_eq!(status, BlqdStatus::Ok);
    assert_eq!(unsafe { blqd_validation_errors_count(errs) }, 0);

    unsafe {
        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
    }
}

#[test]
fn validate_structure_with_errors() {
    // initial_fill after a non-constant instruction
    let source =
        CString::new(".version 1.0\nhalt\nconst_loc 0x00000000\ninitial_fill 1\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };

    let mut errs: *mut BLQDValidationErrors = ptr::null_mut();
    let status = unsafe { blqd_validate_structure(prog, &mut errs) };
    assert_eq!(status, BlqdStatus::ErrValidation);

    let count = unsafe { blqd_validation_errors_count(errs) };
    assert!(count > 0);

    // Check we can read the error message
    let msg_ptr = unsafe { blqd_validation_error_message(errs, 0) };
    assert!(!msg_ptr.is_null());
    let msg = unsafe { CStr::from_ptr(msg_ptr) }.to_str().unwrap();
    assert!(msg.contains("initial_fill"));

    // Out-of-range index returns NULL
    assert!(unsafe { blqd_validation_error_message(errs, count) }.is_null());

    unsafe {
        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
    }
}

#[test]
fn simulate_stack_valid() {
    let source =
        CString::new(".version 1.0\nconst_loc 0x00000000\ninitial_fill 1\nhalt\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };

    let mut errs: *mut BLQDValidationErrors = ptr::null_mut();
    let status = unsafe { blqd_simulate_stack(prog, &mut errs) };
    assert_eq!(status, BlqdStatus::Ok);
    assert_eq!(unsafe { blqd_validation_errors_count(errs) }, 0);

    unsafe {
        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
    }
}

#[test]
fn simulate_stack_with_errors() {
    // Pop on empty stack
    let source = CString::new(".version 1.0\npop\n").unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };

    let mut errs: *mut BLQDValidationErrors = ptr::null_mut();
    let status = unsafe { blqd_simulate_stack(prog, &mut errs) };
    assert_eq!(status, BlqdStatus::ErrValidation);
    assert!(unsafe { blqd_validation_errors_count(errs) } > 0);

    unsafe {
        blqd_validation_errors_free(errs);
        blqd_program_free(prog);
    }
}

// --- Arch spec tests ---

#[test]
fn arch_from_json_valid() {
    let json = CString::new(
        r#"{
        "version": "1.0",
        "geometry": {
            "sites_per_word": 2,
            "words": [{
                "positions": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [], "y_spacing": [2.0] },
                "site_indices": [[0, 0], [0, 1]]
            }]
        },
        "buses": { "site_buses": [], "word_buses": [] },
        "words_with_site_buses": [],
        "sites_with_word_buses": [],
        "zones": [{ "words": [0] }],
        "entangling_zones": [],
        "measurement_mode_zones": [0]
    }"#,
    )
    .unwrap();

    let mut arch: *mut BLQDArchSpec = ptr::null_mut();
    let status = unsafe { blqd_arch_from_json(json.as_ptr(), &mut arch) };
    assert_eq!(status, BlqdStatus::Ok);
    assert!(!arch.is_null());

    unsafe { blqd_arch_free(arch) };
}

#[test]
fn arch_from_json_invalid() {
    let bad_json = CString::new("not json").unwrap();
    let mut arch: *mut BLQDArchSpec = ptr::null_mut();
    let status = unsafe { blqd_arch_from_json(bad_json.as_ptr(), &mut arch) };
    assert_eq!(status, BlqdStatus::ErrJson);
    assert!(arch.is_null());

    let err_ptr = blqd_last_error();
    assert!(!err_ptr.is_null());
}

#[test]
fn validate_addresses_with_arch() {
    let source = CString::new(
        ".version 1.0\nconst_loc 0x00000000\nconst_loc 0x00000001\ninitial_fill 2\nhalt\n",
    )
    .unwrap();
    let mut prog: *mut BLQDProgram = ptr::null_mut();
    unsafe { blqd_program_from_text(source.as_ptr(), &mut prog) };

    let json = CString::new(
        r#"{
        "version": "1.0",
        "geometry": {
            "sites_per_word": 2,
            "words": [{
                "positions": { "x_start": 1.0, "y_start": 2.0, "x_spacing": [], "y_spacing": [2.0] },
                "site_indices": [[0, 0], [0, 1]]
            }]
        },
        "buses": { "site_buses": [], "word_buses": [] },
        "words_with_site_buses": [],
        "sites_with_word_buses": [],
        "zones": [{ "words": [0] }],
        "entangling_zones": [],
        "measurement_mode_zones": [0]
    }"#,
    )
    .unwrap();
    let mut arch: *mut BLQDArchSpec = ptr::null_mut();
    unsafe { blqd_arch_from_json(json.as_ptr(), &mut arch) };

    let mut errs: *mut BLQDValidationErrors = ptr::null_mut();
    let status = unsafe { blqd_validate_addresses(prog, arch, &mut errs) };
    assert_eq!(status, BlqdStatus::Ok);
    assert_eq!(unsafe { blqd_validation_errors_count(errs) }, 0);

    unsafe {
        blqd_validation_errors_free(errs);
        blqd_arch_free(arch);
        blqd_program_free(prog);
    }
}

// --- Free functions are safe on NULL ---

#[test]
fn free_null_pointers_safe() {
    unsafe {
        blqd_program_free(ptr::null_mut());
        blqd_arch_free(ptr::null_mut());
        blqd_validation_errors_free(ptr::null_mut());
        blqd_free_string(ptr::null_mut());
        blqd_free_bytes(ptr::null_mut(), 0);
    }
}
