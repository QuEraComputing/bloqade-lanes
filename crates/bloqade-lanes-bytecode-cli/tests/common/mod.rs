//! Helpers shared by this crate's integration tests.
//!
//! Each integration test file is its own crate, so anything two of them need
//! has to live in a module both declare with `mod common;`. Cargo treats
//! `tests/common/mod.rs` as a module rather than a test target, which is why
//! it is a directory and not `tests/common.rs`.

/// Wrap a program body in vihaco's `sst v1` section container.
///
/// The framing is eleven lines that say nothing about any test, and it was
/// pasted into every program in both `c_api.rs` and `cli_integration.rs`.
/// Behind this helper a container change is one edit.
///
/// `body` is the `.text(root)` payload — normally a whole `fn @main()` block,
/// trailing newline included.
pub fn sst(body: &str) -> String {
    sst_version("1.0", body)
}

/// [`sst`] with an explicit version, for the tests that care about it.
pub fn sst_version(version: &str, body: &str) -> String {
    format!(
        "sst v1\n\n.section(root):\n.header(root):\nversion {version}\n\
         .header(root).\n.text(root):\n{body}.text(root).\n.section(root).\n"
    )
}
