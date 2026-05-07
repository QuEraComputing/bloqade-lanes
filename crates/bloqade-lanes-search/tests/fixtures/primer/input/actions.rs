// Stub registration file used by the primer-generator golden test.
#[starlark_module]
pub fn register_actions(builder: &mut GlobalsBuilder) {
    /// Stub-doc: insert child.
    fn insert_child(parent: i32, write: i32) -> anyhow::Result<()> { Ok(()) }
    /// Stub-doc: halt with reason.
    fn halt(reason: &str) -> anyhow::Result<()> { Ok(()) }
}
