#[starlark_module]
pub fn register_lib_methods(builder: &mut GlobalsBuilder) {
    /// Stub-doc: hop distance.
    fn hop_distance(qubit: u32) -> anyhow::Result<u32> { Ok(0) }
}
