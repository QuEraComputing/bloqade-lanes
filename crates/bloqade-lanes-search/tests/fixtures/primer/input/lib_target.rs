#[starlark_module]
pub fn register_lib_target_methods(builder: &mut GlobalsBuilder) {
    /// Stub-doc: cz partner.
    fn cz_partner(qubit: u32) -> anyhow::Result<u32> { Ok(0) }
}
