#[starlark_module]
pub fn register_graph_methods(builder: &mut GlobalsBuilder) {
    /// Stub-doc: depth of node.
    fn depth(node: i32) -> anyhow::Result<u32> { Ok(0) }
}
