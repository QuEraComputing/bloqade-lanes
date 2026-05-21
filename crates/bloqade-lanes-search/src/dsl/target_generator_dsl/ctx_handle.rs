//! Starlark `ctx` handle for target generator policies.
//!
//! Exposes a read-only view of `TargetContext` to a `.star` policy:
//! - `ctx.arch_spec`           — `StarlarkArchSpec`
//! - `ctx.placement`           — `StarlarkPlacement` (qubit → location map)
//! - `ctx.controls`            — `list[int]`
//! - `ctx.targets`             — `list[int]`
//! - `ctx.lookahead_cz_layers` — `list[(list[int], list[int])]`
//! - `ctx.cz_stage_index`      — `int`

use bloqade_lanes_dsl_core::primitives::StarlarkPlacement;
use bloqade_lanes_dsl_core::primitives::arch_spec::StarlarkArchSpec;
use starlark::values::list::AllocList;
use starlark::values::tuple::AllocTuple;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkTargetContext {
    pub arch_spec: StarlarkArchSpec,
    pub placement: StarlarkPlacement,
    pub controls: Vec<u32>,
    pub targets: Vec<u32>,
    pub lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
    pub cz_stage_index: u32,
}

impl StarlarkTargetContext {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        arch_spec: StarlarkArchSpec,
        placement: StarlarkPlacement,
        controls: Vec<u32>,
        targets: Vec<u32>,
        lookahead_cz_layers: Vec<(Vec<u32>, Vec<u32>)>,
        cz_stage_index: u32,
    ) -> Self {
        Self {
            arch_spec,
            placement,
            controls,
            targets,
            lookahead_cz_layers,
            cz_stage_index,
        }
    }
}

impl allocative::Allocative for StarlarkTargetContext {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

starlark::starlark_simple_value!(StarlarkTargetContext);

impl std::fmt::Display for StarlarkTargetContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TargetContext(stage={}, controls={}, targets={})",
            self.cz_stage_index,
            self.controls.len(),
            self.targets.len(),
        )
    }
}

#[starlark::values::starlark_value(type = "TargetContext")]
impl<'v> StarlarkValue<'v> for StarlarkTargetContext {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "arch_spec" => Some(heap.alloc(self.arch_spec.clone())),
            "placement" => Some(heap.alloc(self.placement.clone())),
            "controls" => Some(heap.alloc(AllocList(self.controls.iter().map(|q| *q as i32)))),
            "targets" => Some(heap.alloc(AllocList(self.targets.iter().map(|q| *q as i32)))),
            "lookahead_cz_layers" => {
                let layers: Vec<Value<'v>> = self
                    .lookahead_cz_layers
                    .iter()
                    .map(|(c, t)| {
                        let cs = heap.alloc(AllocList(c.iter().map(|q| *q as i32)));
                        let ts = heap.alloc(AllocList(t.iter().map(|q| *q as i32)));
                        heap.alloc(AllocTuple(vec![cs, ts]))
                    })
                    .collect();
                Some(heap.alloc(AllocList(layers.into_iter())))
            }
            "cz_stage_index" => Some(heap.alloc(self.cz_stage_index as i32)),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;
    use bloqade_lanes_bytecode_core::version::Version;

    use super::*;

    fn loc(zone: u32, word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            zone_id: zone,
            word_id: word,
            site_id: site,
        }
    }

    fn empty_arch() -> Arc<ArchSpec> {
        Arc::new(ArchSpec {
            version: Version::new(2, 0),
            words: vec![],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward: false,
            atom_reloading: false,
            blockade_radius: None,
        })
    }

    #[test]
    fn display_summarises_shape() {
        let ctx = StarlarkTargetContext::new(
            StarlarkArchSpec(empty_arch()),
            StarlarkPlacement::from_pairs(vec![(0, loc(0, 0, 0)), (1, loc(0, 1, 0))]),
            vec![0],
            vec![1],
            vec![],
            7,
        );
        let s = format!("{ctx}");
        assert!(s.contains("stage=7"), "got: {s}");
        assert!(s.contains("controls=1"), "got: {s}");
        assert!(s.contains("targets=1"), "got: {s}");
    }

    #[test]
    fn fields_round_trip() {
        let ctx = StarlarkTargetContext::new(
            StarlarkArchSpec(empty_arch()),
            StarlarkPlacement::from_pairs(vec![(0, loc(0, 0, 0))]),
            vec![0, 5],
            vec![1, 6],
            vec![(vec![2], vec![3])],
            42,
        );
        assert_eq!(ctx.controls, vec![0, 5]);
        assert_eq!(ctx.targets, vec![1, 6]);
        assert_eq!(ctx.lookahead_cz_layers.len(), 1);
        assert_eq!(ctx.cz_stage_index, 42);
    }
}
