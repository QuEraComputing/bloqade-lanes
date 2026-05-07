//! Starlark wrapper around a qubit→location map.
//!
//! Used as the `ctx.placement` view in the Target Generator DSL: a read-only,
//! dict-like handle over the current placement (qubit_id → location).
//!
//! # API adaptation notes
//!
//! - Internally a `BTreeMap<u32, LocationAddr>` so iteration order is
//!   deterministic (sorted by qubit id), matching the project-wide
//!   determinism guarantee.
//! - Allocative impl mirrors `StarlarkLocation` / `StarlarkArchSpec`:
//!   `visit_simple_sized::<Self>()`. The `BTreeMap` heap data is small and
//!   cloned per Starlark value, so treating the wrapper as a sized leaf is
//!   acceptable.
//! - `heap.alloc((a, b))` is not implemented for tuples in starlark-0.13;
//!   `items()` builds 2-element `AllocTuple(vec![…])` values, matching the
//!   pattern in `arch_spec.rs`.

use std::collections::BTreeMap;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
use starlark::starlark_module;
use starlark::values::list::AllocList;
use starlark::values::tuple::AllocTuple;
use starlark::values::{Heap, NoSerialize, ProvidesStaticType, StarlarkValue, Value};

use crate::primitives::types::StarlarkLocation;

/// Read-only Starlark handle exposing a qubit→location placement map.
///
/// Exposed surface:
/// - `len` (attr) — number of (qubit, location) pairs.
/// - `get(qid)` — returns `Location` or `None`.
/// - `qubits()` — returns `list[int]` of qubit ids in sorted order.
/// - `items()` — returns `list[(int, Location)]` in sorted order.
#[derive(Debug, Clone, ProvidesStaticType, NoSerialize)]
pub struct StarlarkPlacement {
    pub pairs: BTreeMap<u32, LocationAddr>,
}

impl StarlarkPlacement {
    /// Build a placement view from any iterator of `(qubit_id, location)` pairs.
    pub fn from_pairs(pairs: impl IntoIterator<Item = (u32, LocationAddr)>) -> Self {
        Self {
            pairs: pairs.into_iter().collect(),
        }
    }

    /// Iterate the placement entries in sorted (qubit id) order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, LocationAddr)> + '_ {
        self.pairs.iter().map(|(k, v)| (*k, *v))
    }
}

impl allocative::Allocative for StarlarkPlacement {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        visitor.visit_simple_sized::<Self>();
    }
}

starlark::starlark_simple_value!(StarlarkPlacement);

impl std::fmt::Display for StarlarkPlacement {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Placement(len={})", self.pairs.len())
    }
}

#[starlark::values::starlark_value(type = "Placement")]
impl<'v> StarlarkValue<'v> for StarlarkPlacement {
    fn get_attr(&self, attr: &str, heap: &'v Heap) -> Option<Value<'v>> {
        match attr {
            "len" => Some(heap.alloc(self.pairs.len() as i32)),
            _ => None,
        }
    }

    fn get_methods() -> Option<&'static starlark::environment::Methods> {
        static METHODS: starlark::environment::MethodsStatic =
            starlark::environment::MethodsStatic::new();
        METHODS.methods(register_placement_methods)
    }
}

#[starlark_module]
fn register_placement_methods(builder: &mut starlark::environment::MethodsBuilder) {
    /// Look up the location of `qid`. Returns `None` if the qubit is not present.
    fn get<'v>(this: &StarlarkPlacement, qid: i32, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        if qid < 0 {
            return Ok(Value::new_none());
        }
        match this.pairs.get(&(qid as u32)) {
            Some(loc) => Ok(heap.alloc(StarlarkLocation(*loc))),
            None => Ok(Value::new_none()),
        }
    }

    /// Return the qubit ids in sorted order as a list of ints.
    fn qubits<'v>(this: &StarlarkPlacement, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let ids = this.pairs.keys().map(|k| *k as i32);
        Ok(heap.alloc(AllocList(ids)))
    }

    /// Return `list[(qid, Location)]` in sorted order.
    fn items<'v>(this: &StarlarkPlacement, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        let entries: Vec<Value<'v>> = this
            .pairs
            .iter()
            .map(|(k, v)| {
                let qv = heap.alloc(*k as i32);
                let lv = heap.alloc(StarlarkLocation(*v));
                heap.alloc(AllocTuple(vec![qv, lv]))
            })
            .collect();
        Ok(heap.alloc(AllocList(entries.into_iter())))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

    use super::*;

    fn loc(zone: u32, word: u32, site: u32) -> LocationAddr {
        LocationAddr {
            zone_id: zone,
            word_id: word,
            site_id: site,
        }
    }

    #[test]
    fn from_pairs_builds_sorted_map() {
        let p = StarlarkPlacement::from_pairs(vec![(2, loc(0, 5, 0)), (0, loc(0, 1, 0))]);
        let qids: Vec<u32> = p.pairs.keys().copied().collect();
        assert_eq!(qids, vec![0, 2], "BTreeMap iteration must be sorted");
    }

    #[test]
    fn iter_yields_sorted_pairs() {
        let p = StarlarkPlacement::from_pairs(vec![(3, loc(0, 7, 1)), (1, loc(0, 2, 4))]);
        let collected: Vec<(u32, LocationAddr)> = p.iter().collect();
        assert_eq!(collected[0].0, 1);
        assert_eq!(collected[1].0, 3);
    }

    #[test]
    fn display_includes_len() {
        let p = StarlarkPlacement::from_pairs(vec![(0, loc(0, 0, 0))]);
        let s = format!("{p}");
        assert!(s.contains("len=1"), "got: {s}");
    }
}
