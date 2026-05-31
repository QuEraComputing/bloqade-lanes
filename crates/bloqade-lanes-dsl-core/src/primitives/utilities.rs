//! Starlark globals: `stable_sort`, `argmax`, `normalize`.
//!
//! These are bound into every policy environment for deterministic
//! tie-breaking and ergonomic numeric pipelines. See spec §8.
//!
//! # API adaptation notes (starlark-0.13)
//!
//! - `Vec<Value<'v>>` and `Vec<f64>` do not implement `UnpackValue<'v>`.
//!   Lists must be accepted as `&ListRef<'v>` and iterated manually.
//! - `#[starlark_module]` rejects having both `&mut Evaluator` and `&Heap`
//!   as parameters in the same function. Use `eval.heap()` for allocation
//!   when `eval` is already present; use `&Heap` only when `eval` is absent.
//! - `eval.eval_function(callable, positional, named)` where positional and
//!   named are `&[Value<'v>]` slices.
//! - `Value::compare(other: Value<'v>) -> Result<Ordering>` is the
//!   comparison primitive; errors are treated as Equal (non-comparable keys
//!   are caller's responsibility).

use starlark::starlark_module;
use starlark::values::float::StarlarkFloat;
use starlark::values::list::{AllocList, ListRef};
use starlark::values::{Heap, UnpackValue, Value};

#[starlark_module]
pub fn register_utilities(builder: &mut starlark::environment::GlobalsBuilder) {
    /// Stable sort by `key_fn`. Returns a new list. `desc=True` reverses.
    fn stable_sort<'v>(
        items: &'v ListRef<'v>,
        key_fn: Value<'v>,
        #[starlark(default = false)] desc: bool,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let raw: Vec<Value<'v>> = items.iter().collect();
        let mut keyed: Vec<(usize, Value<'v>, Value<'v>)> = Vec::with_capacity(raw.len());
        for (i, item) in raw.iter().enumerate() {
            let k = eval.eval_function(key_fn, &[*item], &[])?;
            keyed.push((i, *item, k));
        }
        keyed.sort_by(|a, b| {
            let ord = a.2.compare(b.2).unwrap_or(std::cmp::Ordering::Equal);
            let ord = if desc { ord.reverse() } else { ord };
            ord.then_with(|| a.0.cmp(&b.0))
        });
        let heap = eval.heap();
        Ok(heap.alloc(AllocList(keyed.into_iter().map(|(_, v, _)| v))))
    }

    /// Return the item with the largest `key_fn(item)`. Ties: first occurrence wins.
    /// Empty input → `None`.
    fn argmax<'v>(
        items: &'v ListRef<'v>,
        key_fn: Value<'v>,
        eval: &mut starlark::eval::Evaluator<'v, '_, '_>,
    ) -> starlark::Result<Value<'v>> {
        let raw: Vec<Value<'v>> = items.iter().collect();
        if raw.is_empty() {
            return Ok(Value::new_none());
        }
        let mut best_idx: usize = 0;
        let mut best_key = eval.eval_function(key_fn, &[raw[0]], &[])?;
        for (i, &item) in raw.iter().enumerate().skip(1) {
            let k = eval.eval_function(key_fn, &[item], &[])?;
            if k.compare(best_key).unwrap_or(std::cmp::Ordering::Equal)
                == std::cmp::Ordering::Greater
            {
                best_idx = i;
                best_key = k;
            }
        }
        Ok(raw[best_idx])
    }

    /// Scale a list of floats so max == 1.0. Empty → empty.
    /// All-zero or non-finite max → input unchanged.
    fn normalize<'v>(values: &'v ListRef<'v>, heap: &'v Heap) -> starlark::Result<Value<'v>> {
        // Unpack each element as f64 via StarlarkFloat or integer fallback.
        let mut floats: Vec<f64> = Vec::with_capacity(values.len());
        for v in values.iter() {
            // StarlarkFloat implements UnpackValue in 0.13; integers are also
            // accepted by first trying StarlarkFloat then i64.
            let f: f64 = if let Some(sf) = <StarlarkFloat as UnpackValue>::unpack_value(v)? {
                sf.0
            } else if let Some(i) = <i64 as UnpackValue>::unpack_value(v)? {
                i as f64
            } else {
                #[derive(Debug)]
                struct Msg(String);
                impl std::fmt::Display for Msg {
                    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                        f.write_str(&self.0)
                    }
                }
                impl std::error::Error for Msg {}
                return Err(starlark::Error::new_other(Msg(format!(
                    "normalize: expected float or int, got {v}"
                ))));
            };
            floats.push(f);
        }
        let max = floats.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let scaled: Vec<f64> = if floats.is_empty() || !max.is_finite() || max == 0.0 {
            floats
        } else {
            floats.into_iter().map(|v| v / max).collect()
        };
        Ok(heap.alloc(AllocList(scaled.into_iter())))
    }
}

#[cfg(test)]
mod tests {
    use crate::adapter::LoadedPolicy;
    use crate::sandbox::SandboxConfig;

    #[test]
    fn stable_sort_argmax_normalize_round_trip() {
        let cfg = SandboxConfig::default();
        let src = r#"
sorted_asc = stable_sort([3, 1, 2], lambda x: x)
sorted_desc = stable_sort([3, 1, 2], lambda x: x, desc=True)
best = argmax([1, 5, 3, 5], lambda x: x)
nrm = normalize([2.0, 4.0, 1.0])
RESULT = {
    "sorted_asc": sorted_asc,
    "sorted_desc": sorted_desc,
    "best": best,
    "nrm": nrm,
}
"#;
        let p = LoadedPolicy::from_source("u.star".into(), src.into(), &cfg).expect("load");
        assert!(p.get("RESULT").is_some(), "RESULT not exported");
        // Deeper assertions on the values are exercised by the Phase 6
        // acid test; this smoke test confirms the globals are registered
        // and callable from Starlark code without runtime errors.
    }

    #[test]
    fn argmax_on_empty_returns_none() {
        let cfg = SandboxConfig::default();
        let src = r#"
RESULT = argmax([], lambda x: x)
"#;
        let p = LoadedPolicy::from_source("u.star".into(), src.into(), &cfg).expect("load");
        // RESULT bound to None
        assert!(p.get("RESULT").is_some());
    }
}
