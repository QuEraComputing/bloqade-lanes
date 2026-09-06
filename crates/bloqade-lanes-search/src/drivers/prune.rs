//! Completion-bound pruning: the cut test and its bookkeeping, owned by one
//! type.
//!
//! [`Pruner`] answers one question for a driver — can this node still lead to
//! something strictly better than the incumbent? — and keeps the statistics
//! that answer produces. It is the logic of the entropy driver's
//! `bound_estimate`, `classify_cut` and `record_cut` restructured as methods on
//! the state they share; the entropy driver keeps its own copies for now so
//! its behaviour is untouched (migrating it is a follow-up).
//!
//! Two properties carry over verbatim:
//!
//! - With a [`TRIVIAL`](CompletionBound::TRIVIAL) bound `h` folds to `0.0` at
//!   monomorphization, nothing is memoized and nothing is counted, so the
//!   bound-disabled configuration compiles to the same code as having no
//!   bounding at all and reports an all-zero [`BoundStats`].
//! - `+∞` is the only estimate read as "infeasible"; `-∞` is a sound but
//!   vacuous lower bound and falls through to the `g + h` test, where it can
//!   never reach the cap.
//!
//! The `h` memo is indexed by the node's configuration **slot**
//! ([`SearchGraph::slot`]), not by [`NodeId`]: a cheaper rediscovery mints a
//! new id for a configuration whose estimate is already known.

use crate::bounds::{BoundStats, CompletionBound};
use crate::primitives::graph::{NodeId, SearchGraph};

/// Why a node was cut, for [`BoundStats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cut {
    /// `h = +∞`: no completion exists, whatever the incumbent.
    Infeasible,
    /// `g` alone reached the incumbent — would have been cut without a bound.
    ByG,
    /// Only the bound could make this cut: `g < C <= g + h`.
    ByH,
}

/// The incumbent gate and its statistics for one search run.
pub(crate) struct Pruner<'a, B: CompletionBound> {
    bound: &'a B,
    /// Memoized `h` by configuration slot; `None` = not yet computed. Stays
    /// empty when the bound is trivial.
    h_memo: Vec<Option<f64>>,
    /// Nodes whose cut has been folded into `stats`, by [`NodeId`], so a node
    /// tested at several gates is counted once. Stays empty when trivial.
    counted: Vec<bool>,
    stats: BoundStats,
    /// The objective's per-shot floor (C4), for the depth bookkeeping of
    /// [`Cut::ByH`].
    min_shot_cost: f64,
}

impl<'a, B: CompletionBound> Pruner<'a, B> {
    /// A pruner over `bound`. `min_shot_cost` is `Objective::min_shot_cost()`
    /// of the objective the bound was built for.
    pub(crate) fn new(bound: &'a B, min_shot_cost: f64) -> Self {
        Self {
            bound,
            h_memo: Vec::new(),
            counted: Vec::new(),
            stats: BoundStats {
                bound_enabled: !B::TRIVIAL,
                ..BoundStats::default()
            },
            min_shot_cost,
        }
    }

    /// `h` for `node`: the bound's estimate of its configuration, memoized by
    /// slot. `0.0` under a trivial bound, with no allocation.
    #[inline]
    pub(crate) fn h(&mut self, graph: &SearchGraph, node: NodeId) -> f64 {
        if B::TRIVIAL {
            return 0.0;
        }
        let slot = graph.slot(node) as usize;
        if slot >= self.h_memo.len() {
            self.h_memo.resize(slot + 1, None);
        }
        if let Some(cached) = self.h_memo[slot] {
            return cached;
        }
        let h = self.bound.estimate(graph.config(node));
        // `classify` fails both `== +∞` and `>= cap` on a `NaN`, which would
        // silently disable pruning for the node rather than err; it is not a
        // meaningful lower bound either way.
        debug_assert!(
            !h.is_nan(),
            "CompletionBound::estimate returned NaN; use 0.0 for \"no floor\" \
             or f64::INFINITY for \"infeasible\""
        );
        self.h_memo[slot] = Some(h);
        h
    }

    /// Record `h(root)` as the instance's certified lower bound. Call once,
    /// before the search, so `root_lower_bound` is a measurement rather than
    /// a default zero.
    pub(crate) fn measure_root(&mut self, graph: &SearchGraph) {
        self.stats.root_lower_bound = self.h(graph, graph.root());
    }

    /// The branch-and-bound test: `Some(reason)` when `node` cannot lead to
    /// anything strictly cheaper than `cap` (the incumbent), or when the bound
    /// proves no completion exists at all. Ties are cut. `cap == None` means
    /// no incumbent yet, so only infeasibility can cut.
    ///
    /// A cut is folded into the statistics once per node; testing the same
    /// node at several gates counts it once.
    #[inline]
    pub(crate) fn cut(
        &mut self,
        graph: &SearchGraph,
        node: NodeId,
        cap: Option<f64>,
    ) -> Option<Cut> {
        let cut = self.classify(graph, node, cap)?;
        self.record(graph, node, cut, cap);
        Some(cut)
    }

    #[inline]
    fn classify(&mut self, graph: &SearchGraph, node: NodeId, cap: Option<f64>) -> Option<Cut> {
        let h = self.h(graph, node);
        if h == f64::INFINITY {
            return Some(Cut::Infeasible);
        }
        let cap = cap?;
        let g = graph.g_score(node);
        if g >= cap {
            Some(Cut::ByG)
        } else if g + h >= cap {
            Some(Cut::ByH)
        } else {
            None
        }
    }

    /// Fold one cut into the statistics, once per node. A no-op under a
    /// trivial bound: the `g >= C` cuts still happen, but they are not the
    /// bound's work and the stats promise to stay all-zero.
    #[inline]
    fn record(&mut self, graph: &SearchGraph, node: NodeId, cut: Cut, cap: Option<f64>) {
        if B::TRIVIAL {
            return;
        }
        let idx = node.0 as usize;
        if idx >= self.counted.len() {
            self.counted.resize(idx + 1, false);
        }
        if std::mem::replace(&mut self.counted[idx], true) {
            return;
        }
        match cut {
            Cut::Infeasible => self.stats.cuts_infeasible += 1,
            Cut::ByG => self.stats.cuts_by_g += 1,
            Cut::ByH => {
                let depth = u64::from(graph.depth(node));
                let g = graph.g_score(node);
                self.stats.cuts_by_h += 1;
                self.stats.cut_depth_sum += depth;
                // How much earlier the bound fired than `g` alone would have:
                // `g` grows by at least `min_shot_cost` per shot.
                let extra = match cap {
                    Some(c) if self.min_shot_cost > 0.0 => {
                        ((c - g) / self.min_shot_cost).ceil().max(0.0) as u64
                    }
                    _ => 0,
                };
                self.stats.cut_depth_g_only_sum += depth + extra;
            }
        }
    }

    /// Record the cost of the best solution found so far.
    pub(crate) fn set_incumbent(&mut self, cost: Option<f64>) {
        self.stats.incumbent_cost = cost;
    }

    pub(crate) fn stats(&self) -> &BoundStats {
        &self.stats
    }

    pub(crate) fn into_stats(self) -> BoundStats {
        self.stats
    }

    /// Number of configurations with a memoized estimate (tests).
    #[cfg(test)]
    fn memo_len(&self) -> usize {
        self.h_memo.len()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::bounds::NoBound;
    use crate::cost::UniformCost;
    use crate::primitives::config::Config;
    use crate::primitives::graph::MoveSet;
    use crate::test_utils::{lane, loc};
    use crate::traits::{Objective, ObjectiveId};

    /// A bound with a hand-written estimate per configuration (default 0.0),
    /// counting how often it is asked.
    struct TableBound {
        table: HashMap<Config, f64>,
        calls: AtomicUsize,
    }

    impl CompletionBound for TableBound {
        type Obj = UniformCost;
        fn objective_id(&self) -> ObjectiveId {
            UniformCost.id()
        }
        fn estimate(&self, config: &Config) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.table.get(config).copied().unwrap_or(0.0)
        }
    }

    fn cfg(site: u32) -> Config {
        Config::new([(0, loc(0, site))]).unwrap()
    }

    /// Root at site 0; children at sites 1, 2, 3 with g = 1, and site 2 also
    /// rediscovered cheaper (g = 0.5) as a fourth node on the same slot.
    fn graph() -> (SearchGraph, Vec<NodeId>) {
        let mut g = SearchGraph::new(cfg(0));
        let root = g.root();
        let ids: Vec<NodeId> = (1..=3)
            .map(|s| g.insert(root, MoveSet::new([lane(0, 0, 0)]), cfg(s), 1.0).0)
            .collect();
        let (again, is_new) = g.insert(root, MoveSet::new([lane(0, 0, 1)]), cfg(2), 0.5);
        assert!(is_new);
        (g, vec![ids[0], ids[1], ids[2], again])
    }

    #[test]
    fn trivial_bound_cuts_by_g_only_and_records_nothing() {
        let (graph, ids) = graph();
        let bound = NoBound::for_objective(&UniformCost);
        let mut pruner = Pruner::new(&bound, 1.0);
        pruner.measure_root(&graph);
        assert_eq!(pruner.h(&graph, ids[0]), 0.0);
        assert_eq!(pruner.cut(&graph, ids[0], None), None);
        assert_eq!(pruner.cut(&graph, ids[0], Some(1.0)), Some(Cut::ByG));
        assert_eq!(pruner.cut(&graph, ids[0], Some(2.0)), None);
        assert_eq!(pruner.memo_len(), 0, "no memo under a trivial bound");
        assert_eq!(pruner.into_stats(), BoundStats::default());
    }

    #[test]
    fn classification_matches_the_incumbent_rules() {
        let (graph, ids) = graph();
        let bound = TableBound {
            table: HashMap::from([
                (cfg(1), 2.0),
                (cfg(2), f64::INFINITY),
                (cfg(3), f64::NEG_INFINITY),
            ]),
            calls: AtomicUsize::new(0),
        };
        let mut pruner = Pruner::new(&bound, 1.0);
        // No incumbent: only infeasibility cuts.
        assert_eq!(pruner.cut(&graph, ids[0], None), None);
        assert_eq!(pruner.cut(&graph, ids[1], None), Some(Cut::Infeasible));
        // Incumbent 2.5: node 1 has g = 1, h = 2 → g + h = 3 ≥ 2.5 → ByH.
        assert_eq!(pruner.cut(&graph, ids[0], Some(2.5)), Some(Cut::ByH));
        // g alone reaching the cap is ByG even though h would also cut.
        assert_eq!(pruner.cut(&graph, ids[0], Some(1.0)), Some(Cut::ByG));
        // -∞ is vacuous: never reaches the cap, never infeasible.
        assert_eq!(pruner.cut(&graph, ids[2], Some(100.0)), None);
        let stats = pruner.into_stats();
        assert!(stats.bound_enabled);
        assert_eq!(stats.cuts_infeasible, 1);
        // Node 1 was cut twice (ByH then ByG) but is counted once, as ByH.
        assert_eq!(stats.cuts_by_h, 1);
        assert_eq!(stats.cuts_by_g, 0);
        assert_eq!(stats.cut_depth_sum, 1);
        // depth 1 + ceil((2.5 - 1) / 1) = 1 + 2.
        assert_eq!(stats.cut_depth_g_only_sum, 3);
    }

    #[test]
    fn a_rediscovered_configuration_hits_the_memo() {
        let (graph, ids) = graph();
        let bound = TableBound {
            table: HashMap::from([(cfg(2), 4.0)]),
            calls: AtomicUsize::new(0),
        };
        let mut pruner = Pruner::new(&bound, 1.0);
        assert_eq!(pruner.h(&graph, ids[1]), 4.0);
        assert_eq!(bound.calls.load(Ordering::Relaxed), 1);
        // Same configuration, different NodeId (the cheaper rediscovery).
        assert_ne!(ids[1], ids[3]);
        assert_eq!(graph.slot(ids[1]), graph.slot(ids[3]));
        assert_eq!(pruner.h(&graph, ids[3]), 4.0);
        assert_eq!(
            bound.calls.load(Ordering::Relaxed),
            1,
            "memoized by slot, not by id"
        );
        // But a cut is counted per node.
        assert_eq!(pruner.cut(&graph, ids[1], Some(3.0)), Some(Cut::ByH));
        assert_eq!(pruner.cut(&graph, ids[3], Some(3.0)), Some(Cut::ByH));
        assert_eq!(pruner.stats().cuts_by_h, 2);
    }

    #[test]
    fn root_measurement_and_incumbent_are_reported() {
        let (graph, _) = graph();
        let bound = TableBound {
            table: HashMap::from([(cfg(0), 1.5)]),
            calls: AtomicUsize::new(0),
        };
        let mut pruner = Pruner::new(&bound, 1.0);
        pruner.measure_root(&graph);
        pruner.set_incumbent(Some(3.0));
        let stats = pruner.into_stats();
        assert_eq!(stats.root_lower_bound, 1.5);
        assert_eq!(stats.incumbent_cost, Some(3.0));
    }
}
