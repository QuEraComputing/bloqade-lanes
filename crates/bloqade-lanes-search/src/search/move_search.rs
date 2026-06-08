//! Composable search configuration: which algorithm, what options.
//!
//! [`MoveSearch`] is a pure value bundle (no `Arc`'d state) that pairs
//! [`SolveOptions`] with [`EntropyOptions`]. Compose with an
//! `Arc<SearchEngine>` via [`TargetSolver`](super::target_solver::TargetSolver)
//! (single fixed-target solve) or any of the
//! `placement::*CzPlacement` peers (CZ-stage placement strategies).

use crate::search::options::{EntropyOptions, InnerStrategy, SolveOptions, Strategy};

/// Composable search configuration: which algorithm + tuning knobs.
///
/// Pure value type — clone-friendly, no per-arch state. The
/// architecture-bound state lives in
/// [`SearchEngine`](super::engine::SearchEngine); compose them via
/// [`TargetSolver`](super::target_solver::TargetSolver) or the
/// `placement::*CzPlacement` peers.
#[derive(Debug, Clone, Default)]
pub struct MoveSearch {
    /// Core search-tuning knobs (strategy, weight, restarts, deadlock
    /// policy, lookahead, top_c).
    pub options: SolveOptions,
    /// Entropy-specific knobs — only consulted when the chosen
    /// strategy is [`Strategy::Entropy`] or a [`Strategy::Cascade`]
    /// whose inner phase is entropy.
    pub entropy_options: EntropyOptions,
}

impl MoveSearch {
    /// Build a `MoveSearch` from explicit option bundles.
    pub fn new(options: SolveOptions, entropy_options: EntropyOptions) -> Self {
        Self {
            options,
            entropy_options,
        }
    }

    /// Convenience: A* with the given heuristic weight (`1.0` = standard
    /// A*; `>1.0` = bounded suboptimal Weighted A*).
    pub fn astar(weight: f64) -> Self {
        Self {
            options: SolveOptions {
                strategy: Strategy::AStar,
                weight,
                ..SolveOptions::default()
            },
            entropy_options: EntropyOptions::default(),
        }
    }

    /// Convenience: entropy-guided search with default entropy params.
    pub fn entropy() -> Self {
        Self {
            options: SolveOptions {
                strategy: Strategy::Entropy,
                ..SolveOptions::default()
            },
            entropy_options: EntropyOptions::default(),
        }
    }

    /// Convenience: Iterative Diving Search.
    pub fn ids() -> Self {
        Self {
            options: SolveOptions {
                strategy: Strategy::Ids,
                ..SolveOptions::default()
            },
            entropy_options: EntropyOptions::default(),
        }
    }

    /// Convenience: cascade (fast inner strategy then bounded A*
    /// refinement).
    pub fn cascade(inner: InnerStrategy) -> Self {
        Self {
            options: SolveOptions {
                strategy: Strategy::Cascade { inner },
                ..SolveOptions::default()
            },
            entropy_options: EntropyOptions::default(),
        }
    }

    /// Set the [`SolveOptions`] bundle.
    pub fn with_options(mut self, options: SolveOptions) -> Self {
        self.options = options;
        self
    }

    /// Set the [`EntropyOptions`] bundle.
    pub fn with_entropy_options(mut self, entropy_options: EntropyOptions) -> Self {
        self.entropy_options = entropy_options;
        self
    }
}
