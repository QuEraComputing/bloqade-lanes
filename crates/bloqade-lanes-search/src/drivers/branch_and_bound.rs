//! Branch-and-bound driver with staged widening.
//!
//! One loop, three orthogonal choices. **Branching** is a [`Schedule`] of
//! generators, one per stage; a node's next stage is revealed only when the
//! frontier is empty, shallowest first, so every node's stage 1 is tried
//! before any node's stage 2 (a hybrid of Ginsberg & Harvey's iterative
//! broadening and depth-first search). **Bounding** is a [`CompletionBound`]
//! paired with the [`Objective`] that accumulates `g`: a node is dropped when
//! `g + h >= C` against the incumbent `C`, or when `h = +∞`, at every stage
//! and for no other reason. **Ordering** is the [`Frontier`], which only ever
//! sees stage-0 nodes and receives children in the generator's order.
//!
//! Edges already in the arena stay there, so the effective out-edges of a
//! node after `s` stages are the union of everything emitted at stages
//! `<= s`: widening is an ordering decision, not a generator contract, and no
//! nesting between stages is required. With a schedule whose terminal stage
//! is the exhaustive generator at the solve's capacity, draining both the
//! frontier and the widening queue — with no stage ever withheld — proves the
//! incumbent optimal over the exhaustive search space, or proves no plan
//! exists. [`Termination`] records which.
//!
//! See the branch-and-bound design spec, *The algorithm* and *Completeness
//! and widening*.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::bounds::CompletionBound;
use crate::drivers::frontier::{Frontier, debug_assert_candidates_valid};
use crate::drivers::prune::Pruner;
use crate::drivers::result::{SearchResult, Termination};
use crate::generators::exhaustive::{ExhaustiveGenerator, SeedPolicy};
use crate::observer::{SearchEvent, SearchObserver};
use crate::primitives::config::Config;
use crate::primitives::context::{AodCapacity, MoveCandidate, SearchContext, SearchState};
use crate::primitives::graph::{NodeId, SearchGraph};
use crate::traits::{Goal, MoveGenerator, Objective};

/// The generators a run branches with, one per stage, stage 0 first.
///
/// Completeness is recorded at construction: a `&dyn MoveGenerator` cannot be
/// asked what it is, so [`Schedule::complete`] takes the terminal
/// [`ExhaustiveGenerator`] by concrete type, checks that it is the whole
/// search space at the solve's capacity, and appends it. Nothing else about
/// the list is required — not nesting, not a shared generator family —
/// because union semantics make any list correct.
///
/// The stages are plain `&dyn MoveGenerator` references: the schedule is
/// built per restart (its stage-0 generator is seeded per restart and need
/// not be `Sync`), while the exhaustive stages it points at are `Sync` and
/// shared across restarts by reference.
pub struct Schedule<'a> {
    stages: Vec<&'a dyn MoveGenerator>,
    complete: bool,
}

impl<'a> Schedule<'a> {
    /// A schedule that makes no completeness claim. Panics on an empty list.
    pub fn partial(stages: Vec<&'a dyn MoveGenerator>) -> Self {
        assert!(!stages.is_empty(), "a schedule needs at least one stage");
        Self {
            stages,
            complete: false,
        }
    }

    /// `stages` followed by `terminal`, which must be the exhaustive
    /// generator at [`SeedPolicy::Any`] with an own capacity of `None` or at
    /// least `solve_cap` componentwise — caps combine by minimum, so that is
    /// what makes the terminal stage exactly the search space at the solve's
    /// capacity. Panics otherwise.
    pub fn complete(
        mut stages: Vec<&'a dyn MoveGenerator>,
        terminal: &'a ExhaustiveGenerator,
        solve_cap: Option<AodCapacity>,
    ) -> Self {
        assert_eq!(
            terminal.seed(),
            SeedPolicy::Any,
            "the terminal stage must seed on every atom to be the whole search space"
        );
        match (terminal.capacity(), solve_cap) {
            (None, _) => {}
            (Some(own), Some(solve)) => assert!(
                own.x >= solve.x && own.y >= solve.y,
                "the terminal stage's capacity {own:?} is below the solve's {solve:?}"
            ),
            (Some(own), None) => panic!(
                "the terminal stage's capacity {own:?} is below the solve's unlimited capacity"
            ),
        }
        stages.push(terminal);
        Self {
            stages,
            complete: true,
        }
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Never true: both constructors reject an empty list.
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Whether the last stage is the whole search space at the solve's
    /// capacity, so that exhaustion is a proof.
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    fn stage(&self, s: u8) -> &'a dyn MoveGenerator {
        self.stages[s as usize]
    }
}

/// How the widening queue orders the stage entries it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WidenOrder {
    /// Stage-major, depth-minor: every node's stage 1 before any node's
    /// stage 2, shallowest first within a stage.
    #[default]
    StageThenDepth,
    /// By `g + h`: the entry with the most promising bound first.
    BestBound,
}

/// When and in what order a node's later stages are revealed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Widening {
    pub order: WidenOrder,
    /// Highest stage still processed once an incumbent exists. `0` is "stop
    /// at the first productive level"; `u8::MAX` is unlimited, which is what
    /// makes exhaustion a proof of optimality.
    pub after_incumbent: u8,
}

impl Default for Widening {
    fn default() -> Self {
        Self {
            order: WidenOrder::StageThenDepth,
            after_incumbent: 0,
        }
    }
}

impl Widening {
    /// Every stage, always: the setting under which exhaustion proves
    /// optimality.
    pub const UNLIMITED: Widening = Widening {
        order: WidenOrder::StageThenDepth,
        after_incumbent: u8::MAX,
    };
}

/// The solve-scoped, immutable half of the driver: shared by reference
/// across restarts. The objective/bound pairing is asserted once, here.
pub struct BranchAndBound<'a, O, B, Go> {
    objective: &'a O,
    bound: &'a B,
    goal: &'a Go,
    ctx: &'a SearchContext<'a>,
    max_expansions: Option<u32>,
    widening: Widening,
}

impl<'a, O, B, Go> BranchAndBound<'a, O, B, Go>
where
    O: Objective,
    B: CompletionBound<Obj = O>,
    Go: Goal,
{
    /// Panics unless `bound` was built against `objective`'s instance: a
    /// bound paired to a different instance would prune unsoundly.
    pub fn new(
        objective: &'a O,
        bound: &'a B,
        goal: &'a Go,
        ctx: &'a SearchContext<'a>,
        max_expansions: Option<u32>,
        widening: Widening,
    ) -> Self {
        assert_eq!(
            bound.objective_id(),
            objective.id(),
            "the completion bound must be built against the same objective instance the \
             driver accumulates g with"
        );
        Self {
            objective,
            bound,
            goal,
            ctx,
            max_expansions,
            widening,
        }
    }

    /// One run from `root`. `frontier` must start empty (the trait cannot
    /// check it; taking it by value rules out reuse after the run).
    /// `seed_incumbent` is a prior solution's cost, if the caller has one:
    /// the run then only reports something strictly cheaper.
    pub fn run<F, Ob>(
        &self,
        root: Config,
        schedule: &Schedule<'_>,
        mut frontier: F,
        observer: &mut Ob,
        seed_incumbent: Option<f64>,
    ) -> SearchResult
    where
        F: Frontier,
        Ob: SearchObserver,
    {
        let n_stages = schedule.len();
        let mut incumbent = Incumbent::new(seed_incumbent);

        // Root is already a goal: nothing to search, and by C4 no plan costs
        // less than the root's 0, so a wider schedule could find nothing.
        if self.goal.is_goal(&root) {
            let graph = SearchGraph::new(root);
            let mut pruner = Pruner::new(self.bound, self.objective.min_shot_cost());
            pruner.measure_root(&graph);
            let goal = incumbent.offer(0.0).then(|| graph.root());
            if goal.is_some() {
                pruner.set_incumbent(Some(0.0));
                observer.on_event(SearchEvent::GoalFound {
                    depth: 0,
                    node_id: graph.root(),
                    config: graph.config(graph.root()),
                });
            }
            return SearchResult {
                goal,
                nodes_expanded: 0,
                max_depth_reached: 0,
                graph,
                bound_stats: pruner.into_stats(),
                termination: Termination::Exhausted { proof: true },
                stage_expansions: vec![0; n_stages],
                via_stage: vec![0],
            };
        }

        let mut run = Run::new(
            root,
            self.bound,
            self.objective.min_shot_cost(),
            n_stages,
            incumbent,
        );
        run.pruner.measure_root(&run.graph);
        let root_id = run.graph.root();
        frontier.receive_children(&[root_id], &run.graph);

        let mut children: Vec<NodeId> = Vec::new();
        let mut budget_hit = false;
        loop {
            let (node, stage) = match frontier.select_next() {
                Some(n) => (n, 0u8),
                None => match run.widen.pop() {
                    Some(entry) => (entry.node, entry.stage),
                    None => break,
                },
            };
            if self
                .max_expansions
                .is_some_and(|m| run.nodes_expanded() >= m)
            {
                budget_hit = true;
                break;
            }
            // Superseded by a cheaper rediscovery: the new id starts again at
            // stage 0, this one is never expanded at its worse g.
            if !run.graph.is_current(node) {
                continue;
            }
            let idx = node.0 as usize;
            if idx >= run.stage_done.len() {
                run.stage_done.resize(idx + 1, None);
            }
            if run.stage_done[idx].is_some_and(|done| done >= stage) {
                continue;
            }
            if stage > self.widening.after_incumbent && run.incumbent.cost().is_some() {
                run.withheld = true;
                continue;
            }
            // The pop gate, at every stage.
            if run
                .pruner
                .cut(&run.graph, node, run.incumbent.cost())
                .is_some()
            {
                continue;
            }
            run.stage_done[idx] = Some(stage);
            run.stage_expansions[stage as usize] += 1;

            children.clear();
            run.expand(
                node,
                stage,
                schedule.stage(stage),
                self.objective,
                self.goal,
                self.ctx,
                observer,
                &mut children,
            );
            if !children.is_empty() {
                frontier.receive_children(&children, &run.graph);
            }
            if usize::from(stage) + 1 < n_stages {
                let depth = run.graph.depth(node);
                let key = match self.widening.order {
                    WidenOrder::StageThenDepth => 0.0,
                    WidenOrder::BestBound => {
                        run.graph.g_score(node) + run.pruner.h(&run.graph, node)
                    }
                };
                run.widen.push(WidenEntry {
                    key,
                    stage: stage + 1,
                    depth,
                    node,
                });
            }
        }

        let termination = if budget_hit {
            Termination::Budget
        } else {
            Termination::Exhausted {
                proof: schedule.is_complete() && !run.withheld,
            }
        };
        run.finish(termination)
    }
}

/// The lowest cost of any goal found so far (or handed in as a seed).
/// Ties are rejected by [`Self::offer`], which is what makes the sequence of
/// reported incumbents strictly decreasing.
struct Incumbent {
    cost: Option<f64>,
}

impl Incumbent {
    fn new(seed: Option<f64>) -> Self {
        Self { cost: seed }
    }

    fn cost(&self) -> Option<f64> {
        self.cost
    }

    /// Record `cost` if it is strictly cheaper; `true` when it improved.
    fn offer(&mut self, cost: f64) -> bool {
        match self.cost {
            Some(current) if cost >= current => false,
            _ => {
                self.cost = Some(cost);
                true
            }
        }
    }
}

/// One pending stage of one node in the widening queue. Ordered ascending on
/// `(key, stage, depth, node)`; `key` is `0.0` under
/// [`WidenOrder::StageThenDepth`] and `g + h` under [`WidenOrder::BestBound`].
/// `node` last so the order is total and stated.
struct WidenEntry {
    key: f64,
    stage: u8,
    depth: u32,
    node: NodeId,
}

impl PartialEq for WidenEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for WidenEntry {}

impl Ord for WidenEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed: `BinaryHeap` is a max-heap and the smallest entry pops.
        other
            .key
            .total_cmp(&self.key)
            .then(other.stage.cmp(&self.stage))
            .then(other.depth.cmp(&self.depth))
            .then(other.node.0.cmp(&self.node.0))
    }
}

impl PartialOrd for WidenEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Everything one run creates and discards.
struct Run<'a, B: CompletionBound> {
    graph: SearchGraph,
    /// Highest stage expanded per node; replaces a closed set.
    stage_done: Vec<Option<u8>>,
    /// The stage of the edge that reached each node, by `NodeId`.
    via_stage: Vec<u8>,
    widen: BinaryHeap<WidenEntry>,
    pruner: Pruner<'a, B>,
    incumbent: Incumbent,
    best: Option<NodeId>,
    stage_expansions: Vec<u32>,
    max_depth_seen: u32,
    /// Required by `MoveGenerator::generate`; the driver never reads it.
    state: SearchState,
    candidates: Vec<MoveCandidate>,
    /// Set at the one place the after-incumbent knob drops an entry.
    withheld: bool,
}

impl<'a, B: CompletionBound> Run<'a, B> {
    fn new(
        root: Config,
        bound: &'a B,
        min_shot_cost: f64,
        n_stages: usize,
        incumbent: Incumbent,
    ) -> Self {
        Self {
            graph: SearchGraph::new(root),
            stage_done: Vec::new(),
            via_stage: vec![0],
            widen: BinaryHeap::new(),
            pruner: Pruner::new(bound, min_shot_cost),
            incumbent,
            best: None,
            stage_expansions: vec![0; n_stages],
            max_depth_seen: 0,
            state: SearchState::default(),
            candidates: Vec::new(),
            withheld: false,
        }
    }

    fn nodes_expanded(&self) -> u32 {
        self.stage_expansions.iter().sum()
    }

    /// The one expansion seam: generate `node`'s children at `stage`, insert
    /// them, and sort each into goal / cut / frontier.
    #[allow(clippy::too_many_arguments)]
    fn expand<O: Objective, Go: Goal, Ob: SearchObserver>(
        &mut self,
        node: NodeId,
        stage: u8,
        generator: &dyn MoveGenerator,
        objective: &O,
        goal: &Go,
        ctx: &SearchContext<'_>,
        observer: &mut Ob,
        children: &mut Vec<NodeId>,
    ) {
        let depth = self.graph.depth(node);
        self.max_depth_seen = self.max_depth_seen.max(depth);

        self.candidates.clear();
        generator.generate(
            self.graph.config(node),
            node,
            ctx,
            &mut self.state,
            &mut self.candidates,
        );
        debug_assert_candidates_valid(&self.candidates, ctx);
        observer.on_event(SearchEvent::NodeExpanded {
            depth,
            num_candidates: self.candidates.len(),
            node_id: node,
            config: self.graph.config(node),
            stage,
        });

        let g = self.graph.g_score(node);
        for candidate in self.candidates.drain(..) {
            let edge = objective.edge_cost(
                &candidate.move_set,
                self.graph.config(node),
                &candidate.new_config,
            );
            debug_assert!(edge.is_finite(), "edge_cost must be finite");
            let child_g = g + edge;
            let (child, is_new) =
                self.graph
                    .insert(node, candidate.move_set, candidate.new_config, child_g);
            if !is_new {
                continue; // transposition at <= g'
            }
            debug_assert_eq!(child.0 as usize, self.via_stage.len());
            self.via_stage.push(stage);
            self.max_depth_seen = self.max_depth_seen.max(depth + 1);

            if goal.is_goal(self.graph.config(child)) {
                // Never expand a goal: its children are all costlier (C2, C4).
                if self.incumbent.offer(child_g) {
                    self.best = Some(child);
                    observer.on_event(SearchEvent::GoalFound {
                        depth: depth + 1,
                        node_id: child,
                        config: self.graph.config(child),
                    });
                }
                continue;
            }
            // The push gate.
            if self
                .pruner
                .cut(&self.graph, child, self.incumbent.cost())
                .is_some()
            {
                continue;
            }
            children.push(child);
        }
    }

    fn finish(mut self, termination: Termination) -> SearchResult {
        let nodes_expanded = self.nodes_expanded();
        self.pruner
            .set_incumbent(self.best.map(|id| self.graph.g_score(id)));
        SearchResult {
            goal: self.best,
            nodes_expanded,
            max_depth_reached: self.max_depth_seen,
            graph: self.graph,
            bound_stats: self.pruner.into_stats(),
            termination,
            stage_expansions: self.stage_expansions,
            via_stage: self.via_stage,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};

    use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;
    use bloqade_lanes_bytecode_core::arch::types::ArchSpec;

    use super::*;
    use crate::bounds::{BoundStats, NoBound, WeightedDistanceBound};
    use crate::cost::{SolveObjective, WeightedDuration};
    use crate::drivers::frontier::{
        BfsFrontier, DfsFrontier, IdsFrontier, LifoFrontier, PriorityFrontier, run_search,
    };
    use crate::generators::{DeadlockPolicy, HeuristicGenerator};
    use crate::goals::AllAtTarget;
    use crate::observer::NoOpObserver;
    use crate::primitives::distance::{DistanceTable, HopDistanceHeuristic};
    use crate::primitives::graph::MoveSet;
    use crate::primitives::lane_index::LaneIndex;
    use crate::scorers::DistanceScorer;
    use crate::search::options::ObjectiveKind;
    use crate::test_utils::{asymmetric_duration_arch_json, example_arch_json, loc};
    use crate::traits::{CostFn, ObjectiveId};

    // ── Fixture: a real arch, a fixed-target instance ──

    struct Fx {
        index: LaneIndex,
        targets: Vec<(u32, u64)>,
        target_locs: Vec<(u32, LocationAddr)>,
        dist_table: DistanceTable,
        blocked: HashSet<u64>,
    }

    impl Fx {
        fn new(json: &str, targets: &[(u32, LocationAddr)]) -> Self {
            let spec: ArchSpec = serde_json::from_str(json).unwrap();
            let index = LaneIndex::new(spec);
            let enc: Vec<(u32, u64)> = targets.iter().map(|&(q, l)| (q, l.encode())).collect();
            let locs: Vec<u64> = enc.iter().map(|&(_, l)| l).collect();
            let dist_table = DistanceTable::new(&locs, &index).with_time_distances(&index);
            Self {
                index,
                targets: enc,
                target_locs: targets.to_vec(),
                dist_table,
                blocked: HashSet::new(),
            }
        }

        fn ctx(&self) -> SearchContext<'_> {
            SearchContext {
                index: &self.index,
                dist_table: &self.dist_table,
                blocked: &self.blocked,
                targets: &self.targets,
                cz_pairs: None,
                capacity: None,
            }
        }

        fn goal(&self) -> AllAtTarget {
            AllAtTarget::new(&self.targets)
        }

        fn h_sum(&self) -> HopDistanceHeuristic<'_> {
            HopDistanceHeuristic::new(self.target_locs.iter().copied(), &self.dist_table)
        }
    }

    fn cfg(pairs: &[(u32, LocationAddr)]) -> Config {
        Config::new(pairs.iter().copied()).unwrap()
    }

    fn exhaustive(ctx: &SearchContext<'_>) -> ExhaustiveGenerator {
        ExhaustiveGenerator::for_solve(ctx, SeedPolicy::Any, None).unwrap()
    }

    /// Dijkstra over the same generator and objective: `run_search` with a
    /// zero heuristic and goal-on-pop.
    fn dijkstra<O: Objective>(
        ctx: &SearchContext<'_>,
        root: Config,
        generator: &ExhaustiveGenerator,
        objective: &O,
        goal: &AllAtTarget,
        budget: u32,
    ) -> Option<f64> {
        let mut f = PriorityFrontier::astar(|_: &Config| 0.0, 1.0);
        let r = run_search(
            root,
            generator,
            &DistanceScorer,
            objective,
            goal,
            &mut f,
            ctx,
            &mut SearchState::default(),
            &mut NoOpObserver,
            Some(budget),
            None,
            None,
        );
        r.goal.map(|id| r.graph.g_score(id))
    }

    /// Observer copying out what the tests read.
    #[derive(Default)]
    struct Recorder {
        /// `(node, stage, depth, config)` per expansion, in order.
        expanded: Vec<(NodeId, u8, u32, Config)>,
        /// `(node, index into `expanded` at the time)` per incumbent.
        goals: Vec<(NodeId, usize)>,
    }

    impl SearchObserver for Recorder {
        fn on_event(&mut self, event: SearchEvent<'_>) {
            match event {
                SearchEvent::NodeExpanded {
                    node_id,
                    stage,
                    depth,
                    config,
                    ..
                } => self.expanded.push((node_id, stage, depth, config.clone())),
                SearchEvent::GoalFound { node_id, .. } => {
                    self.goals.push((node_id, self.expanded.len()))
                }
                _ => {}
            }
        }
    }

    /// Qubit 0 walks its column's path to the far end (three shots), qubit 1
    /// takes one shot; the first shots share a rectangle, so the optimum is
    /// three and the alternatives give the bound something to cut.
    ///
    /// On the example arch the sites form five independent four-node paths
    /// `(0,i) – (0,i+5) – (1,i+5) – (1,i)`: an atom never leaves its column
    /// and two atoms in one column cannot pass each other, which every
    /// hand-picked instance here respects.
    fn two_atom() -> Fx {
        Fx::new(example_arch_json(), &[(0, loc(1, 0)), (1, loc(0, 6))])
    }
    const TWO_ATOM_ROOT: [(u32, LocationAddr); 2] = [
        (
            0,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 0,
            },
        ),
        (
            1,
            LocationAddr {
                zone_id: 0,
                word_id: 0,
                site_id: 1,
            },
        ),
    ];

    // ── Synthetic single-atom graphs with empty move sets ──
    //
    // The generators emit empty `MoveSet`s (`check_lanes(&[]) == []`, so the
    // debug validation passes) and hand-written edge costs, so a test can pin
    // the loop's bookkeeping on a graph it controls, as the frontier tests do.

    struct Table {
        edges: HashMap<u32, Vec<(u32, f64)>>,
    }

    impl Table {
        fn new(edges: &[(u32, &[(u32, f64)])]) -> Self {
            Self {
                edges: edges.iter().map(|(s, e)| (*s, e.to_vec())).collect(),
            }
        }
        fn site(config: &Config) -> u32 {
            config.location_of(0).unwrap().site_id
        }
    }

    impl MoveGenerator for Table {
        fn generate(
            &self,
            config: &Config,
            _node_id: NodeId,
            _ctx: &SearchContext,
            _state: &mut SearchState,
            out: &mut Vec<MoveCandidate>,
        ) {
            if let Some(edges) = self.edges.get(&Self::site(config)) {
                for &(to, _) in edges {
                    out.push(MoveCandidate {
                        move_set: MoveSet::from_encoded(vec![]),
                        new_config: config.with_moves(&[(0, loc(0, to))]),
                    });
                }
            }
        }
    }

    /// Costs read from the tables of every stage (a `(from, to)` edge has one
    /// cost wherever it appears).
    struct TableCost {
        costs: HashMap<(u32, u32), f64>,
    }

    impl TableCost {
        fn of(tables: &[&Table]) -> Self {
            let mut costs = HashMap::new();
            for t in tables {
                for (&from, edges) in &t.edges {
                    for &(to, c) in edges {
                        costs.insert((from, to), c);
                    }
                }
            }
            Self { costs }
        }
    }

    impl CostFn for TableCost {
        fn edge_cost(&self, _m: &MoveSet, from: &Config, to: &Config) -> f64 {
            self.costs[&(Table::site(from), Table::site(to))]
        }
    }

    impl Objective for TableCost {
        fn lane_weight(&self, _lane: bloqade_lanes_bytecode_core::arch::addr::LaneAddr) -> f64 {
            0.0
        }
        fn min_shot_cost(&self) -> f64 {
            1.0
        }
        fn id(&self) -> ObjectiveId {
            ObjectiveId {
                kind: "test-table",
                params: 0,
            }
        }
    }

    struct SiteGoal(u32);

    impl Goal for SiteGoal {
        fn is_goal(&self, config: &Config) -> bool {
            Table::site(config) == self.0
        }
    }

    /// A hand-written admissible estimate per site (default 0).
    struct TableBound {
        h: HashMap<u32, f64>,
    }

    impl CompletionBound for TableBound {
        type Obj = TableCost;
        fn objective_id(&self) -> ObjectiveId {
            ObjectiveId {
                kind: "test-table",
                params: 0,
            }
        }
        fn estimate(&self, config: &Config) -> f64 {
            self.h.get(&Table::site(config)).copied().unwrap_or(0.0)
        }
    }

    fn synthetic_ctx() -> Fx {
        Fx::new(example_arch_json(), &[])
    }

    // ── Optimality ──

    #[test]
    fn optimal_against_brute_force_under_uniform_and_weighted_duration() {
        type Pairs<'a> = &'a [(u32, LocationAddr)];
        let instances: [(Pairs, Pairs); 3] = [
            (&[(0, loc(0, 0))], &[(0, loc(0, 5))]),
            (&TWO_ATOM_ROOT, &[(0, loc(1, 0)), (1, loc(0, 6))]),
            (
                &[(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(1, 2))],
                &[(0, loc(1, 5)), (1, loc(0, 6)), (2, loc(0, 2))],
            ),
        ];
        for kind in [
            ObjectiveKind::Uniform,
            ObjectiveKind::WeightedDuration { tau: Some(1.0) },
        ] {
            for (root, targets) in instances {
                let fx = Fx::new(&asymmetric_duration_arch_json(), targets);
                let ctx = fx.ctx();
                let goal = fx.goal();
                let objective = SolveObjective::from_kind(kind, &fx.index);
                let bound =
                    WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
                let ex = exhaustive(&ctx);
                let schedule = Schedule::complete(vec![], &ex, None);
                let bnb =
                    BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::UNLIMITED);
                let r = bnb.run(
                    cfg(root),
                    &schedule,
                    LifoFrontier::new(),
                    &mut NoOpObserver,
                    None,
                );
                let goal_id = r.goal.expect("solvable instance");
                assert_eq!(r.termination, Termination::Exhausted { proof: true });
                let cost = r.graph.g_score(goal_id);
                let reference = dijkstra(&ctx, cfg(root), &ex, &objective, &goal, 200_000)
                    .expect("reference solves");
                assert!(
                    (cost - reference).abs() < 1e-9,
                    "{kind:?} {root:?}: B&B {cost} vs Dijkstra {reference}"
                );
                assert_eq!(r.nodes_expanded, r.stage_expansions.iter().sum::<u32>());
            }
        }
    }

    #[test]
    fn no_bound_collapses_to_dfs_with_incumbent() {
        let fx = two_atom();
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let ex = exhaustive(&ctx);
        let schedule = Schedule::complete(vec![], &ex, None);

        let none = NoBound::for_objective(&objective);
        let plain = BranchAndBound::new(&objective, &none, &goal, &ctx, None, Widening::UNLIMITED)
            .run(
                cfg(&TWO_ATOM_ROOT),
                &schedule,
                LifoFrontier::new(),
                &mut NoOpObserver,
                None,
            );
        let h0 = WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
        let bounded = BranchAndBound::new(&objective, &h0, &goal, &ctx, None, Widening::UNLIMITED)
            .run(
                cfg(&TWO_ATOM_ROOT),
                &schedule,
                LifoFrontier::new(),
                &mut NoOpObserver,
                None,
            );

        let cost = |r: &SearchResult| r.graph.g_score(r.goal.unwrap());
        assert_eq!(cost(&plain), cost(&bounded));
        assert_eq!(
            plain.bound_stats,
            BoundStats {
                incumbent_cost: Some(cost(&plain)),
                ..BoundStats::default()
            }
        );
        assert!(bounded.bound_stats.bound_enabled);
        assert!(
            bounded.bound_stats.cuts_by_h > 0,
            "{:?}",
            bounded.bound_stats
        );
        assert!(bounded.nodes_expanded <= plain.nodes_expanded);
    }

    #[test]
    fn incumbents_strictly_decrease() {
        let fx = two_atom();
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let none = NoBound::for_objective(&objective);
        let ex = exhaustive(&ctx);
        let schedule = Schedule::complete(vec![], &ex, None);
        let mut rec = Recorder::default();
        let r = BranchAndBound::new(&objective, &none, &goal, &ctx, None, Widening::UNLIMITED).run(
            cfg(&TWO_ATOM_ROOT),
            &schedule,
            LifoFrontier::new(),
            &mut rec,
            None,
        );
        assert!(!rec.goals.is_empty());
        let costs: Vec<f64> = rec
            .goals
            .iter()
            .map(|&(id, _)| r.graph.g_score(id))
            .collect();
        assert!(costs.windows(2).all(|w| w[1] < w[0]), "{costs:?}");
        assert_eq!(r.goal, Some(rec.goals.last().unwrap().0));
    }

    /// Under BFS the deep, dear path to site 3 (via 1, cost 6) is queued
    /// before the cheap one (via 2, cost 2) is found; the stale id is still
    /// in the frontier when its cheaper twin is minted, and must never be
    /// expanded.
    #[test]
    fn superseded_ids_are_not_expanded() {
        let fx = synthetic_ctx();
        let ctx = fx.ctx();
        let table = Table::new(&[
            (0, &[(1, 1.0), (2, 1.0)]),
            (1, &[(3, 5.0)]),
            (2, &[(3, 1.0)]),
            (3, &[(4, 1.0)]),
        ]);
        let cost = TableCost::of(&[&table]);
        let none = NoBound::for_objective(&cost);
        let goal = SiteGoal(4);
        let schedule = Schedule::partial(vec![&table]);
        let mut rec = Recorder::default();
        let r = BranchAndBound::new(&cost, &none, &goal, &ctx, None, Widening::UNLIMITED).run(
            cfg(&[(0, loc(0, 0))]),
            &schedule,
            BfsFrontier::new(),
            &mut rec,
            None,
        );
        assert_eq!(r.graph.g_score(r.goal.unwrap()), 3.0);
        assert!(
            r.graph.len() > r.graph.num_configs(),
            "a rediscovery happened"
        );
        for (id, ..) in &rec.expanded {
            assert!(r.graph.is_current(*id), "stale {id:?} was expanded");
        }
        assert_eq!(rec.expanded.len() as u32, r.nodes_expanded);
        assert_eq!(r.nodes_expanded, r.stage_expansions.iter().sum::<u32>());
        assert_eq!(r.nodes_expanded, 4, "root, 1, 2 and the re-minted 3");
    }

    #[test]
    #[should_panic(expected = "same objective instance")]
    fn pairing_assertion_fires() {
        let fx = Fx::new(&asymmetric_duration_arch_json(), &[(0, loc(0, 5))]);
        let ctx = fx.ctx();
        let goal = fx.goal();
        let driver_objective = WeightedDuration::new(&fx.index, 5.0);
        let bound_objective = WeightedDuration::new(&fx.index, 1.0);
        let bound =
            WeightedDistanceBound::new(&bound_objective, &fx.targets, &fx.index, &fx.blocked);
        let _ = BranchAndBound::new(
            &driver_objective,
            &bound,
            &goal,
            &ctx,
            None,
            Widening::UNLIMITED,
        );
    }

    #[test]
    fn termination_is_reported_not_inferred() {
        let fx = two_atom();
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let bound = WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
        let ex = exhaustive(&ctx);
        let complete = Schedule::complete(vec![], &ex, None);
        let root = cfg(&TWO_ATOM_ROOT);

        let budget = BranchAndBound::new(
            &objective,
            &bound,
            &goal,
            &ctx,
            Some(1),
            Widening::UNLIMITED,
        )
        .run(
            root.clone(),
            &complete,
            LifoFrontier::new(),
            &mut NoOpObserver,
            None,
        );
        assert_eq!(budget.termination, Termination::Budget);
        assert_eq!(budget.nodes_expanded, 1);

        let proven =
            BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::UNLIMITED).run(
                root.clone(),
                &complete,
                LifoFrontier::new(),
                &mut NoOpObserver,
                None,
            );
        assert_eq!(proven.termination, Termination::Exhausted { proof: true });

        let partial = Schedule::partial(vec![&ex]);
        let unproven =
            BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::UNLIMITED).run(
                root.clone(),
                &partial,
                LifoFrontier::new(),
                &mut NoOpObserver,
                None,
            );
        assert_eq!(
            unproven.termination,
            Termination::Exhausted { proof: false }
        );

        // A seed makes C finite from the first pop, so with the knob at 0 the
        // exhaustive stage is withheld from every node stage 0 expanded.
        let heuristic = HeuristicGenerator::configured(0, DeadlockPolicy::Skip, false, None);
        let staged = Schedule::complete(vec![&heuristic], &ex, None);
        let withheld =
            BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::default()).run(
                root.clone(),
                &staged,
                LifoFrontier::new(),
                &mut NoOpObserver,
                Some(100.0),
            );
        assert!(withheld.goal.is_some());
        assert_eq!(
            withheld.termination,
            Termination::Exhausted { proof: false }
        );

        // A seed that already cuts the root at the pop gate withholds nothing:
        // the exhaustion proves the seed optimal.
        let seed = bound.estimate(&root);
        assert!(seed > 0.0);
        let root_cut =
            BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::default()).run(
                root,
                &staged,
                LifoFrontier::new(),
                &mut NoOpObserver,
                Some(seed),
            );
        assert_eq!(root_cut.goal, None);
        assert_eq!(root_cut.nodes_expanded, 0);
        assert_eq!(root_cut.termination, Termination::Exhausted { proof: true });
        assert_eq!(root_cut.bound_stats.incumbent_cost, None);
    }

    #[test]
    fn root_is_a_goal() {
        let fx = Fx::new(example_arch_json(), &[(0, loc(0, 0))]);
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let bound = WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
        let ex = exhaustive(&ctx);
        let schedule = Schedule::complete(vec![], &ex, None);
        let bnb = BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::default());
        let root = cfg(&[(0, loc(0, 0))]);

        let r = bnb.run(
            root.clone(),
            &schedule,
            LifoFrontier::new(),
            &mut NoOpObserver,
            None,
        );
        assert_eq!(r.goal, Some(r.graph.root()));
        assert_eq!(r.nodes_expanded, 0);
        assert_eq!(r.termination, Termination::Exhausted { proof: true });
        assert!(r.bound_stats.bound_enabled);
        assert_eq!(r.bound_stats.root_lower_bound, 0.0);
        assert_eq!(r.bound_stats.incumbent_cost, Some(0.0));
        assert_eq!(r.stage_expansions, vec![0]);

        let seeded = bnb.run(
            root.clone(),
            &schedule,
            LifoFrontier::new(),
            &mut NoOpObserver,
            Some(3.0),
        );
        assert_eq!(seeded.goal, Some(seeded.graph.root()));

        // A seed of 0 is not beaten by the root's 0: the seed stands.
        let tied = bnb.run(
            root,
            &schedule,
            LifoFrontier::new(),
            &mut NoOpObserver,
            Some(0.0),
        );
        assert_eq!(tied.goal, None);
        assert_eq!(tied.bound_stats.incumbent_cost, None);
    }

    // ── Ordering and staging ──

    #[test]
    fn generator_order_reaches_the_stack() {
        let fx = two_atom();
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let none = NoBound::for_objective(&objective);
        let ex = exhaustive(&ctx);
        let schedule = Schedule::complete(vec![], &ex, None);
        let root = cfg(&TWO_ATOM_ROOT);
        let mut cands = Vec::new();
        ex.generate(
            &root,
            NodeId(0),
            &ctx,
            &mut SearchState::default(),
            &mut cands,
        );
        assert!(cands.len() > 1);
        let first = cands.first().unwrap().new_config.clone();
        let last = cands.last().unwrap().new_config.clone();
        assert!(!goal.is_goal(&first) && !goal.is_goal(&last));

        let bnb = BranchAndBound::new(&objective, &none, &goal, &ctx, Some(2), Widening::UNLIMITED);
        let mut lifo = Recorder::default();
        bnb.run(
            root.clone(),
            &schedule,
            LifoFrontier::new(),
            &mut lifo,
            None,
        );
        assert_eq!(
            lifo.expanded[1].3, first,
            "LIFO pops the generator's first child"
        );

        let mut dfs = Recorder::default();
        bnb.run(
            root,
            &schedule,
            DfsFrontier::new(|_: &Config| 0.0),
            &mut dfs,
            None,
        );
        assert_eq!(
            dfs.expanded[1].3, last,
            "DFS with a constant h pops the last child"
        );
    }

    #[test]
    fn stages_reveal_in_order() {
        let fx = two_atom();
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let none = NoBound::for_objective(&objective);
        let ex = exhaustive(&ctx);
        let heuristic = HeuristicGenerator::configured(0, DeadlockPolicy::Skip, false, None);
        let schedule = Schedule::complete(vec![&heuristic], &ex, None);
        let mut rec = Recorder::default();
        let r = BranchAndBound::new(&objective, &none, &goal, &ctx, None, Widening::UNLIMITED).run(
            cfg(&TWO_ATOM_ROOT),
            &schedule,
            LifoFrontier::new(),
            &mut rec,
            None,
        );
        assert_eq!(r.stage_expansions.len(), 2);
        assert!(
            r.stage_expansions[1] > 0,
            "the exhaustive stage was revealed"
        );
        assert_eq!(r.nodes_expanded, r.stage_expansions.iter().sum::<u32>());
        // Each (node, stage) at most once.
        let mut seen = HashSet::new();
        for (id, stage, ..) in &rec.expanded {
            assert!(
                seen.insert((*id, *stage)),
                "{id:?} expanded twice at stage {stage}"
            );
        }
        // The first stage-1 expansion is the shallowest pending entry: the root.
        let first_stage1 = rec.expanded.iter().find(|e| e.1 == 1).unwrap();
        assert_eq!(first_stage1.0, r.graph.root());
        assert_eq!(first_stage1.2, 0);
        // No stage-1 expansion before the whole stage-0 tree was tried: every
        // stage-0 expansion of a node reached at stage 0 precedes the first
        // stage-1 pop.
        let first_stage1_pos = rec.expanded.iter().position(|e| e.1 == 1).unwrap();
        let stage0_before = rec.expanded[..first_stage1_pos]
            .iter()
            .filter(|e| e.1 == 0)
            .count();
        assert_eq!(stage0_before, first_stage1_pos);
    }

    #[test]
    fn widening_after_incumbent() {
        let fx = two_atom();
        let ctx = fx.ctx();
        let goal = fx.goal();
        let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
        let bound = WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
        let ex = exhaustive(&ctx);
        let heuristic = HeuristicGenerator::configured(0, DeadlockPolicy::Skip, false, None);
        let schedule = Schedule::complete(vec![&heuristic], &ex, None);

        let mut rec = Recorder::default();
        let r = BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::default())
            .run(
                cfg(&TWO_ATOM_ROOT),
                &schedule,
                LifoFrontier::new(),
                &mut rec,
                None,
            );
        assert!(r.goal.is_some());
        let first_goal_at = rec.goals[0].1;
        assert!(
            rec.expanded[first_goal_at..].iter().all(|e| e.1 == 0),
            "no widening after the first incumbent with the knob at 0"
        );
        assert_eq!(r.termination, Termination::Exhausted { proof: false });

        let r = BranchAndBound::new(&objective, &bound, &goal, &ctx, None, Widening::UNLIMITED)
            .run(
                cfg(&TWO_ATOM_ROOT),
                &schedule,
                LifoFrontier::new(),
                &mut NoOpObserver,
                None,
            );
        let reference =
            dijkstra(&ctx, cfg(&TWO_ATOM_ROOT), &ex, &objective, &goal, 200_000).unwrap();
        assert_eq!(r.graph.g_score(r.goal.unwrap()), reference);
        assert_eq!(r.termination, Termination::Exhausted { proof: true });
    }

    /// Two hand-built cases. (a) P's stage-0 child is cut but P's own
    /// `g + h < C`, so P is widened; Q's own `g + h >= C`, so Q is never
    /// expanded at any stage. (b) N is expanded at stage 0 and its child is
    /// the goal; the incumbent then makes N's own `g + h >= C`, so N is not
    /// widened even though its stage-0 subtree was fine.
    #[test]
    fn pruning_is_the_only_filter() {
        let fx = synthetic_ctx();
        let ctx = fx.ctx();

        // (a)
        let stage0 = Table::new(&[(0, &[(1, 1.0), (2, 1.0)]), (1, &[(3, 1.0)])]);
        let stage1 = Table::new(&[(1, &[(4, 1.0)])]);
        let cost = TableCost::of(&[&stage0, &stage1]);
        let bound = TableBound {
            h: HashMap::from([(1, 2.0), (2, 20.0), (3, 50.0)]),
        };
        let schedule = Schedule::partial(vec![&stage0, &stage1]);
        let mut rec = Recorder::default();
        let r = BranchAndBound::new(&cost, &bound, &SiteGoal(9), &ctx, None, Widening::UNLIMITED)
            .run(
                cfg(&[(0, loc(0, 0))]),
                &schedule,
                LifoFrontier::new(),
                &mut rec,
                Some(10.0),
            );
        assert_eq!(r.goal, None);
        let expanded: Vec<(u32, u8)> = rec
            .expanded
            .iter()
            .map(|(_, s, _, c)| (Table::site(c), *s))
            .collect();
        assert!(
            expanded.contains(&(1, 0)) && expanded.contains(&(1, 1)),
            "{expanded:?}"
        );
        assert!(!expanded.iter().any(|&(site, _)| site == 2), "{expanded:?}");
        assert!(!expanded.iter().any(|&(site, _)| site == 3), "{expanded:?}");
        assert_eq!(
            r.bound_stats.cuts_by_h, 2,
            "Q at the push gate, site 3 at the push gate"
        );

        // (b)
        let stage0 = Table::new(&[(0, &[(1, 1.0)]), (1, &[(2, 1.0)])]);
        let stage1 = Table::new(&[(1, &[(5, 1.0)]), (0, &[(6, 1.0)])]);
        let cost = TableCost::of(&[&stage0, &stage1]);
        let bound = TableBound {
            h: HashMap::from([(1, 5.0)]),
        };
        let schedule = Schedule::partial(vec![&stage0, &stage1]);
        let mut rec = Recorder::default();
        let r = BranchAndBound::new(&cost, &bound, &SiteGoal(2), &ctx, None, Widening::UNLIMITED)
            .run(
                cfg(&[(0, loc(0, 0))]),
                &schedule,
                LifoFrontier::new(),
                &mut rec,
                None,
            );
        assert_eq!(r.graph.g_score(r.goal.unwrap()), 2.0);
        let expanded: Vec<(u32, u8)> = rec
            .expanded
            .iter()
            .map(|(_, s, _, c)| (Table::site(c), *s))
            .collect();
        assert!(expanded.contains(&(1, 0)));
        assert!(
            !expanded.contains(&(1, 1)),
            "N's own bound reaches C: not widened {expanded:?}"
        );
        assert!(
            expanded.contains(&(0, 1)),
            "the root (h = 0) is still widened"
        );
    }

    // ── Randomised: brute force and the disconnection detector ──

    /// A random instance: the initial placement and its targets.
    type Instance = (Vec<(u32, LocationAddr)>, Vec<(u32, LocationAddr)>);

    fn random_instances(
        rng: &mut rand::rngs::SmallRng,
        index: &LaneIndex,
        per_size: usize,
    ) -> Vec<Instance> {
        use rand::seq::{IndexedRandom, SliceRandom};
        let sites: Vec<LocationAddr> = (0..2u32)
            .flat_map(|w| (0..10u32).map(move |s| loc(w, s)))
            .collect();
        let encs: Vec<u64> = sites.iter().map(|l| l.encode()).collect();
        let reach = DistanceTable::new(&encs, index);
        let mut out = Vec::new();
        for n_atoms in 1..=3usize {
            for _ in 0..per_size {
                let mut picked = sites.clone();
                picked.shuffle(rng);
                let starts = &picked[..n_atoms];
                let mut targets: Vec<LocationAddr> = Vec::new();
                let mut ok = true;
                for &start in starts {
                    let options: Vec<LocationAddr> = sites
                        .iter()
                        .copied()
                        .filter(|&t| {
                            !targets.contains(&t)
                                && reach.distance(start.encode(), t.encode()).is_some()
                        })
                        .collect();
                    match options.choose(rng) {
                        Some(&t) => targets.push(t),
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                if ok {
                    out.push((
                        starts
                            .iter()
                            .enumerate()
                            .map(|(i, &l)| (i as u32, l))
                            .collect(),
                        targets
                            .iter()
                            .enumerate()
                            .map(|(i, &l)| (i as u32, l))
                            .collect(),
                    ));
                }
            }
        }
        out
    }

    #[test]
    fn bnb_matches_brute_force_on_randomized_instances() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0xB0B0_5EED);
        let spec: ArchSpec = serde_json::from_str(&asymmetric_duration_arch_json()).unwrap();
        let probe = LaneIndex::new(spec);
        let instances = random_instances(&mut rng, &probe, 4);
        let mut verified = 0usize;
        for (root, targets) in &instances {
            let fx = Fx::new(&asymmetric_duration_arch_json(), targets);
            let ctx = fx.ctx();
            let goal = fx.goal();
            let ex = exhaustive(&ctx);
            let schedule = Schedule::complete(vec![], &ex, None);
            let h = fx.h_sum();
            for kind in [
                ObjectiveKind::Uniform,
                ObjectiveKind::WeightedDuration { tau: Some(1.0) },
            ] {
                let objective = SolveObjective::from_kind(kind, &fx.index);
                let bound =
                    WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
                let bnb = BranchAndBound::new(
                    &objective,
                    &bound,
                    &goal,
                    &ctx,
                    Some(30_000),
                    Widening::UNLIMITED,
                );
                let reference = dijkstra(&ctx, cfg(root), &ex, &objective, &goal, 60_000);
                let runs = [
                    bnb.run(
                        cfg(root),
                        &schedule,
                        LifoFrontier::new(),
                        &mut NoOpObserver,
                        None,
                    ),
                    bnb.run(
                        cfg(root),
                        &schedule,
                        DfsFrontier::new(|c: &Config| h.estimate_sum(c)),
                        &mut NoOpObserver,
                        None,
                    ),
                    bnb.run(
                        cfg(root),
                        &schedule,
                        IdsFrontier::new(|c: &Config| h.estimate_sum(c)),
                        &mut NoOpObserver,
                        None,
                    ),
                ];
                for r in runs {
                    if !matches!(r.termination, Termination::Exhausted { .. }) {
                        continue;
                    }
                    assert_eq!(r.termination, Termination::Exhausted { proof: true });
                    let cost = r.goal.map(|id| r.graph.g_score(id));
                    let Some(reference) = reference else { continue };
                    let cost = cost.expect("exhausted with a proof on a solvable instance");
                    assert!(
                        (cost - reference).abs() < 1e-9,
                        "{kind:?} {root:?} -> {targets:?}: {cost} vs {reference}"
                    );
                    verified += 1;
                }
            }
        }
        assert!(
            verified >= 30,
            "only {verified} runs verified; the test is not exercising the driver"
        );
    }

    /// Push and Rotate is complete at two or more empties: an instance it
    /// solves must not exhaust the complete schedule without a goal.
    #[test]
    fn disconnection_detector() {
        use crate::search::result::SolveStatus;
        use rand::SeedableRng;
        let mut rng = rand::rngs::SmallRng::seed_from_u64(0xD15C_0000);
        let spec: ArchSpec = serde_json::from_str(example_arch_json()).unwrap();
        let probe = LaneIndex::new(spec);
        let instances = random_instances(&mut rng, &probe, 4);
        let mut checked = 0usize;
        for (root, targets) in &instances {
            let pr =
                crate::push_rotate::solve_push_rotate(&probe, root, targets, &[], 10_000).unwrap();
            if pr.status != SolveStatus::Solved {
                continue;
            }
            let fx = Fx::new(example_arch_json(), targets);
            let ctx = fx.ctx();
            let goal = fx.goal();
            let objective = SolveObjective::from_kind(ObjectiveKind::Uniform, &fx.index);
            let bound = WeightedDistanceBound::new(&objective, &fx.targets, &fx.index, &fx.blocked);
            let ex = exhaustive(&ctx);
            let schedule = Schedule::complete(vec![], &ex, None);
            let r = BranchAndBound::new(
                &objective,
                &bound,
                &goal,
                &ctx,
                Some(50_000),
                Widening::UNLIMITED,
            )
            .run(
                cfg(root),
                &schedule,
                LifoFrontier::new(),
                &mut NoOpObserver,
                None,
            );
            if matches!(r.termination, Termination::Exhausted { .. }) {
                assert!(
                    r.goal.is_some(),
                    "Push and Rotate solved {root:?} -> {targets:?} but the complete schedule exhausted without a goal"
                );
                checked += 1;
            }
        }
        assert!(checked >= 6, "only {checked} instances checked");
    }
}
