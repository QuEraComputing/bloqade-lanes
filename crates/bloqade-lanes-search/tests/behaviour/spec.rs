//! Test-domain types for the behaviour net.
//!
//! Nothing in this module names a `bloqade_lanes_search` type. Cases are
//! plain data written against these types, and only `interface.rs` translates
//! them into crate calls and crate results back into an [`Outcome`]. That is
//! what lets the crate's API change without touching a single case.

use std::fmt::Write as _;

/// Which architecture a case runs on. The Gemini specs are the bundled ones.
/// The rest live in `tests/fixtures/behaviour/arch/` and cover topologies
/// Gemini does not have: snapshots of the crate's synthetic unit-test specs,
/// plus `TwoZoneGrid`, written for this net.
#[derive(Clone, Copy, Debug)]
pub enum Arch {
    /// Gemini logical: 20 words of one site; word buses only; CZ pairs are
    /// the word pairs (0, 1), (2, 3), ...
    GeminiLogical,
    /// Gemini physical: 20 words of eight sites; site buses on odd words,
    /// word buses on every site; CZ pairs as for logical.
    GeminiPhysical,
    /// Two words of ten sites. A site bus lifts sites 0-4 onto 5-9, a word
    /// bus joins the words on sites 5-9, and words 0 and 1 are CZ partners.
    /// An atom keeps its site index modulo 5.
    Example,
    /// The example arch with site bus 0 rewired as a conveyor chain
    /// 0 -> 1 -> 2 -> 3 -> 4. The only kind of spec on which the chain
    /// assembly paths are reachable.
    Chain,
    /// Two words of three sites, a chain site bus 0 -> 1 -> 2 in each, and a
    /// word bus joining them at every site: a chain whose head can be blocked.
    ChainWithSiding,
    /// Zone 0 holds word 0 and zone 1 word 1, each of one site, joined only by
    /// a zone bus from zone 1 to zone 0.
    TwoZoneBus,
    /// Two zones, one word each, each with a site bus lifting sites 0, 1 onto
    /// 2, 3. The buses share an id, so a shot could wrongly mix the zones.
    TwoZoneAlignedSiteBus,
    /// The example arch with transport paths that make a lane and its reverse
    /// take different times.
    AsymmetricDuration,
    /// A small 2D two-zone arch with sparse connectivity: a storage zone
    /// (words 0-3) and a gate zone (words 4-7), each a 4x4 grid of four
    /// row-words. Along a row, sites form a path 0-1-2-3; between rows, words
    /// form a path too, but only on the edge columns (sites 0 and 3). A single
    /// zone bus joins storage's top row (word 3) to gate's bottom row
    /// (word 4), and only the gate zone has CZ pairs (words 4 & 5, 6 & 7).
    /// Every inter-zone move funnels through that one bus.
    TwoZoneGrid,
}

/// A location. Zone 0 unless built with [`zloc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Loc {
    pub zone: u32,
    pub word: u32,
    pub site: u32,
}

pub const fn loc(word: u32, site: u32) -> Loc {
    Loc {
        zone: 0,
        word,
        site,
    }
}

pub const fn zloc(zone: u32, word: u32, site: u32) -> Loc {
    Loc { zone, word, site }
}

/// How the heuristic generator handles a deadlocked configuration.
#[derive(Clone, Copy, Debug)]
pub enum Deadlock {
    Skip,
    MoveBlockers,
    AllMoves,
}

/// The routing strategy, one per crate strategy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Strategy {
    AStar,
    Dfs,
    Bfs,
    Greedy,
    Ids,
    CascadeIds,
    CascadeDfs,
    CascadeEntropy,
    Entropy,
    PushRotate,
}

impl Strategy {
    pub const ALL: [Strategy; 10] = [
        Strategy::AStar,
        Strategy::Dfs,
        Strategy::Bfs,
        Strategy::Greedy,
        Strategy::Ids,
        Strategy::CascadeIds,
        Strategy::CascadeDfs,
        Strategy::CascadeEntropy,
        Strategy::Entropy,
        Strategy::PushRotate,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Strategy::AStar => "astar",
            Strategy::Dfs => "dfs",
            Strategy::Bfs => "bfs",
            Strategy::Greedy => "greedy",
            Strategy::Ids => "ids",
            Strategy::CascadeIds => "cascade_ids",
            Strategy::CascadeDfs => "cascade_dfs",
            Strategy::CascadeEntropy => "cascade_entropy",
            Strategy::Entropy => "entropy",
            Strategy::PushRotate => "push_rotate",
        }
    }
}

/// Solver knobs. `None` means "the crate's default".
#[derive(Clone, Debug, Default)]
pub struct Knobs {
    pub restarts: Option<u32>,
    pub weight: Option<f64>,
    pub fallback_push_rotate: bool,
    pub backwards_search: bool,
    pub max_goal_candidates: Option<usize>,
    pub seed: Option<u64>,
    /// Request the weighted-distance completion bound.
    pub completion_bound: bool,
    pub bound_terminates: Option<bool>,
    pub deadlock_policy: Option<Deadlock>,
    pub lookahead: bool,
    pub top_c: Option<usize>,
    /// AOD tone limit per shot as (source columns, source rows).
    pub aod_capacity: Option<(usize, usize)>,
    /// Weight of lane duration against hop count in the entropy heuristic.
    pub w_t: Option<f64>,
}

/// Which CZ-stage placement a [`Problem::CzStage`] runs through.
#[derive(Clone, Debug)]
pub enum Placement {
    /// Single-heuristic placement. `None` uses the crate's default target
    /// generator; `Some` offers exactly these candidate placements, in this
    /// order.
    SingleHeuristic {
        candidates: Option<Vec<Vec<(u32, Loc)>>>,
    },
    LooseGoal,
    NoHome,
    RecedingHorizon,
}

/// The problem instance.
#[derive(Clone, Debug)]
pub enum Problem {
    /// Fixed-target routing.
    Route {
        initial: Vec<(u32, Loc)>,
        target: Vec<(u32, Loc)>,
        blocked: Vec<Loc>,
    },
    /// One CZ stage through a placement. `controls[i]` and `targets[i]` are
    /// partnered. The lists are kept separate, rather than as pairs, so a case
    /// can express the mismatched-length input that today's API accepts.
    CzStage {
        placement: Placement,
        initial: Vec<(u32, Loc)>,
        controls: Vec<u32>,
        targets: Vec<u32>,
        blocked: Vec<Loc>,
        future: Vec<Vec<(u32, u32)>>,
    },
}

#[derive(Clone, Debug)]
pub struct ProblemSpec {
    pub arch: Arch,
    pub strategy: Strategy,
    pub knobs: Knobs,
    pub problem: Problem,
    pub budget: Option<u32>,
}

/// Hand-verified expectations for a semantic case. Only the fields set are
/// checked; a characterization-only case leaves this at its default.
#[derive(Clone, Debug, Default)]
pub struct Expect {
    pub status: Option<Status>,
    pub layers: Option<usize>,
    pub proven: Option<bool>,
    pub error: bool,
    pub panic: bool,
}

impl Expect {
    pub fn is_empty(&self) -> bool {
        self.status.is_none()
            && self.layers.is_none()
            && self.proven.is_none()
            && !self.error
            && !self.panic
    }
}

pub struct Case {
    pub name: String,
    pub spec: ProblemSpec,
    pub expect: Expect,
    /// The outcome depends on a `debug_assert!`, so it is only recorded in a
    /// debug build (which is what `cargo test` and CI use).
    pub debug_only: bool,
}

// ── Outcomes ───────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Solved,
    Unsolvable,
    BudgetExceeded,
}

impl Status {
    fn label(self) -> &'static str {
        match self {
            Status::Solved => "solved",
            Status::Unsolvable => "unsolvable",
            Status::BudgetExceeded => "budget_exceeded",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Termination {
    Budget,
    Exhausted { proof: bool },
    Stopped,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoundSummary {
    pub cuts_by_g: u64,
    pub cuts_by_h: u64,
    pub cuts_infeasible: u64,
    pub root_lower_bound: f64,
    pub incumbent_cost: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    pub index: usize,
    pub status: Status,
    pub nodes_expanded: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttemptLog {
    pub chosen: Option<usize>,
    pub total_expansions: u32,
    pub attempts: Vec<Attempt>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Run {
    pub status: Status,
    pub layers: usize,
    pub lanes: usize,
    pub cost: f64,
    pub nodes_expanded: u32,
    pub deadlocks: u32,
    pub proven: bool,
    pub termination: Termination,
    pub final_placement: Vec<(u32, Loc)>,
    /// A stable hash of the move sequence, so a changed plan is caught even
    /// when every count stays the same.
    pub plan_digest: u64,
    /// Present only when a real bound was active.
    pub bound: Option<BoundSummary>,
    /// Present only for placements that report one.
    pub attempts: Option<AttemptLog>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    Ran(Run),
    /// The call returned an error; the payload is its `Display`.
    Error(String),
    /// The call panicked; the payload is the first line of the message.
    Panicked(String),
}

impl Outcome {
    /// The stable text form compared against the golden file.
    pub fn render(&self) -> String {
        let mut out = String::new();
        match self {
            Outcome::Error(msg) => {
                let _ = writeln!(out, "error: {msg}");
            }
            Outcome::Panicked(msg) => {
                let _ = writeln!(out, "panicked: {msg}");
            }
            Outcome::Ran(run) => {
                let termination = match run.termination {
                    Termination::Budget => "budget",
                    Termination::Exhausted { proof: false } => "exhausted",
                    Termination::Exhausted { proof: true } => "exhausted_proof",
                    Termination::Stopped => "stopped",
                };
                let _ = writeln!(
                    out,
                    "status: {} | layers: {} | lanes: {} | cost: {} | expanded: {} | deadlocks: {} | proven: {} | termination: {}",
                    run.status.label(),
                    run.layers,
                    run.lanes,
                    run.cost,
                    run.nodes_expanded,
                    run.deadlocks,
                    run.proven,
                    termination,
                );
                let placement: Vec<String> = run
                    .final_placement
                    .iter()
                    // Zone 0 renders as `word.site`, so single-zone goldens
                    // stay as they were; other zones as `zone:word.site`.
                    .map(|(q, l)| match l.zone {
                        0 => format!("{q}@{}.{}", l.word, l.site),
                        z => format!("{q}@{z}:{}.{}", l.word, l.site),
                    })
                    .collect();
                let _ = writeln!(out, "final: {}", placement.join(" "));
                let _ = writeln!(out, "plan: {:016x}", run.plan_digest);
                if let Some(b) = &run.bound {
                    let incumbent = b
                        .incumbent_cost
                        .map_or_else(|| "none".to_string(), |c| c.to_string());
                    let _ = writeln!(
                        out,
                        "bound: cuts_g={} cuts_h={} cuts_inf={} root_lb={} incumbent={}",
                        b.cuts_by_g, b.cuts_by_h, b.cuts_infeasible, b.root_lower_bound, incumbent
                    );
                }
                if let Some(log) = &run.attempts {
                    let tries: Vec<String> = log
                        .attempts
                        .iter()
                        .map(|a| format!("{}:{}/{}", a.index, a.status.label(), a.nodes_expanded))
                        .collect();
                    let chosen = log
                        .chosen
                        .map_or_else(|| "none".to_string(), |i| i.to_string());
                    let _ = writeln!(
                        out,
                        "attempts: chosen={chosen} total={} [{}]",
                        log.total_expansions,
                        tries.join(", ")
                    );
                }
            }
        }
        out
    }
}
