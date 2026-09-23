//! The behaviour-net case corpus: plain data against the `spec` types.
//!
//! The corpus is a permanent fixture: add a case whenever a change reaches a
//! path it does not cover yet, and never delete one just because its golden
//! moved.
//!
//! Group prefixes in the case names:
//! - `route/`: fixed-target routing, every strategy on every instance. The
//!   instances span the Gemini specs and six synthetic topologies (conveyor
//!   chains, a blocked chain head, a zone bus, zones with aligned site buses,
//!   asymmetric lane durations).
//! - `fallback/`, `mirror/`, `bound/`, `edge/`: the routing paths the
//!   benchmark gate covers weakly or not at all.
//! - `knobs/`: solver options no other group sets (AOD capacity, restarts,
//!   weight, seed, goal quota, deadlock policy, lookahead, `top_c`, `w_t`).
//! - `cz/`: CZ stages through each placement, including larger ones and
//!   stages on a non-Gemini arch.
//! - `anticipate/`: current behaviour that planned work is expected to change
//!   on purpose (see `docs/superpowers/plans/2026-08-18-search-refactor-epics.md`).
//!   When that work lands, only these goldens should move.

use crate::spec::{
    Arch, Case, Deadlock, Expect, Knobs, Loc, Placement, Problem, ProblemSpec, Status, Strategy,
    loc, zloc,
};

/// Expansion budget for routing cases: enough for the small instances here,
/// small enough to keep the debug-build suite fast.
const BUDGET: Option<u32> = Some(300);

type Placed = Vec<(u32, Loc)>;

fn placed(list: &[(u32, Loc)]) -> Placed {
    list.to_vec()
}

struct Instance {
    name: &'static str,
    arch: Arch,
    initial: Placed,
    target: Placed,
}

/// The fixed-target instances every strategy is run on.
fn instances() -> Vec<Instance> {
    vec![
        Instance {
            name: "logical_one_atom",
            arch: Arch::GeminiLogical,
            initial: placed(&[(0, loc(0, 0))]),
            target: placed(&[(0, loc(2, 0))]),
        },
        Instance {
            name: "logical_three_atoms",
            arch: Arch::GeminiLogical,
            initial: placed(&[(0, loc(0, 0)), (1, loc(2, 0)), (2, loc(4, 0))]),
            target: placed(&[(0, loc(1, 0)), (1, loc(3, 0)), (2, loc(5, 0))]),
        },
        Instance {
            name: "physical_two_atoms",
            arch: Arch::GeminiPhysical,
            initial: placed(&[(0, loc(0, 0)), (1, loc(0, 1))]),
            target: placed(&[(0, loc(1, 0)), (1, loc(1, 1))]),
        },
        Instance {
            name: "physical_four_atoms",
            arch: Arch::GeminiPhysical,
            initial: placed(&[
                (0, loc(0, 0)),
                (1, loc(0, 2)),
                (2, loc(2, 1)),
                (3, loc(2, 3)),
            ]),
            target: placed(&[
                (0, loc(1, 1)),
                (1, loc(1, 3)),
                (2, loc(3, 0)),
                (3, loc(3, 2)),
            ]),
        },
        // A cyclic permutation of words: no atom can go straight to its
        // target, so every plan needs a temporary move.
        Instance {
            name: "logical_cycle",
            arch: Arch::GeminiLogical,
            initial: placed(&[(0, loc(0, 0)), (1, loc(2, 0)), (2, loc(4, 0))]),
            target: placed(&[(0, loc(2, 0)), (1, loc(4, 0)), (2, loc(0, 0))]),
        },
        // A cyclic permutation of sites inside an even word, which has no
        // site bus: the atoms have to detour through an odd word.
        Instance {
            name: "physical_site_cycle",
            arch: Arch::GeminiPhysical,
            initial: placed(&[(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]),
            target: placed(&[(0, loc(0, 1)), (1, loc(0, 2)), (2, loc(0, 0))]),
        },
        // Six atoms crossing between two word pairs.
        Instance {
            name: "physical_congested",
            arch: Arch::GeminiPhysical,
            initial: placed(&[
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(2, 0)),
                (4, loc(2, 1)),
                (5, loc(2, 2)),
            ]),
            target: placed(&[
                (0, loc(2, 1)),
                (1, loc(2, 2)),
                (2, loc(3, 0)),
                (3, loc(0, 1)),
                (4, loc(0, 2)),
                (5, loc(1, 0)),
            ]),
        },
        // ── Synthetic architectures ──
        // Site bus 0 -> 5, then word bus 0 -> 1 on site 5.
        Instance {
            name: "example_site_then_word",
            arch: Arch::Example,
            initial: placed(&[(0, loc(0, 0))]),
            target: placed(&[(0, loc(1, 5))]),
        },
        // The same two hops for two atoms side by side: each hop is one
        // rectangle carrying both.
        Instance {
            name: "example_pair_in_parallel",
            arch: Arch::Example,
            initial: placed(&[(0, loc(0, 0)), (1, loc(0, 1))]),
            target: placed(&[(0, loc(1, 5)), (1, loc(1, 6))]),
        },
        // One atom along the whole conveyor: four hops.
        Instance {
            name: "chain_single_atom",
            arch: Arch::Chain,
            initial: placed(&[(0, loc(0, 0))]),
            target: placed(&[(0, loc(0, 4))]),
        },
        // A row of four shifts one site along the conveyor. Every
        // destination but the last is vacated in the same shot, so one
        // conveyor operation does it (#896).
        Instance {
            name: "chain_row_shift",
            arch: Arch::Chain,
            initial: placed(&[
                (0, loc(0, 0)),
                (1, loc(0, 1)),
                (2, loc(0, 2)),
                (3, loc(0, 3)),
            ]),
            target: placed(&[
                (0, loc(0, 1)),
                (1, loc(0, 2)),
                (2, loc(0, 3)),
                (3, loc(0, 4)),
            ]),
        },
        // Three atoms fill the chain, so its head (site 2) cannot vacate on
        // the chain bus. The head escapes over the word bus, then the other
        // two shift along the chain: two shots on two different buses (#910).
        Instance {
            name: "chain_blocked_head",
            arch: Arch::ChainWithSiding,
            initial: placed(&[(0, loc(0, 0)), (1, loc(0, 1)), (2, loc(0, 2))]),
            target: placed(&[(0, loc(0, 1)), (1, loc(0, 2)), (2, loc(1, 2))]),
        },
        // Across the zone bus, memory (zone 1) to gate (zone 0): one hop.
        Instance {
            name: "zone_bus_forward",
            arch: Arch::TwoZoneBus,
            initial: placed(&[(0, zloc(1, 1, 0))]),
            target: placed(&[(0, zloc(0, 0, 0))]),
        },
        // And back, on the reverse lane.
        Instance {
            name: "zone_bus_backward",
            arch: Arch::TwoZoneBus,
            initial: placed(&[(0, zloc(0, 0, 0))]),
            target: placed(&[(0, zloc(1, 1, 0))]),
        },
        // One lift in each zone. The buses share an id, but a shot must not
        // mix zones, so this takes two shots.
        Instance {
            name: "two_zone_lift_both",
            arch: Arch::TwoZoneAlignedSiteBus,
            initial: placed(&[(0, zloc(0, 0, 0)), (1, zloc(1, 1, 0))]),
            target: placed(&[(0, zloc(0, 0, 2)), (1, zloc(1, 1, 2))]),
        },
        // Two lifts inside one zone: a single rectangle.
        Instance {
            name: "two_zone_lift_one_zone",
            arch: Arch::TwoZoneAlignedSiteBus,
            initial: placed(&[(0, zloc(0, 0, 0)), (1, zloc(0, 0, 1))]),
            target: placed(&[(0, zloc(0, 0, 2)), (1, zloc(0, 0, 3))]),
        },
        // Out and back over the lane whose reverse takes a different time.
        Instance {
            name: "asymmetric_swap",
            arch: Arch::AsymmetricDuration,
            initial: placed(&[(0, loc(0, 0)), (1, loc(0, 6))]),
            target: placed(&[(0, loc(1, 5)), (1, loc(0, 1))]),
        },
    ]
}

fn route(arch: Arch, strategy: Strategy, initial: Placed, target: Placed) -> ProblemSpec {
    ProblemSpec {
        arch,
        strategy,
        knobs: Knobs::default(),
        problem: Problem::Route {
            initial,
            target,
            blocked: Vec::new(),
        },
        budget: BUDGET,
    }
}

fn case(name: impl Into<String>, spec: ProblemSpec) -> Case {
    Case {
        name: name.into(),
        spec,
        expect: Expect::default(),
        debug_only: false,
    }
}

impl Case {
    fn expect(mut self, expect: Expect) -> Self {
        self.expect = expect;
        self
    }

    fn debug_only(mut self) -> Self {
        self.debug_only = true;
        self
    }
}

impl ProblemSpec {
    fn knobs(mut self, knobs: Knobs) -> Self {
        self.knobs = knobs;
        self
    }

    fn budget(mut self, budget: Option<u32>) -> Self {
        self.budget = budget;
        self
    }

    fn blocked(mut self, list: &[Loc]) -> Self {
        match &mut self.problem {
            Problem::Route { blocked, .. } | Problem::CzStage { blocked, .. } => {
                *blocked = list.to_vec();
            }
        }
        self
    }
}

pub fn all() -> Vec<Case> {
    let mut cases = Vec::new();
    cases.extend(route_cases());
    cases.extend(fallback_cases());
    cases.extend(mirror_cases());
    cases.extend(bound_cases());
    cases.extend(edge_cases());
    cases.extend(knob_cases());
    cases.extend(cz_cases());
    cases.extend(big_cz_cases());
    cases.extend(anticipate_cases());
    cases
}

fn instance(name: &str) -> Instance {
    instances()
        .into_iter()
        .find(|i| i.name == name)
        .expect("instance exists")
}

fn on(instance: &Instance, strategy: Strategy) -> ProblemSpec {
    route(
        instance.arch,
        strategy,
        instance.initial.clone(),
        instance.target.clone(),
    )
}

/// Solver knobs no other group sets, each on an instance where it can matter.
fn knob_cases() -> Vec<Case> {
    let congested = instance("physical_congested");
    let cycle = instance("logical_cycle");
    let two = instance("physical_two_atoms");
    let row = instance("chain_row_shift");
    let asym = instance("asymmetric_swap");
    let cap = |x, y| Knobs {
        aod_capacity: Some((x, y)),
        ..Knobs::default()
    };
    let mut cases = vec![
        // A 1x1 AOD cap allows one atom per shot. The two atoms that shared a
        // shot now need two.
        case(
            "knobs/aod_1x1/physical_two_atoms/astar",
            on(&two, Strategy::AStar).knobs(cap(1, 1)),
        )
        .expect(solved_in(2)),
        // The conveyor shift has to go head first, one atom per shot: four.
        case(
            "knobs/aod_1x1/chain_row_shift/astar",
            on(&row, Strategy::AStar).knobs(cap(1, 1)),
        )
        .expect(solved_in(4)),
        // Push and Rotate does not honour the cap (documented on the option).
        case(
            "knobs/aod_1x1/physical_two_atoms/push_rotate",
            on(&two, Strategy::PushRotate).knobs(cap(1, 1)),
        ),
        case(
            "knobs/aod_2x1/physical_congested/entropy",
            on(&congested, Strategy::Entropy).knobs(cap(2, 1)),
        ),
        case(
            "knobs/aod_2x1/physical_congested/dfs",
            on(&congested, Strategy::Dfs).knobs(cap(2, 1)),
        ),
        // Weighted A*.
        case(
            "knobs/weight_2/physical_congested/astar",
            on(&congested, Strategy::AStar).knobs(Knobs {
                weight: Some(2.0),
                ..Knobs::default()
            }),
        ),
        case(
            "knobs/weight_2/logical_cycle/astar",
            on(&cycle, Strategy::AStar).knobs(Knobs {
                weight: Some(2.0),
                ..Knobs::default()
            }),
        ),
        // Entropy's seeded perturbation, alone and across restarts.
        case(
            "knobs/seed_7/physical_congested/entropy",
            on(&congested, Strategy::Entropy).knobs(Knobs {
                seed: Some(7),
                ..Knobs::default()
            }),
        ),
        case(
            "knobs/seed_7_restarts_3/physical_congested/entropy",
            on(&congested, Strategy::Entropy).knobs(Knobs {
                seed: Some(7),
                restarts: Some(3),
                ..Knobs::default()
            }),
        ),
        // The entropy driver's goal quota.
        case(
            "knobs/goal_candidates_1/logical_cycle/entropy",
            on(&cycle, Strategy::Entropy).knobs(Knobs {
                max_goal_candidates: Some(1),
                ..Knobs::default()
            }),
        ),
        case(
            "knobs/goal_candidates_10/logical_cycle/entropy",
            on(&cycle, Strategy::Entropy).knobs(Knobs {
                max_goal_candidates: Some(10),
                ..Knobs::default()
            }),
        ),
    ];
    // Parallel restarts, reduced by pick_best.
    for strategy in [Strategy::Entropy, Strategy::Ids, Strategy::Dfs] {
        cases.push(case(
            format!("knobs/restarts_4/physical_congested/{}", strategy.label()),
            on(&congested, strategy).knobs(Knobs {
                restarts: Some(4),
                ..Knobs::default()
            }),
        ));
    }
    // The heuristic generator's deadlock policy.
    for (label, policy) in [
        ("skip", Deadlock::Skip),
        ("move_blockers", Deadlock::MoveBlockers),
        ("all_moves", Deadlock::AllMoves),
    ] {
        for strategy in [Strategy::Dfs, Strategy::Greedy] {
            cases.push(case(
                format!(
                    "knobs/deadlock_{label}/physical_congested/{}",
                    strategy.label()
                ),
                on(&congested, strategy).knobs(Knobs {
                    deadlock_policy: Some(policy),
                    ..Knobs::default()
                }),
            ));
        }
    }
    for strategy in [Strategy::AStar, Strategy::Entropy] {
        cases.push(case(
            format!("knobs/lookahead/physical_congested/{}", strategy.label()),
            on(&congested, strategy).knobs(Knobs {
                lookahead: true,
                ..Knobs::default()
            }),
        ));
    }
    for strategy in [Strategy::Greedy, Strategy::Dfs] {
        cases.push(case(
            format!("knobs/top_c_2/physical_congested/{}", strategy.label()),
            on(&congested, strategy).knobs(Knobs {
                top_c: Some(2),
                ..Knobs::default()
            }),
        ));
    }
    // Lane-duration weighting in the entropy heuristic, on the arch where a
    // lane and its reverse take different times, and on a hard instance.
    for (label, w_t) in [("0", 0.0), ("1", 1.0)] {
        for (name, inst) in [
            ("asymmetric_swap", &asym),
            ("physical_congested", &congested),
        ] {
            cases.push(case(
                format!("knobs/w_t_{label}/{name}/entropy"),
                on(inst, Strategy::Entropy).knobs(Knobs {
                    w_t: Some(w_t),
                    ..Knobs::default()
                }),
            ));
        }
    }
    cases
}

/// A CZ stage with the given atoms and pairs.
fn stage(
    arch: Arch,
    placement: Placement,
    strategy: Strategy,
    initial: &[(u32, Loc)],
    pairs: &[(u32, u32)],
) -> ProblemSpec {
    ProblemSpec {
        arch,
        strategy,
        knobs: Knobs::default(),
        problem: Problem::CzStage {
            placement,
            initial: placed(initial),
            controls: pairs.iter().map(|p| p.0).collect(),
            targets: pairs.iter().map(|p| p.1).collect(),
            blocked: Vec::new(),
            future: Vec::new(),
        },
        budget: BUDGET,
    }
}

/// Larger CZ stages, and CZ stages on a non-Gemini arch.
fn big_cz_cases() -> Vec<Case> {
    let placements: [(&str, Placement, Strategy); 4] = [
        (
            "single_heuristic",
            Placement::SingleHeuristic { candidates: None },
            Strategy::AStar,
        ),
        ("loose_goal", Placement::LooseGoal, Strategy::Ids),
        ("nohome", Placement::NoHome, Strategy::Entropy),
        (
            "receding_horizon",
            Placement::RecedingHorizon,
            Strategy::Ids,
        ),
    ];
    // Eight atoms on home words, four pairs.
    let logical_four_pairs: Vec<(u32, Loc)> = (0..8).map(|q| (q, loc(2 * q, 0))).collect();
    // Four pairs over two word pairs and two sites, with two spectators.
    let physical_four_pairs = [
        (0, loc(0, 0)),
        (1, loc(2, 0)),
        (2, loc(0, 1)),
        (3, loc(2, 1)),
        (4, loc(4, 0)),
        (5, loc(6, 0)),
        (6, loc(4, 1)),
        (7, loc(6, 1)),
        (8, loc(0, 2)),
        (9, loc(2, 2)),
    ];
    let four_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)];
    // On the example arch an atom keeps its site index modulo 5, and CZ
    // partners share a site index: sites 0 and 5 can pair, sites 0 and 1
    // never can.
    let example_pairable = [(0, loc(0, 0)), (1, loc(0, 5))];
    let example_unpairable = [(0, loc(0, 0)), (1, loc(0, 1))];
    let mut cases = Vec::new();
    for (name, placement, strategy) in placements {
        cases.push(case(
            format!("cz/{name}/logical_four_pairs"),
            stage(
                Arch::GeminiLogical,
                placement.clone(),
                strategy,
                &logical_four_pairs,
                &four_pairs,
            ),
        ));
        cases.push(case(
            format!("cz/{name}/physical_four_pairs_spectators"),
            stage(
                Arch::GeminiPhysical,
                placement.clone(),
                strategy,
                &physical_four_pairs,
                &four_pairs,
            ),
        ));
        cases.push(case(
            format!("cz/{name}/example_pairable"),
            stage(
                Arch::Example,
                placement.clone(),
                strategy,
                &example_pairable,
                &[(0, 1)],
            ),
        ));
        cases.push(case(
            format!("cz/{name}/example_unpairable"),
            stage(
                Arch::Example,
                placement.clone(),
                strategy,
                &example_unpairable,
                &[(0, 1)],
            ),
        ));
    }
    cases
}

fn solved_in(layers: usize) -> Expect {
    Expect {
        status: Some(Status::Solved),
        layers: Some(layers),
        ..Expect::default()
    }
}

/// Hand-verified answers for the instances small enough to check by hand.
fn hand_verified(instance: &str, strategy: Strategy) -> Option<Expect> {
    // Every strategy, Push and Rotate included, must find these.
    let for_all = match instance {
        // Word 0 to word 2 on the logical arch is three word-bus hops.
        "logical_one_atom" => Some(3),
        // Both atoms shift word 0 -> word 1 on their own sites: one AOD shot.
        "physical_two_atoms" => Some(1),
        _ => None,
    };
    if let Some(layers) = for_all {
        return Some(solved_in(layers));
    }
    // Hand-derived optima (see each instance's comment). Only the strategies
    // that are provably optimal here, A* and BFS, must hit them; the others
    // may legitimately do worse.
    let optimum = match instance {
        "example_site_then_word" | "example_pair_in_parallel" => 2,
        "chain_single_atom" => 4,
        "chain_row_shift" => 1,
        "chain_blocked_head" => 2,
        "zone_bus_forward" | "zone_bus_backward" => 1,
        "two_zone_lift_both" => 2,
        "two_zone_lift_one_zone" => 1,
        // Forward and backward site-bus moves need separate shots, then the
        // word-bus hop: three.
        "asymmetric_swap" => 3,
        _ => return None,
    };
    matches!(strategy, Strategy::AStar | Strategy::Bfs).then(|| solved_in(optimum))
}

fn route_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for instance in instances() {
        for strategy in Strategy::ALL {
            let spec = route(
                instance.arch,
                strategy,
                instance.initial.clone(),
                instance.target.clone(),
            );
            let mut c = case(
                format!("route/{}/{}", instance.name, strategy.label()),
                spec,
            );
            if let Some(expect) = hand_verified(instance.name, strategy) {
                c = c.expect(expect);
            }
            cases.push(c);
        }
    }
    cases
}

fn fallback_cases() -> Vec<Case> {
    let one_atom = || {
        route(
            Arch::GeminiLogical,
            Strategy::AStar,
            placed(&[(0, loc(0, 0))]),
            placed(&[(0, loc(2, 0))]),
        )
        .budget(Some(1))
    };
    vec![
        // Three layers cannot be found in one expansion.
        case("fallback/off_budget_exhausted", one_atom()).expect(Expect {
            status: Some(Status::BudgetExceeded),
            ..Expect::default()
        }),
        // Push and Rotate finishes it; the search's counters are not kept.
        case(
            "fallback/on_rescues_exhausted_search",
            one_atom().knobs(Knobs {
                fallback_push_rotate: true,
                ..Knobs::default()
            }),
        )
        .expect(Expect {
            status: Some(Status::Solved),
            layers: Some(3),
            ..Expect::default()
        }),
        // A target on a blocked site: Push and Rotate proves it unsolvable,
        // and that proof is promoted over the search's unproven verdict.
        case(
            "fallback/on_promotes_unsolvable_proof",
            one_atom()
                .budget(BUDGET)
                .blocked(&[loc(2, 0)])
                .knobs(Knobs {
                    fallback_push_rotate: true,
                    ..Knobs::default()
                }),
        )
        .expect(Expect {
            status: Some(Status::Unsolvable),
            proven: Some(true),
            ..Expect::default()
        }),
    ]
}

fn mirror_cases() -> Vec<Case> {
    let mirrored = |budget| {
        route(
            Arch::GeminiPhysical,
            Strategy::AStar,
            placed(&[(0, loc(0, 0)), (1, loc(0, 1))]),
            placed(&[(0, loc(1, 0)), (1, loc(1, 1))]),
        )
        .budget(budget)
        .knobs(Knobs {
            backwards_search: true,
            ..Knobs::default()
        })
    };
    vec![
        case("mirror/solves", mirrored(BUDGET)).expect(Expect {
            status: Some(Status::Solved),
            layers: Some(1),
            ..Expect::default()
        }),
        // A failed mirror returns the root, not a partial.
        case("mirror/fails_returns_root", mirrored(Some(1))).expect(Expect {
            status: Some(Status::BudgetExceeded),
            layers: Some(0),
            ..Expect::default()
        }),
    ]
}

fn bound_cases() -> Vec<Case> {
    let bounded = |instance: &Instance, strategy, terminates| {
        route(
            instance.arch,
            strategy,
            instance.initial.clone(),
            instance.target.clone(),
        )
        .knobs(Knobs {
            completion_bound: true,
            bound_terminates: Some(terminates),
            ..Knobs::default()
        })
    };
    let mut cases = Vec::new();
    for instance in instances() {
        // On logical_one_atom the bound at the root is 3, the optimum, so the
        // root certificate proves the plan optimal -- but only when the bound
        // may end the search.
        let certified = instance.name == "logical_one_atom";
        let mut terminates = case(
            format!("bound/{}/entropy", instance.name),
            bounded(&instance, Strategy::Entropy, true),
        );
        let mut spins = case(
            format!("bound/{}/entropy_no_terminate", instance.name),
            bounded(&instance, Strategy::Entropy, false),
        );
        if certified {
            terminates = terminates.expect(Expect {
                status: Some(Status::Solved),
                layers: Some(3),
                proven: Some(true),
                ..Expect::default()
            });
            spins = spins.expect(Expect {
                status: Some(Status::Solved),
                proven: Some(false),
                ..Expect::default()
            });
        }
        cases.push(terminates);
        cases.push(spins);
    }
    cases
}

fn edge_cases() -> Vec<Case> {
    let logical = |initial: &[(u32, Loc)], target: &[(u32, Loc)]| {
        route(
            Arch::GeminiLogical,
            Strategy::AStar,
            placed(initial),
            placed(target),
        )
    };
    vec![
        case(
            "edge/already_at_goal",
            logical(&[(0, loc(0, 0))], &[(0, loc(0, 0))]),
        )
        .expect(Expect {
            status: Some(Status::Solved),
            layers: Some(0),
            ..Expect::default()
        }),
        // The search drains its frontier without proving anything.
        case(
            "edge/blocked_destination",
            logical(&[(0, loc(0, 0))], &[(0, loc(2, 0))]).blocked(&[loc(2, 0)]),
        )
        .expect(Expect {
            status: Some(Status::Unsolvable),
            proven: Some(false),
            ..Expect::default()
        }),
        case(
            "edge/duplicate_target_location",
            logical(
                &[(0, loc(0, 0)), (1, loc(2, 0))],
                &[(0, loc(4, 0)), (1, loc(4, 0))],
            ),
        )
        .expect(Expect {
            error: true,
            ..Expect::default()
        }),
        case(
            "edge/duplicate_initial_qubit",
            logical(&[(0, loc(0, 0)), (0, loc(2, 0))], &[(0, loc(4, 0))]),
        )
        .expect(Expect {
            error: true,
            ..Expect::default()
        }),
        // Partial targets: the corner no benchmark or caller reaches.
        case(
            "edge/partial_target/astar",
            logical(&[(0, loc(0, 0)), (1, loc(2, 0))], &[(0, loc(4, 0))]),
        ),
        // Push and Rotate builds its goal from the target list alone, so an
        // untargeted atom is missing from it and the replay check panics.
        case(
            "edge/partial_target/push_rotate",
            ProblemSpec {
                strategy: Strategy::PushRotate,
                ..logical(&[(0, loc(0, 0)), (1, loc(2, 0))], &[(0, loc(4, 0))])
            },
        )
        .expect(Expect {
            panic: true,
            ..Expect::default()
        }),
    ]
}

/// A CZ stage on the logical arch: four atoms on home (even) words, pairing
/// words 0 & 2 and words 4 & 6.
fn logical_stage(placement: Placement, strategy: Strategy) -> ProblemSpec {
    ProblemSpec {
        arch: Arch::GeminiLogical,
        strategy,
        knobs: Knobs::default(),
        problem: Problem::CzStage {
            placement,
            initial: placed(&[
                (0, loc(0, 0)),
                (1, loc(2, 0)),
                (2, loc(4, 0)),
                (3, loc(6, 0)),
            ]),
            controls: vec![0, 2],
            targets: vec![1, 3],
            blocked: Vec::new(),
            future: Vec::new(),
        },
        budget: BUDGET,
    }
}

/// A CZ stage on the physical arch, with a spectator atom next to the pairs.
fn physical_stage(placement: Placement, strategy: Strategy) -> ProblemSpec {
    ProblemSpec {
        arch: Arch::GeminiPhysical,
        strategy,
        knobs: Knobs::default(),
        problem: Problem::CzStage {
            placement,
            initial: placed(&[
                (0, loc(0, 0)),
                (1, loc(2, 0)),
                (2, loc(0, 1)),
                (3, loc(2, 1)),
                (4, loc(0, 2)),
            ]),
            controls: vec![0, 2],
            targets: vec![1, 3],
            blocked: Vec::new(),
            future: Vec::new(),
        },
        budget: BUDGET,
    }
}

fn with_future(mut spec: ProblemSpec, layers: Vec<Vec<(u32, u32)>>) -> ProblemSpec {
    if let Problem::CzStage { future, .. } = &mut spec.problem {
        *future = layers;
    }
    spec
}

fn mismatched(mut spec: ProblemSpec) -> ProblemSpec {
    if let Problem::CzStage { targets, .. } = &mut spec.problem {
        targets.pop();
    }
    spec
}

fn cz_cases() -> Vec<Case> {
    let placements: [(&str, Placement, Strategy); 4] = [
        (
            "single_heuristic",
            Placement::SingleHeuristic { candidates: None },
            Strategy::AStar,
        ),
        ("loose_goal", Placement::LooseGoal, Strategy::Ids),
        ("nohome", Placement::NoHome, Strategy::Entropy),
        (
            "receding_horizon",
            Placement::RecedingHorizon,
            Strategy::Ids,
        ),
    ];
    let mut cases = Vec::new();
    for (name, placement, strategy) in placements {
        cases.push(case(
            format!("cz/{name}/logical"),
            logical_stage(placement.clone(), strategy),
        ));
        cases.push(case(
            format!("cz/{name}/physical"),
            physical_stage(placement.clone(), strategy),
        ));
        cases.push(case(
            format!("cz/{name}/physical_with_future"),
            with_future(
                physical_stage(placement.clone(), strategy),
                vec![vec![(0, 3)], vec![(1, 2)]],
            ),
        ));
        // Mismatched controls/targets. The three placements handle it
        // differently today; the reshape to `CzStage` retires these cases.
        let mismatch = case(
            format!("cz/{name}/mismatched_lengths"),
            mismatched(logical_stage(placement.clone(), strategy)),
        );
        cases.push(match placement {
            // NoHome and RecedingHorizon only `debug_assert!` the lengths.
            Placement::NoHome | Placement::RecedingHorizon => mismatch.debug_only(),
            _ => mismatch,
        });
    }
    // Loose goal with spectators already facing each other, or facing a pair
    // qubit, across a CZ word pair. The goal itself forbids two spectators on
    // partner sites, so these cover that rule. They do NOT reach the
    // accidental-CZ cleanup leg in `solve_loose_goal`: that leg runs only on a
    // solved result and checks the same predicate against the same partner
    // map, so it can never find anything (verified by instrumenting it).
    let loose = |initial: &[(u32, Loc)], arch| ProblemSpec {
        arch,
        problem: Problem::CzStage {
            placement: Placement::LooseGoal,
            initial: placed(initial),
            controls: vec![0],
            targets: vec![1],
            blocked: Vec::new(),
            future: Vec::new(),
        },
        ..logical_stage(Placement::LooseGoal, Strategy::Ids)
    };
    cases.push(case(
        "cz/loose_goal/logical_spectators_facing",
        loose(
            &[
                (0, loc(0, 0)),
                (1, loc(2, 0)),
                (2, loc(4, 0)),
                (3, loc(5, 0)),
            ],
            Arch::GeminiLogical,
        ),
    ));
    cases.push(case(
        "cz/loose_goal/logical_spectator_facing_pair_qubit",
        loose(
            &[(0, loc(0, 0)), (1, loc(2, 0)), (2, loc(3, 0))],
            Arch::GeminiLogical,
        ),
    ));
    cases.push(case(
        "cz/loose_goal/physical_spectators_facing",
        loose(
            &[
                (0, loc(0, 0)),
                (1, loc(2, 0)),
                (2, loc(4, 3)),
                (3, loc(5, 3)),
                (4, loc(1, 0)),
            ],
            Arch::GeminiPhysical,
        ),
    ));
    // NoHome with an atom off its home word, so the return phase runs.
    cases.push(case(
        "cz/nohome/logical_with_returner",
        ProblemSpec {
            problem: Problem::CzStage {
                placement: Placement::NoHome,
                initial: placed(&[
                    (0, loc(1, 0)),
                    (1, loc(2, 0)),
                    (2, loc(4, 0)),
                    (3, loc(6, 0)),
                ]),
                controls: vec![0, 2],
                targets: vec![1, 3],
                blocked: Vec::new(),
                future: Vec::new(),
            },
            ..logical_stage(Placement::NoHome, Strategy::Entropy)
        },
    ));
    // NoHome mirrored, as the default pipeline runs it under palindrome.
    cases.push(case(
        "cz/nohome/logical_mirrored",
        logical_stage(Placement::NoHome, Strategy::Entropy).knobs(Knobs {
            backwards_search: true,
            ..Knobs::default()
        }),
    ));
    cases
}

fn anticipate_cases() -> Vec<Case> {
    let instance = |name: &str| {
        instances()
            .into_iter()
            .find(|i| i.name == name)
            .expect("instance exists")
    };
    let cycle = instance("logical_cycle");
    let congested = instance("physical_congested");
    let on = |i: &Instance, strategy| route(i.arch, strategy, i.initial.clone(), i.target.clone());
    // Candidate placements for the logical stage's first pair (qubits 0 & 1).
    // Today the first candidate that solves wins, so order alone decides the
    // output, even when a cheaper candidate comes later. Epic 4 changes that
    // on purpose.
    let move_control = placed(&[
        (0, loc(3, 0)),
        (1, loc(2, 0)),
        (2, loc(4, 0)),
        (3, loc(6, 0)),
    ]);
    let move_target = placed(&[
        (0, loc(0, 0)),
        (1, loc(1, 0)),
        (2, loc(4, 0)),
        (3, loc(6, 0)),
    ]);
    // Both atoms travel to a distant word pair: solvable, but costlier.
    let move_both_far = placed(&[
        (0, loc(8, 0)),
        (1, loc(9, 0)),
        (2, loc(4, 0)),
        (3, loc(6, 0)),
    ]);
    let single_pair = |candidates: Vec<Placed>| ProblemSpec {
        problem: Problem::CzStage {
            placement: Placement::SingleHeuristic {
                candidates: Some(candidates),
            },
            initial: placed(&[
                (0, loc(0, 0)),
                (1, loc(2, 0)),
                (2, loc(4, 0)),
                (3, loc(6, 0)),
            ]),
            controls: vec![0],
            targets: vec![1],
            blocked: Vec::new(),
            future: Vec::new(),
        },
        ..logical_stage(
            Placement::SingleHeuristic { candidates: None },
            Strategy::AStar,
        )
    };
    vec![
        // Frontier bound-wiring (Epic 2B). On logical_cycle, IDS finds 7
        // layers and the cascade's A* refinement, capped only by g, spends its
        // expansions finding 6. A g + h prune should cut that work.
        case(
            "anticipate/cascade_memory/ids",
            on(&cycle, Strategy::CascadeIds),
        ),
        case(
            "anticipate/cascade_memory/entropy_bounded",
            on(&cycle, Strategy::CascadeEntropy).knobs(Knobs {
                completion_bound: true,
                ..Knobs::default()
            }),
        ),
        // P&R resume (Epic 2B). A* runs out of budget on physical_congested,
        // and today the fallback restarts Push and Rotate from the initial
        // placement, discarding the search's partial progress.
        case(
            "anticipate/pr_resume/restart_from_initial",
            on(&congested, Strategy::AStar).knobs(Knobs {
                fallback_push_rotate: true,
                ..Knobs::default()
            }),
        ),
        // Candidate ranking (Epic 4): first-solve-wins.
        case(
            "anticipate/candidate_order/move_control_first",
            single_pair(vec![move_control.clone(), move_target.clone()]),
        ),
        case(
            "anticipate/candidate_order/move_target_first",
            single_pair(vec![move_target.clone(), move_control.clone()]),
        ),
        case(
            "anticipate/candidate_order/costly_first",
            single_pair(vec![move_both_far, move_control, move_target]),
        ),
    ]
}
