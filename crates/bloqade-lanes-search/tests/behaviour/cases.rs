//! The behaviour-net case corpus: plain data against the `spec` types.
//!
//! Group prefixes in the case names:
//! - `route/`: fixed-target routing through every strategy.
//! - `fallback/`, `mirror/`, `bound/`, `edge/`: the routing paths the
//!   benchmark gate covers weakly or not at all.
//! - `cz/`: one CZ stage through each placement.
//! - `anticipate/`: current behaviour that a later phase of the refactor is
//!   expected to change on purpose. When that phase lands, only these goldens
//!   should move.

use crate::spec::{
    Arch, Case, Expect, Knobs, Loc, Placement, Problem, ProblemSpec, Status, Strategy, loc,
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
    cases.extend(cz_cases());
    cases.extend(anticipate_cases());
    cases
}

/// Hand-verified answers for the instances small enough to check by hand.
/// Every strategy, including Push and Rotate, must find them.
fn hand_verified(instance: &str) -> Option<Expect> {
    let layers = match instance {
        // Word 0 to word 2 on the logical arch is three word-bus hops.
        "logical_one_atom" => 3,
        // Both atoms shift word 0 -> word 1 on their own sites: one AOD shot.
        "physical_two_atoms" => 1,
        _ => return None,
    };
    Some(Expect {
        status: Some(Status::Solved),
        layers: Some(layers),
        ..Expect::default()
    })
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
            if let Some(expect) = hand_verified(instance.name) {
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
