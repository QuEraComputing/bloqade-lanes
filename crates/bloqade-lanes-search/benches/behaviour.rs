//! Local A/B timing over the behaviour-net corpus.
//!
//! The cases, the test-domain types and the interface layer are the behaviour
//! net's own files (`tests/behaviour/`), included here unchanged. So every
//! timed case is also pinned by the net's golden: an optimization that changes
//! behaviour fails `cargo test` even when it looks faster here.
//!
//! This is for comparing before and after on one machine, not for CI. Wall
//! time varies run to run and machine to machine.
//!
//! ```text
//! cargo bench -p bloqade-lanes-search --bench behaviour            # everything
//! cargo bench -p bloqade-lanes-search --bench behaviour -- route   # one group
//! cargo bench -p bloqade-lanes-search --bench behaviour -- congested
//! ```
//!
//! Each engine is built once, outside the timed loop, and warmed by an untimed
//! run: production builds one engine per architecture and reuses it, so the
//! timed number is the solve. Engine construction has its own group,
//! `engine_build`. Where a case expands nodes, the expansion count is
//! reported as divan's item counter, so the output includes expansions per
//! second.

#[allow(dead_code)]
#[path = "../tests/behaviour/spec.rs"]
mod spec;

#[allow(dead_code)]
#[path = "../tests/behaviour/cases.rs"]
mod cases;

#[allow(dead_code)]
#[path = "../tests/behaviour/interface.rs"]
mod interface;

use divan::counter::ItemsCount;
use divan::{Bencher, black_box};

use spec::{Arch, Outcome};

fn main() {
    divan::main();
}

/// Cases on the instances that do real search work. Most of the corpus
/// finishes in a handful of expansions, where timing would measure the call
/// overhead rather than the search.
fn is_timed(name: &str) -> bool {
    const HARD: [&str; 5] = [
        "logical_cycle",
        "physical_site_cycle",
        "physical_congested",
        "four_pairs",
        "zoned",
    ];
    HARD.iter().any(|h| name.contains(h)) || name.starts_with("anticipate/")
}

/// The timed case names, leaked to `'static` for divan's argument list.
fn timed(prefix: &str) -> Vec<&'static str> {
    cases::all()
        .into_iter()
        .filter(|c| c.name.starts_with(prefix) && is_timed(&c.name))
        .map(|c| &*Box::leak(c.name.into_boxed_str()))
        .collect()
}

fn time_case(bencher: Bencher, name: &str) {
    let case = cases::all()
        .into_iter()
        .find(|c| c.name == name)
        .expect("case exists");
    let engine = interface::engine_for(&case.spec).expect("engine builds");
    // Untimed warm-up: fills the engine's lazy caches and yields the count.
    let expanded = match interface::run_with(&case.spec, &engine) {
        Outcome::Ran(run) => u64::from(run.nodes_expanded),
        other => panic!("{name} does not run cleanly: {}", other.render()),
    };
    let bencher = if expanded > 0 {
        bencher.counter(ItemsCount::new(expanded))
    } else {
        bencher
    };
    bencher.bench(|| interface::run_with(black_box(&case.spec), &engine));
}

#[divan::bench(args = timed("route/"), max_time = 2)]
fn route(bencher: Bencher, name: &str) {
    time_case(bencher, name);
}

#[divan::bench(args = timed("bound/"), max_time = 2)]
fn bound(bencher: Bencher, name: &str) {
    time_case(bencher, name);
}

#[divan::bench(args = timed("knobs/"), max_time = 2)]
fn knobs(bencher: Bencher, name: &str) {
    time_case(bencher, name);
}

#[divan::bench(args = timed("cz/"), max_time = 2)]
fn cz(bencher: Bencher, name: &str) {
    time_case(bencher, name);
}

#[divan::bench(args = timed("anticipate/"), max_time = 2)]
fn anticipate(bencher: Bencher, name: &str) {
    time_case(bencher, name);
}

const ARCHES: [(&str, Arch); 9] = [
    ("gemini_logical", Arch::GeminiLogical),
    ("gemini_physical", Arch::GeminiPhysical),
    ("example", Arch::Example),
    ("chain", Arch::Chain),
    ("chain_with_siding", Arch::ChainWithSiding),
    ("two_zone_bus", Arch::TwoZoneBus),
    ("two_zone_aligned_site_bus", Arch::TwoZoneAlignedSiteBus),
    ("asymmetric_duration", Arch::AsymmetricDuration),
    ("two_zone_grid", Arch::TwoZoneGrid),
];

/// Building an engine: parsing and validating the spec, and the lane index.
#[divan::bench(args = ARCHES.map(|(name, _)| name), max_time = 2)]
fn engine_build(name: &str) {
    let (_, arch) = ARCHES
        .iter()
        .find(|(n, _)| *n == name)
        .expect("arch exists");
    black_box(interface::engine(*arch).expect("engine builds"));
}
