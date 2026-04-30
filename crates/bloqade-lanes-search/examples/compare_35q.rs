//! Quick comparison: baseline palindrome vs loose-goal IDS at 35q.
//!
//! Run with: cargo run -p bloqade-lanes-search --example compare_35q --release

use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use bloqade_lanes_search::solve::{MoveSolver, SolveOptions, SolveStatus, Strategy};
use bloqade_lanes_search::target_generator::DefaultTargetGenerator;

fn loc(word: u32, site: u32) -> LocationAddr {
    LocationAddr {
        zone_id: 0,
        word_id: word,
        site_id: site,
    }
}

/// Build a Gemini-inspired single-zone arch.
fn build_gemini_arch(n_pairs: usize, sites_per_word: usize) -> String {
    let n_words = n_pairs * 2;
    let x_spacing: Vec<f64> = (0..sites_per_word.saturating_sub(1))
        .map(|_| 10.0)
        .collect();
    let y_spacing: Vec<f64> = (0..n_words.saturating_sub(1)).map(|_| 10.0).collect();

    let words: Vec<serde_json::Value> = (0..n_words)
        .map(|w| {
            let sites: Vec<serde_json::Value> = (0..sites_per_word)
                .map(|s| serde_json::json!([s, w]))
                .collect();
            serde_json::json!({ "sites": sites })
        })
        .collect();

    let n_storage = sites_per_word / 2;
    let n_active = sites_per_word - n_storage;
    let vert_src: Vec<usize> = (0..n_storage).collect();
    let vert_dst: Vec<usize> = (n_storage..n_storage + vert_src.len().min(n_active)).collect();
    let shift_src: Vec<usize> = (0..sites_per_word - 1).collect();
    let shift_dst: Vec<usize> = (1..sites_per_word).collect();
    let mut site_buses = vec![
        serde_json::json!({ "src": vert_src, "dst": vert_dst }),
        serde_json::json!({ "src": shift_src, "dst": shift_dst }),
    ];
    if sites_per_word >= 6 {
        let s2_src: Vec<usize> = (0..sites_per_word - 2).collect();
        let s2_dst: Vec<usize> = (2..sites_per_word).collect();
        site_buses.push(serde_json::json!({ "src": s2_src, "dst": s2_dst }));
    }

    let mut word_buses: Vec<serde_json::Value> = Vec::new();
    let mut shift = 1;
    while shift < n_words {
        let wb_src: Vec<usize> = (0..n_words - shift).collect();
        let wb_dst: Vec<usize> = (shift..n_words).collect();
        word_buses.push(serde_json::json!({ "src": wb_src, "dst": wb_dst }));
        shift *= 2;
    }

    let ent_pairs: Vec<serde_json::Value> = (0..n_pairs)
        .map(|i| serde_json::json!([i, i + n_pairs]))
        .collect();

    let words_with_site_buses: Vec<usize> = (0..n_words).collect();
    let sites_with_word_buses: Vec<usize> = (0..sites_per_word).collect();

    serde_json::json!({
        "version": "2.0",
        "words": words,
        "zones": [{
            "grid": {
                "x_start": 0.0, "y_start": 0.0,
                "x_spacing": x_spacing, "y_spacing": y_spacing
            },
            "site_buses": site_buses,
            "word_buses": word_buses,
            "words_with_site_buses": words_with_site_buses,
            "sites_with_word_buses": sites_with_word_buses,
            "entangling_pairs": ent_pairs
        }],
        "zone_buses": [],
        "modes": [{ "name": "default", "zones": [0], "bitstring_order": [] }]
    })
    .to_string()
}

fn build_home(
    num_qubits: usize,
    n_word_pairs: usize,
    sites_per_word: usize,
) -> Vec<(u32, LocationAddr)> {
    let mut positions = Vec::new();
    let mut qid = 0u32;
    'outer: for site in 0..sites_per_word {
        for wp in 0..n_word_pairs {
            if qid as usize >= num_qubits {
                break 'outer;
            }
            positions.push((qid, loc(wp as u32, site as u32)));
            qid += 1;
        }
    }
    positions
}

fn make_random_layers(
    num_qubits: usize,
    n_word_pairs: usize,
    depth: usize,
    seed: u64,
) -> Vec<Vec<(u32, u32)>> {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use rand::seq::SliceRandom;

    let mut qubits_by_pair: Vec<Vec<u32>> = vec![Vec::new(); n_word_pairs];
    for qid in 0..num_qubits as u32 {
        let wp = (qid as usize) % n_word_pairs;
        qubits_by_pair[wp].push(qid);
    }

    let mut rng = SmallRng::seed_from_u64(seed);
    (0..depth)
        .map(|_| {
            let mut all_pairs: Vec<(u32, u32)> = Vec::new();
            for qubits in &qubits_by_pair {
                let mut shuffled = qubits.clone();
                shuffled.shuffle(&mut rng);
                for pair in shuffled.chunks(2) {
                    if pair.len() == 2 {
                        all_pairs.push((pair[0], pair[1]));
                    }
                }
            }
            all_pairs.shuffle(&mut rng);
            let take = (all_pairs.len() / 2).max(1);
            all_pairs.truncate(take);
            all_pairs
        })
        .collect()
}

fn main() {
    let nq = 35usize;
    let n_pairs = 5usize;
    let sites = 10usize;
    let depth = 5usize;
    let n_seeds = 3u64;

    let arch = build_gemini_arch(n_pairs, sites);
    let solver = MoveSolver::from_json(&arch).unwrap();
    let home = build_home(nq, n_pairs, sites);
    // Test multiple budgets to find the sweet spot.
    let _budgets: Vec<u32> = vec![200, 500, 1000, 3500];
    let max_exp = Some(3500u32); // default for baseline

    let _ids_opts = SolveOptions {
        strategy: Strategy::Ids,
        w_t: 0.0,
        ..SolveOptions::default()
    };

    println!(
        "=== 35q Comparison: Baseline Palindrome vs Loose-Goal IDS (depth={depth}, {n_seeds} seeds) ===\n"
    );

    struct Result {
        name: String,
        cost: f64,
        solved: usize,
        total: usize,
        time_ms: f64,
    }

    let mut results: Vec<Result> = Vec::new();

    // Baseline palindrome with varying restarts.
    for base_restarts in [1u32, 5, 10, 20] {
        let base_opts = SolveOptions {
            strategy: Strategy::Ids,
            w_t: 0.0,
            restarts: base_restarts,
            ..SolveOptions::default()
        };

        let mut total_cost = 0.0f64;
        let mut total_solved = 0usize;
        let mut total_layers = 0usize;
        let start = Instant::now();

        for seed in 1..=n_seeds {
            let layers = make_random_layers(nq, n_pairs, depth, seed);
            let mut current: Vec<(u32, LocationAddr)> = home.clone();

            for layer in &layers {
                let controls: Vec<u32> = layer.iter().map(|&(c, _)| c).collect();
                let targets: Vec<u32> = layer.iter().map(|&(_, t)| t).collect();

                let result = solver
                    .solve_with_generator(
                        current.iter().copied(),
                        std::iter::empty(),
                        &controls,
                        &targets,
                        &DefaultTargetGenerator,
                        max_exp,
                        &base_opts,
                    )
                    .unwrap();

                total_layers += 1;
                let solved = result.result.status == SolveStatus::Solved;
                if solved {
                    total_solved += 1;
                }
                let fwd_cost = result.result.cost;
                total_cost += fwd_cost * 2.0; // palindrome: return = forward
                current = home.clone();
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        results.push(Result {
            name: format!("Base(pal) r={base_restarts}"),
            cost: total_cost,
            solved: total_solved,
            total: total_layers,
            time_ms: elapsed,
        });
    }

    // Loose-goal IDS: vary budget × restarts.
    for (restarts, budget) in [
        (1u32, 100u32),
        (5, 100),
        (10, 100),
        (20, 100),
        (50, 100),
        (100, 100),
    ] {
        let loose_opts = SolveOptions {
            strategy: Strategy::Ids,
            w_t: 0.0,
            dynamic_targets: true,
            recompute_interval: 0,
            restarts,
            ..SolveOptions::default()
        };
        let max_exp = Some(budget);

        let mut total_cost = 0.0f64;
        let mut total_solved = 0usize;
        let mut total_layers = 0usize;
        let start = Instant::now();

        for seed in 1..=n_seeds {
            let layers = make_random_layers(nq, n_pairs, depth, seed);
            let mut current: Vec<(u32, LocationAddr)> = home.clone();

            for layer in &layers {
                let result = solver
                    .solve_entangling(
                        current.iter().copied(),
                        layer,
                        std::iter::empty(),
                        max_exp,
                        &loose_opts,
                        &[],
                    )
                    .unwrap();

                total_layers += 1;
                let solved = result.status == SolveStatus::Solved;
                if solved {
                    total_solved += 1;
                    current = result.goal_config.iter().collect();
                } else {
                    current = home.clone();
                }
                total_cost += result.cost;
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        results.push(Result {
            name: format!("Loose r={restarts} b={budget}"),
            cost: total_cost,
            solved: total_solved,
            total: total_layers,
            time_ms: elapsed,
        });
    }

    // Print results.
    println!(
        "{:<24} | {:>8} {:>7} {:>10} {:>8}",
        "Strategy", "Cost", "Solved", "Time(ms)", "Savings"
    );
    println!("{:─<24}─┼─{:─>8}─{:─>7}─{:─>10}─{:─>8}", "", "", "", "", "");

    let base_cost = results[0].cost;
    for r in &results {
        let savings = if base_cost > 0.0 {
            (1.0 - r.cost / base_cost) * 100.0
        } else {
            0.0
        };
        println!(
            "{:<24} | {:>8.1} {:>4}/{:<2} {:>10.0} {:>7.0}%",
            r.name, r.cost, r.solved, r.total, r.time_ms, savings,
        );
    }
}
