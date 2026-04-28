//! Multi-layer CZ benchmark: compares baseline 3-step pipeline against
//! loose-goal entangling constraint search.
//!
//! Run with `cargo test -p bloqade-lanes-search benchmark -- --nocapture`
//! to see comparison tables.

use std::time::Instant;

use bloqade_lanes_bytecode_core::arch::addr::LocationAddr;

use crate::solve::{InnerStrategy, MoveSolver, SolveOptions, SolveStatus, Strategy};
use crate::target_generator::DefaultTargetGenerator;
use crate::test_utils::loc;

// ── Types ──────────────────────────────────────────────────────────

/// A CZ layer: list of (control, target) pairs.
/// Control moves to partner word; target stays.
struct CzLayer {
    pairs: Vec<(u32, u32)>,
}

/// Per-layer statistics.
#[allow(dead_code)]
struct LayerStats {
    forward_cost: f64,
    return_cost: f64,
    total_cost: f64,
    forward_expanded: u32,
    return_expanded: u32,
    solved: bool,
}

/// Full pipeline result across all layers.
struct PipelineResult {
    layers: Vec<LayerStats>,
    total_cost: f64,
    total_expansions: u32,
    all_solved: bool,
    elapsed_ms: f64,
    failures: usize,
}

// ── Pipeline runners ───────────────────────────────────────────────

/// Baseline with palindrome return (faithful to current Python pipeline).
/// Return cost = forward cost (reversed moves).
fn run_baseline_palindrome(
    solver: &MoveSolver,
    home: &[(u32, LocationAddr)],
    cz_layers: &[CzLayer],
    opts: &SolveOptions,
    max_expansions: Option<u32>,
) -> PipelineResult {
    let start = Instant::now();
    let mut current: Vec<(u32, LocationAddr)> = home.to_vec();
    let mut layers = Vec::new();
    let mut total_cost = 0.0;
    let mut total_exp = 0u32;
    let mut all_solved = true;
    let mut failures = 0usize;

    for layer in cz_layers {
        let controls: Vec<u32> = layer.pairs.iter().map(|&(c, _)| c).collect();
        let targets: Vec<u32> = layer.pairs.iter().map(|&(_, t)| t).collect();

        let result = solver
            .solve_with_generator(
                current.iter().copied(),
                std::iter::empty(),
                &controls,
                &targets,
                &DefaultTargetGenerator,
                max_expansions,
                opts,
            )
            .unwrap();

        let solved = result.result.status == SolveStatus::Solved;
        let forward_cost = result.result.cost;
        let forward_exp = result.total_expansions;
        // Palindrome: return cost = forward cost, no extra search.
        let return_cost = forward_cost;

        layers.push(LayerStats {
            forward_cost,
            return_cost,
            total_cost: forward_cost + return_cost,
            forward_expanded: forward_exp,
            return_expanded: 0,
            solved,
        });

        total_cost += forward_cost + return_cost;
        total_exp += forward_exp;
        if !solved {
            all_solved = false;
            failures += 1;
        }

        // After palindrome return, we're back at home.
        current = home.to_vec();
    }

    PipelineResult {
        layers,
        total_cost,
        total_expansions: total_exp,
        all_solved,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        failures,
    }
}

/// Baseline with optimal solve return (best possible 3-step approach).
fn run_baseline_solve_return(
    solver: &MoveSolver,
    home: &[(u32, LocationAddr)],
    cz_layers: &[CzLayer],
    opts: &SolveOptions,
    max_expansions: Option<u32>,
) -> PipelineResult {
    let start = Instant::now();
    let mut current: Vec<(u32, LocationAddr)> = home.to_vec();
    let mut layers = Vec::new();
    let mut total_cost = 0.0;
    let mut total_exp = 0u32;
    let mut all_solved = true;
    let mut failures = 0usize;

    for layer in cz_layers {
        let controls: Vec<u32> = layer.pairs.iter().map(|&(c, _)| c).collect();
        let targets: Vec<u32> = layer.pairs.iter().map(|&(_, t)| t).collect();

        // Forward: place + route to CZ positions.
        let fwd = solver
            .solve_with_generator(
                current.iter().copied(),
                std::iter::empty(),
                &controls,
                &targets,
                &DefaultTargetGenerator,
                max_expansions,
                opts,
            )
            .unwrap();

        let fwd_solved = fwd.result.status == SolveStatus::Solved;
        let forward_cost = fwd.result.cost;
        let forward_exp = fwd.total_expansions;

        // Return: solve optimal path back to home.
        let (return_cost, return_exp, ret_solved) = if fwd_solved {
            let goal_pairs: Vec<(u32, LocationAddr)> = fwd.result.goal_config.iter().collect();
            let ret = solver
                .solve(
                    goal_pairs,
                    home.iter().copied(),
                    std::iter::empty(),
                    max_expansions,
                    opts,
                )
                .unwrap();
            (
                ret.cost,
                ret.nodes_expanded,
                ret.status == SolveStatus::Solved,
            )
        } else {
            (0.0, 0, false)
        };

        let solved = fwd_solved && ret_solved;
        layers.push(LayerStats {
            forward_cost,
            return_cost,
            total_cost: forward_cost + return_cost,
            forward_expanded: forward_exp,
            return_expanded: return_exp,
            solved,
        });

        total_cost += forward_cost + return_cost;
        total_exp += forward_exp + return_exp;
        if !solved {
            all_solved = false;
            failures += 1;
        }

        current = home.to_vec();
    }

    PipelineResult {
        layers,
        total_cost,
        total_expansions: total_exp,
        all_solved,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        failures,
    }
}

/// Loose-goal pipeline: solve_entangling, chained (no return home).
fn run_loose_goal(
    solver: &MoveSolver,
    initial: &[(u32, LocationAddr)],
    cz_layers: &[CzLayer],
    opts: &SolveOptions,
    max_expansions: Option<u32>,
    dynamic: bool,
) -> PipelineResult {
    let loose_opts = SolveOptions {
        dynamic_targets: dynamic,
        ..opts.clone()
    };

    let start = Instant::now();
    let mut current: Vec<(u32, LocationAddr)> = initial.to_vec();
    let mut layers = Vec::new();
    let mut total_cost = 0.0;
    let mut total_exp = 0u32;
    let mut all_solved = true;
    let mut failures = 0usize;

    for layer in cz_layers {
        let result = solver
            .solve_entangling(
                current.iter().copied(),
                &layer.pairs,
                std::iter::empty(),
                max_expansions,
                &loose_opts,
            )
            .unwrap();

        let solved = result.status == SolveStatus::Solved;
        layers.push(LayerStats {
            forward_cost: result.cost,
            return_cost: 0.0,
            total_cost: result.cost,
            forward_expanded: result.nodes_expanded,
            return_expanded: 0,
            solved,
        });

        total_cost += result.cost;
        total_exp += result.nodes_expanded;
        if !solved {
            all_solved = false;
            failures += 1;
        }

        // Chain: output config becomes next input.
        if solved {
            current = result.goal_config.iter().collect();
        }
    }

    PipelineResult {
        layers,
        total_cost,
        total_expansions: total_exp,
        all_solved,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        failures,
    }
}

// ── Printing ───────────────────────────────────────────────────────

fn print_comparison(label: &str, results: &[(&str, &PipelineResult)]) {
    let n_layers = results[0].1.layers.len();

    eprintln!("\n=== {label} ===\n");

    // Header.
    eprint!("{:<20}", "");
    for (name, _) in results {
        eprint!("| {:^15} ", name);
    }
    eprintln!();

    // Per-layer rows.
    for i in 0..n_layers {
        // Cost row.
        eprint!("Layer {} cost        ", i + 1);
        for (_, r) in results {
            if r.layers[i].solved {
                eprint!("| {:>13.1}  ", r.layers[i].total_cost);
            } else {
                eprint!("| {:>13}  ", "FAILED");
            }
        }
        eprintln!();

        // Expanded row.
        eprint!("Layer {} expanded    ", i + 1);
        for (_, r) in results {
            let exp = r.layers[i].forward_expanded + r.layers[i].return_expanded;
            eprint!("| {:>13}  ", exp);
        }
        eprintln!();
    }

    // Separator.
    eprint!("{:─<20}", "");
    for _ in results {
        eprint!("┼{:─<16} ", "");
    }
    eprintln!();

    // Totals.
    eprint!("{:<20}", "Total cost");
    for (_, r) in results {
        eprint!("| {:>13.1}  ", r.total_cost);
    }
    eprintln!();

    eprint!("{:<20}", "Total expanded");
    for (_, r) in results {
        eprint!("| {:>13}  ", r.total_expansions);
    }
    eprintln!();

    eprint!("{:<20}", "All solved");
    for (_, r) in results {
        eprint!("| {:>13}  ", if r.all_solved { "yes" } else { "NO" });
    }
    eprintln!("\n");
}

// ── Test scenarios ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::example_arch_json;

    fn default_opts() -> SolveOptions {
        SolveOptions {
            strategy: Strategy::Cascade {
                inner: InnerStrategy::Ids,
            },
            w_t: 0.0,
            ..SolveOptions::default()
        }
    }

    /// Scenario 1: Simple arch, single pair, 3 repeated layers.
    ///
    /// 2 qubits in column 0↔5. Same CZ pair repeated.
    /// Expected: baseline pays 2× per layer; loose-goal pays 1× first then 0.
    #[ignore]
    #[test]
    fn benchmark_simple_single_pair_3_layers() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let opts = default_opts();
        let max_exp = Some(10_000);

        // Home: both qubits on word 0 in column 0↔5.
        let home = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];

        // 3 identical CZ layers: q0 is control, q1 is target.
        let cz_layers = vec![
            CzLayer {
                pairs: vec![(0, 1)],
            },
            CzLayer {
                pairs: vec![(0, 1)],
            },
            CzLayer {
                pairs: vec![(0, 1)],
            },
        ];

        let bp = run_baseline_palindrome(&solver, &home, &cz_layers, &opts, max_exp);
        let bs = run_baseline_solve_return(&solver, &home, &cz_layers, &opts, max_exp);
        let ls = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, false);
        let ld = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, true);

        print_comparison(
            "Scenario 1: Simple arch, single pair, 3 layers (Cascade/IDS)",
            &[
                ("Base(pal)", &bp),
                ("Base(solve)", &bs),
                ("Loose(static)", &ls),
                ("Loose(dyn)", &ld),
            ],
        );

        assert!(bp.all_solved, "baseline palindrome should solve all layers");
        assert!(ls.all_solved, "loose-goal static should solve all layers");
        assert!(
            ls.total_cost <= bp.total_cost,
            "loose-goal ({}) should be ≤ baseline palindrome ({})",
            ls.total_cost,
            bp.total_cost
        );
    }

    /// Scenario 2: Simple arch, two pairs, 3 layers.
    ///
    /// 4 qubits in 2 columns. Same pairs repeated.
    #[ignore]
    #[test]
    fn benchmark_simple_two_pairs_3_layers() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let opts = default_opts();
        let max_exp = Some(20_000);

        // Home: 4 qubits on word 0 in two columns.
        // Column 0↔5: q0 at site 5, q1 at site 0
        // Column 1↔6: q2 at site 6, q3 at site 1
        let home = vec![
            (0u32, loc(0, 5)),
            (1, loc(0, 0)),
            (2, loc(0, 6)),
            (3, loc(0, 1)),
        ];

        let cz_layers = vec![
            CzLayer {
                pairs: vec![(0, 1), (2, 3)],
            },
            CzLayer {
                pairs: vec![(0, 1), (2, 3)],
            },
            CzLayer {
                pairs: vec![(0, 1), (2, 3)],
            },
        ];

        let bp = run_baseline_palindrome(&solver, &home, &cz_layers, &opts, max_exp);
        let bs = run_baseline_solve_return(&solver, &home, &cz_layers, &opts, max_exp);
        let ls = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, false);
        let ld = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, true);

        print_comparison(
            "Scenario 2: Simple arch, two pairs, 3 layers (Cascade/IDS)",
            &[
                ("Base(pal)", &bp),
                ("Base(solve)", &bs),
                ("Loose(static)", &ls),
                ("Loose(dyn)", &ld),
            ],
        );

        assert!(bp.all_solved, "baseline palindrome should solve all layers");
        assert!(ls.all_solved, "loose-goal static should solve all layers");
        assert!(
            ls.total_cost <= bp.total_cost,
            "loose-goal ({}) should be ≤ baseline palindrome ({})",
            ls.total_cost,
            bp.total_cost
        );
    }

    /// Scenario 3: Simple arch, varying pair assignments across layers.
    ///
    /// 4 qubits. Layer 1: (q0,q1) + (q2,q3). Layer 2: (q0,q3) + (q2,q1).
    /// This tests Strategy 2's chaining advantage — it can pick CZ positions
    /// that are convenient for the next layer.
    ///
    /// Note: With the isolated site columns, q0/q1 must share a column and
    /// q2/q3 must share a column. Swapping partners across columns isn't
    /// possible. So we keep pairs within columns but swap control/target roles.
    #[ignore]
    #[test]
    fn benchmark_simple_varying_pairs() {
        let solver = MoveSolver::from_json(example_arch_json()).unwrap();
        let opts = default_opts();
        let max_exp = Some(20_000);

        let home = vec![
            (0u32, loc(0, 5)),
            (1, loc(0, 0)),
            (2, loc(0, 6)),
            (3, loc(0, 1)),
        ];

        // Swap control/target roles between layers.
        let cz_layers = vec![
            CzLayer {
                pairs: vec![(0, 1), (2, 3)],
            },
            CzLayer {
                pairs: vec![(1, 0), (3, 2)], // roles swapped
            },
            CzLayer {
                pairs: vec![(0, 1), (2, 3)],
            },
        ];

        let bp = run_baseline_palindrome(&solver, &home, &cz_layers, &opts, max_exp);
        let bs = run_baseline_solve_return(&solver, &home, &cz_layers, &opts, max_exp);
        let ls = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, false);
        let ld = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, true);

        print_comparison(
            "Scenario 3: Simple arch, varying pairs, 3 layers (Cascade/IDS)",
            &[
                ("Base(pal)", &bp),
                ("Base(solve)", &bs),
                ("Loose(static)", &ls),
                ("Loose(dyn)", &ld),
            ],
        );

        // Note: baseline may FAIL on swapped layers because the control
        // qubit's path passes through the target qubit's occupied position.
        // Strategy 2 avoids this by choosing flexible CZ placements.
        assert!(ls.all_solved, "loose-goal static should solve all layers");
    }

    /// Scenario 4: Full arch (3 words, 2 zones), single pair, 3 layers.
    #[ignore]
    #[test]
    fn benchmark_full_arch() {
        use crate::test_utils::full_arch_json;

        let solver = MoveSolver::from_json(full_arch_json()).unwrap();
        let opts = default_opts();
        let max_exp = Some(10_000);

        // Home: qubits on word 0 in column 0↔5 (zone 0 has pair [0,1]).
        let home = vec![(0u32, loc(0, 5)), (1, loc(0, 0))];

        let cz_layers = vec![
            CzLayer {
                pairs: vec![(0, 1)],
            },
            CzLayer {
                pairs: vec![(0, 1)],
            },
            CzLayer {
                pairs: vec![(0, 1)],
            },
        ];

        let bp = run_baseline_palindrome(&solver, &home, &cz_layers, &opts, max_exp);
        let bs = run_baseline_solve_return(&solver, &home, &cz_layers, &opts, max_exp);
        let ls = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, false);
        let ld = run_loose_goal(&solver, &home, &cz_layers, &opts, max_exp, true);

        print_comparison(
            "Scenario 4: Full arch (3 words, 2 zones), 3 layers (Cascade/IDS)",
            &[
                ("Base(pal)", &bp),
                ("Base(solve)", &bs),
                ("Loose(static)", &ls),
                ("Loose(dyn)", &ld),
            ],
        );

        assert!(bp.all_solved, "baseline palindrome should solve all layers");
        // On the full arch, static loose-goal may fail if the greedy
        // assignment creates blocking; dynamic loose-goal adapts targets.
        assert!(ld.all_solved, "loose-goal dynamic should solve all layers");
        assert!(
            ld.total_cost <= bp.total_cost,
            "loose-goal dynamic ({}) should be ≤ baseline palindrome ({})",
            ld.total_cost,
            bp.total_cost
        );
    }

    // ── Extensive benchmarks ───────────────────────────────────────

    // ── Architecture builders ──

    /// Build the Gemini full physical architecture.
    ///
    /// 3 zones: storage_top → entangling → storage_bottom.
    /// 10 words per zone (5 rows × 2 cols), 17 sites per word.
    /// Entangling zone has hypercube site buses, diagonal word buses,
    /// and 5 entangling pairs. Storage zones have no buses.
    /// Zone buses connect storage ↔ entangling with matching topology.
    fn build_gemini_full_arch() -> String {
        let n_sites = 17usize;
        let n_rows = 5usize;
        let n_cols = 2usize;
        let n_words_per_zone = n_rows * n_cols; // 10
        let n_words = n_words_per_zone * 3; // 30

        // Grid: 17 x-positions (site spacing 10.0), n_words y-positions.
        let x_spacing: Vec<f64> = (0..n_sites - 1).map(|_| 10.0).collect();
        let y_spacing: Vec<f64> = (0..n_words - 1)
            .map(|i| {
                // Gap between zones.
                if i == n_words_per_zone - 1 || i == 2 * n_words_per_zone - 1 {
                    20.0
                } else {
                    20.0
                }
            })
            .collect();

        // Words: all 30 words, each with 17 sites.
        let words: Vec<serde_json::Value> = (0..n_words)
            .map(|w| {
                let sites: Vec<serde_json::Value> =
                    (0..n_sites).map(|s| serde_json::json!([s, w])).collect();
                serde_json::json!({ "sites": sites })
            })
            .collect();

        // ── Hypercube site buses for entangling zone (N=17, 5 dims) ──
        let site_buses: Vec<serde_json::Value> = {
            let mut buses = Vec::new();
            let mut dim = 0u32;
            let mut stride = 1usize;
            while stride < 32 {
                // round up to next power of 2
                let mut src = Vec::new();
                let mut dst = Vec::new();
                for s in 0..n_sites {
                    // s has bit `dim` clear, partner has bit `dim` set.
                    if s & stride == 0 {
                        let partner = s | stride;
                        if partner < n_sites {
                            src.push(s);
                            dst.push(partner);
                        }
                    }
                }
                if !src.is_empty() {
                    buses.push(serde_json::json!({ "src": src, "dst": dst }));
                }
                dim += 1;
                stride <<= 1;
            }
            buses
        };

        // ── Diagonal word buses for entangling zone (5×2 grid) ──
        // Global word IDs for entangling zone: 10-19.
        // col 0: words 10,12,14,16,18; col 1: words 11,13,15,17,19
        let ent_offset = n_words_per_zone; // 10
        let word_buses: Vec<serde_json::Value> = {
            let mut buses = Vec::new();
            let col0: Vec<usize> = (0..n_rows).map(|r| ent_offset + r * n_cols).collect();
            let col1: Vec<usize> = (0..n_rows).map(|r| ent_offset + r * n_cols + 1).collect();

            // Forward diagonals: shift 0 to n_rows-1.
            for shift in 0..n_rows {
                let mut src = Vec::new();
                let mut dst = Vec::new();
                for r in 0..n_rows {
                    if r + shift < n_rows {
                        src.push(col0[r]);
                        dst.push(col1[r + shift]);
                    }
                }
                if !src.is_empty() {
                    buses.push(serde_json::json!({ "src": src, "dst": dst }));
                }
            }
            // Reverse diagonals: shift 1 to n_rows-1.
            for shift in 1..n_rows {
                let mut src = Vec::new();
                let mut dst = Vec::new();
                for r in shift..n_rows {
                    src.push(col0[r]);
                    dst.push(col1[r - shift]);
                }
                if !src.is_empty() {
                    buses.push(serde_json::json!({ "src": src, "dst": dst }));
                }
            }
            buses
        };

        // ── Entangling pairs: row-wise pairing ──
        let ent_pairs: Vec<serde_json::Value> = (0..n_rows)
            .map(|r| {
                let w0 = ent_offset + r * n_cols;
                let w1 = ent_offset + r * n_cols + 1;
                serde_json::json!([w0, w1])
            })
            .collect();

        let all_words: Vec<usize> = (0..n_words).collect();
        let all_sites: Vec<usize> = (0..n_sites).collect();

        // ── Zone buses: matching topology ──
        // storage_top (0-9) ↔ entangling (10-19)
        let zb0_src: Vec<serde_json::Value> = (0..n_words_per_zone)
            .map(|i| serde_json::json!({"zone_id": 0, "word_id": i}))
            .collect();
        let zb0_dst: Vec<serde_json::Value> = (0..n_words_per_zone)
            .map(|i| serde_json::json!({"zone_id": 1, "word_id": ent_offset + i}))
            .collect();
        // entangling (10-19) ↔ storage_bottom (20-29)
        let zb1_src: Vec<serde_json::Value> = (0..n_words_per_zone)
            .map(|i| serde_json::json!({"zone_id": 1, "word_id": ent_offset + i}))
            .collect();
        let zb1_dst: Vec<serde_json::Value> = (0..n_words_per_zone)
            .map(|i| serde_json::json!({"zone_id": 2, "word_id": 2 * n_words_per_zone + i}))
            .collect();

        // Build the 3 zones. Storage zones have no buses.
        let make_grid = |y_start: f64| {
            serde_json::json!({
                "x_start": 0.0, "y_start": y_start,
                "x_spacing": x_spacing, "y_spacing": y_spacing
            })
        };

        serde_json::json!({
            "version": "2.0",
            "words": words,
            "zones": [
                {
                    "name": "storage_top",
                    "grid": make_grid(0.0),
                    "site_buses": [],
                    "word_buses": [],
                    "words_with_site_buses": [],
                    "sites_with_word_buses": [],
                    "entangling_pairs": []
                },
                {
                    "name": "entangling",
                    "grid": make_grid(0.0),
                    "site_buses": site_buses,
                    "word_buses": word_buses,
                    "words_with_site_buses": all_words,
                    "sites_with_word_buses": all_sites,
                    "entangling_pairs": ent_pairs
                },
                {
                    "name": "storage_bottom",
                    "grid": make_grid(0.0),
                    "site_buses": [],
                    "word_buses": [],
                    "words_with_site_buses": [],
                    "sites_with_word_buses": [],
                    "entangling_pairs": []
                }
            ],
            "zone_buses": [
                { "src": zb0_src, "dst": zb0_dst },
                { "src": zb1_src, "dst": zb1_dst }
            ],
            "modes": [{ "name": "default", "zones": [0, 1, 2], "bitstring_order": [] }],
            "feed_forward": true,
            "atom_reloading": true,
            "blockade_radius": 10.0
        })
        .to_string()
    }

    /// Build a Gemini-inspired arch with `n_pairs` entangling word pairs
    /// and `sites_per_word` sites. Single zone.
    fn build_gemini_arch(n_pairs: usize, sites_per_word: usize) -> String {
        let n_words = n_pairs * 2; // home words + partner words
        let n_storage = sites_per_word / 2;
        let n_active = sites_per_word - n_storage;

        // Grid: enough x-positions for all sites, 1 y-position per word row.
        let x_spacing: Vec<f64> = (0..sites_per_word.saturating_sub(1))
            .map(|_| 10.0)
            .collect();
        let y_spacing: Vec<f64> = (0..n_words.saturating_sub(1)).map(|_| 10.0).collect();

        // Words: each word's sites index into the grid.
        let words: Vec<serde_json::Value> = (0..n_words)
            .map(|w| {
                let sites: Vec<serde_json::Value> = (0..sites_per_word)
                    .map(|s| serde_json::json!([s, w]))
                    .collect();
                serde_json::json!({ "sites": sites })
            })
            .collect();

        // Site buses:
        // 1. Vertical: storage → active (site k → site k + n_storage)
        // 2. Lateral shift right (adjacent sites can reach each other)
        let vert_src: Vec<usize> = (0..n_storage).collect();
        let vert_dst: Vec<usize> = (n_storage..n_storage + vert_src.len().min(n_active)).collect();
        let shift_src: Vec<usize> = (0..sites_per_word - 1).collect();
        let shift_dst: Vec<usize> = (1..sites_per_word).collect();

        let mut site_buses = vec![
            serde_json::json!({ "src": vert_src, "dst": vert_dst }),
            serde_json::json!({ "src": shift_src, "dst": shift_dst }),
        ];
        // 3. For larger archs, add a stride-2 bus for faster lateral movement.
        if sites_per_word >= 6 {
            let s2_src: Vec<usize> = (0..sites_per_word - 2).collect();
            let s2_dst: Vec<usize> = (2..sites_per_word).collect();
            site_buses.push(serde_json::json!({ "src": s2_src, "dst": s2_dst }));
        }

        // Word buses: hypercube connectivity (shift 1, 2, 4, 8, ...).
        // Every word can reach every other in O(log n) hops.
        let mut word_buses: Vec<serde_json::Value> = Vec::new();
        let mut shift = 1;
        while shift < n_words {
            let wb_src: Vec<usize> = (0..n_words - shift).collect();
            let wb_dst: Vec<usize> = (shift..n_words).collect();
            word_buses.push(serde_json::json!({ "src": wb_src, "dst": wb_dst }));
            shift *= 2;
        }

        // Entangling pairs.
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

    /// Build a two-zone Gemini-inspired arch.
    /// Each zone has `n_pairs_per_zone` entangling pairs.
    fn build_two_zone_arch(n_pairs_per_zone: usize, sites_per_word: usize) -> String {
        let n_words = n_pairs_per_zone * 4; // 2 zones × (home + partner)
        let n_storage = sites_per_word / 2;
        let n_active = sites_per_word - n_storage;

        let x_spacing: Vec<f64> = (0..sites_per_word.saturating_sub(1))
            .map(|_| 10.0)
            .collect();
        // Y: rows for zone 0 words, gap, rows for zone 1 words.
        let half = n_pairs_per_zone * 2;
        let mut y_spacing: Vec<f64> = Vec::new();
        for _ in 0..half.saturating_sub(1) {
            y_spacing.push(10.0);
        }
        y_spacing.push(100.0); // gap between zones
        for _ in 0..half.saturating_sub(1) {
            y_spacing.push(10.0);
        }

        let words: Vec<serde_json::Value> = (0..n_words)
            .map(|w| {
                let sites: Vec<serde_json::Value> = (0..sites_per_word)
                    .map(|s| serde_json::json!([s, w]))
                    .collect();
                serde_json::json!({ "sites": sites })
            })
            .collect();

        // Site buses: vertical + lateral shift (same as build_gemini_arch).
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

        let words_with_site_buses: Vec<usize> = (0..n_words).collect();
        let sites_with_word_buses: Vec<usize> = (0..sites_per_word).collect();

        // Zone 0: words 0..half, hypercube word buses within zone.
        let mut z0_word_buses: Vec<serde_json::Value> = Vec::new();
        {
            let mut shift = 1;
            while shift < half {
                let src: Vec<usize> = (0..half - shift).collect();
                let dst: Vec<usize> = (shift..half).collect();
                z0_word_buses.push(serde_json::json!({ "src": src, "dst": dst }));
                shift *= 2;
            }
        }
        let z0_pairs: Vec<serde_json::Value> = (0..n_pairs_per_zone)
            .map(|i| serde_json::json!([i, i + n_pairs_per_zone]))
            .collect();

        // Zone 1: words half..n_words, hypercube word buses within zone.
        let mut z1_word_buses: Vec<serde_json::Value> = Vec::new();
        {
            let mut shift = 1;
            while shift < half {
                let src: Vec<usize> = (half..n_words - shift).collect();
                let dst: Vec<usize> = (half + shift..n_words).collect();
                z1_word_buses.push(serde_json::json!({ "src": src, "dst": dst }));
                shift *= 2;
            }
        }
        let z1_pairs: Vec<serde_json::Value> = (0..n_pairs_per_zone)
            .map(|i| serde_json::json!([half + i, half + n_pairs_per_zone + i]))
            .collect();

        // Zone buses: connect home words across zones.
        let zb_src: Vec<serde_json::Value> = (0..n_pairs_per_zone)
            .map(|i| serde_json::json!({"zone_id": 0, "word_id": i}))
            .collect();
        let zb_dst: Vec<serde_json::Value> = (0..n_pairs_per_zone)
            .map(|i| serde_json::json!({"zone_id": 1, "word_id": half + i}))
            .collect();

        let make_zone =
            |grid_y_start: f64, word_buses: &[serde_json::Value], pairs: &[serde_json::Value]| {
                serde_json::json!({
                    "grid": {
                        "x_start": 0.0, "y_start": grid_y_start,
                        "x_spacing": x_spacing, "y_spacing": y_spacing
                    },
                    "site_buses": site_buses,
                    "word_buses": word_buses,
                    "words_with_site_buses": words_with_site_buses,
                    "sites_with_word_buses": sites_with_word_buses,
                    "entangling_pairs": pairs
                })
            };

        serde_json::json!({
            "version": "2.0",
            "words": words,
            "zones": [
                make_zone(0.0, &z0_word_buses, &z0_pairs),
                make_zone(1000.0, &z1_word_buses, &z1_pairs),
            ],
            "zone_buses": [{ "src": zb_src, "dst": zb_dst }],
            "modes": [{ "name": "default", "zones": [0, 1], "bitstring_order": [] }]
        })
        .to_string()
    }

    // ── Home positions and circuit generators ──
    //
    // All qubits on HOME words only (lower word in each entangling pair).
    // Partner words start empty.
    // With lateral site buses, qubits can move between any sites on the
    // same word. CZ pairs are between qubits on the SAME home word at
    // DIFFERENT sites. The control moves laterally to the target's site,
    // then word-bus to the partner word. Or the solver finds a better route.
    //
    // Baseline cost per CZ pair: ~2 hops (lateral + word bus) + palindrome return.
    // Loose-goal: flexible target selection, no mandatory return.

    /// Place all qubits on home words only, one per site.
    /// Distributes across home words round-robin, then across sites.
    ///
    /// Max capacity: n_word_pairs × sites_per_word
    fn build_home(
        num_qubits: usize,
        n_word_pairs: usize,
        sites_per_word: usize,
        zone_id: u32,
    ) -> Vec<(u32, LocationAddr)> {
        let mut positions = Vec::new();
        let mut qid = 0u32;

        'outer: for site in 0..sites_per_word {
            for wp in 0..n_word_pairs {
                if qid as usize >= num_qubits {
                    break 'outer;
                }
                positions.push((
                    qid,
                    LocationAddr {
                        zone_id,
                        word_id: wp as u32,
                        site_id: site as u32,
                    },
                ));
                qid += 1;
            }
        }
        positions
    }

    /// Generate random CZ layers with cross-site pairs within word pairs.
    ///
    /// Pairs are between qubits that share an entangling word pair but are
    /// at RANDOM sites (not adjacent). Each layer shuffles qubits within
    /// each word pair and pairs them, then takes a random subset.
    ///
    /// This creates site competition: multiple pairs on the same word pair
    /// compete for which entangling site to use.
    fn make_random_layers(
        num_qubits: usize,
        n_word_pairs: usize,
        depth: usize,
        seed: u64,
    ) -> Vec<CzLayer> {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;
        use rand::seq::SliceRandom;

        // Build qubit-to-word-pair mapping from build_home layout:
        // qid = site * n_word_pairs + wp, so wp = qid % n_word_pairs
        let mut qubits_by_pair: Vec<Vec<u32>> = vec![Vec::new(); n_word_pairs];
        for qid in 0..num_qubits as u32 {
            let wp = (qid as usize) % n_word_pairs;
            qubits_by_pair[wp].push(qid);
        }

        let mut rng = SmallRng::seed_from_u64(seed);

        (0..depth)
            .map(|_| {
                let mut all_pairs: Vec<(u32, u32)> = Vec::new();

                // Within each word pair, shuffle qubits and pair them.
                for qubits in &qubits_by_pair {
                    let mut shuffled = qubits.clone();
                    shuffled.shuffle(&mut rng);
                    for pair in shuffled.chunks(2) {
                        if pair.len() == 2 {
                            all_pairs.push((pair[0], pair[1]));
                        }
                    }
                }

                // Take a random subset (half to all).
                all_pairs.shuffle(&mut rng);
                let take = (all_pairs.len() / 2).max(1);
                all_pairs.truncate(take);
                CzLayer { pairs: all_pairs }
            })
            .collect()
    }

    // ── Summary printer ──

    fn print_summary(label: &str, results: &[(&str, &PipelineResult)], n_layers: usize) {
        eprintln!("\n=== {label} ===\n");
        eprintln!(
            "{:<18} | {:>8} | {:>10} | {:>10} | {:>10} | {:>10}",
            "Approach", "Cost", "Expanded", "Time(ms)", "Failures", "Cost/layer"
        );
        eprintln!(
            "{:─<18}─┼─{:─>8}─┼─{:─>10}─┼─{:─>10}─┼─{:─>10}─┼─{:─>10}",
            "", "", "", "", "", ""
        );
        for (name, r) in results {
            let cost_per_layer = if n_layers > 0 {
                r.total_cost / n_layers as f64
            } else {
                0.0
            };
            eprintln!(
                "{:<18} | {:>8.1} | {:>10} | {:>10.1} | {:>7}/{:<2} | {:>10.1}",
                name,
                r.total_cost,
                r.total_expansions,
                r.elapsed_ms,
                r.failures,
                n_layers,
                cost_per_layer,
            );
        }
        eprintln!();
    }

    /// Run a full 4-way comparison and print summary.
    fn run_comparison(
        label: &str,
        arch_json: &str,
        home: &[(u32, LocationAddr)],
        cz_layers: &[CzLayer],
        opts: &SolveOptions,
        max_exp: Option<u32>,
    ) {
        let solver = MoveSolver::from_json(arch_json).unwrap();
        let depth = cz_layers.len();

        let bp = run_baseline_palindrome(&solver, home, cz_layers, opts, max_exp);
        let bs = run_baseline_solve_return(&solver, home, cz_layers, opts, max_exp);
        let ls = run_loose_goal(&solver, home, cz_layers, opts, max_exp, false);
        let ld = run_loose_goal(&solver, home, cz_layers, opts, max_exp, true);

        print_summary(
            label,
            &[
                ("Base(palindrome)", &bp),
                ("Base(solve)", &bs),
                ("Loose(static)", &ls),
                ("Loose(dynamic)", &ld),
            ],
            depth,
        );
    }

    // ── Scaling benchmarks ──

    #[test]
    fn benchmark_scaling_small_4q() {
        let arch = build_gemini_arch(2, 10);
        let home = build_home(4, 2, 10, 0);
        let layers = make_random_layers(4, 2, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Small arch, 4q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(20_000),
        );
    }

    #[test]
    fn benchmark_scaling_small_10q() {
        let arch = build_gemini_arch(2, 10);
        let home = build_home(10, 2, 10, 0);
        let layers = make_random_layers(10, 2, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Small arch, 10q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(50_000),
        );
    }

    #[test]
    fn benchmark_scaling_medium_20q() {
        let arch = build_gemini_arch(5, 10);
        let home = build_home(20, 5, 10, 0);
        let layers = make_random_layers(20, 5, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Medium arch, 20q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(100_000),
        );
    }

    #[ignore]
    #[test]
    fn benchmark_scaling_medium_50q() {
        let arch = build_gemini_arch(5, 10);
        let home = build_home(50, 5, 10, 0);
        let layers = make_random_layers(50, 5, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Medium arch, 50q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(250_000),
        );
    }

    #[ignore]
    #[test]
    fn benchmark_scaling_large_50q() {
        let arch = build_two_zone_arch(5, 20);
        let home = build_home(50, 5, 20, 0);
        let layers = make_random_layers(50, 5, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Large arch (2-zone), 50q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(250_000),
        );
    }

    #[ignore]
    #[test]
    fn benchmark_scaling_large_100q() {
        let arch = build_two_zone_arch(5, 20);
        let home = build_home(100, 5, 20, 0);
        let layers = make_random_layers(100, 5, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Large arch (2-zone), 100q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(500_000),
        );
    }

    #[ignore]
    #[test]
    fn benchmark_scaling_large_200q() {
        let arch = build_two_zone_arch(10, 20);
        let home = build_home(200, 10, 20, 0);
        let layers = make_random_layers(200, 10, 10, 1);
        let opts = default_opts();
        run_comparison(
            "Large arch (2-zone), 200q, alternating, depth=10",
            &arch,
            &home,
            &layers,
            &opts,
            Some(1_000_000),
        );
    }

    // ── Random circuit benchmarks ──

    #[ignore]
    #[test]
    fn benchmark_random_medium_20q() {
        let arch = build_gemini_arch(5, 10);
        let home = build_home(20, 5, 10, 0);
        let opts = default_opts();
        eprintln!("\n=== Random circuits: Medium arch, 20q, depth=10, 5 seeds ===\n");
        for seed in 1..=5u64 {
            let layers = make_random_layers(20, 5, 10, seed);
            run_comparison(
                &format!("Medium 20q random seed={seed}"),
                &arch,
                &home,
                &layers,
                &opts,
                Some(100_000),
            );
        }
    }

    #[ignore]
    #[test]
    fn benchmark_random_medium_50q() {
        let arch = build_gemini_arch(5, 10);
        let home = build_home(50, 5, 10, 0);
        let opts = default_opts();
        eprintln!("\n=== Random circuits: Medium arch, 50q, depth=10, 5 seeds ===\n");
        for seed in 1..=5u64 {
            let layers = make_random_layers(50, 5, 10, seed);
            run_comparison(
                &format!("Medium 50q random seed={seed}"),
                &arch,
                &home,
                &layers,
                &opts,
                Some(250_000),
            );
        }
    }

    #[ignore]
    #[test]
    fn benchmark_random_large_100q() {
        let arch = build_two_zone_arch(5, 20);
        let home = build_home(100, 5, 20, 0);
        let opts = default_opts();
        eprintln!("\n=== Random circuits: Large arch, 100q, depth=10, 5 seeds ===\n");
        for seed in 1..=5u64 {
            let layers = make_random_layers(100, 5, 10, seed);
            run_comparison(
                &format!("Large 100q random seed={seed}"),
                &arch,
                &home,
                &layers,
                &opts,
                Some(500_000),
            );
        }
    }

    #[ignore]
    #[test]
    fn benchmark_random_large_200q() {
        let arch = build_two_zone_arch(10, 20);
        let home = build_home(200, 10, 20, 0);
        let opts = default_opts();
        eprintln!("\n=== Random circuits: Large arch, 200q, depth=10, 5 seeds ===\n");
        for seed in 1..=5u64 {
            let layers = make_random_layers(200, 10, 10, seed);
            run_comparison(
                &format!("Large 200q random seed={seed}"),
                &arch,
                &home,
                &layers,
                &opts,
                Some(1_000_000),
            );
        }
    }

    // ── CSV output for large-scale benchmark ──────────────────────

    /// Result for one (qubits, seed, approach) — a named pipeline result.
    struct ApproachResult {
        name: String,
        result: PipelineResult,
    }

    /// Result of one (qubits, seed) benchmark point.
    struct BenchPoint {
        num_qubits: usize,
        seed: u64,
        depth: usize,
        approaches: Vec<ApproachResult>,
    }

    /// Name for a recompute interval.
    fn interval_name(interval: u32) -> String {
        match interval {
            0 => "loose_dyn_deadlock".to_string(),
            1 => "loose_dyn_1".to_string(),
            n => format!("loose_dyn_{n}"),
        }
    }

    /// Run one benchmark point: baseline + static + enabled dynamic intervals.
    /// Each dynamic variant has its own time limit.
    fn run_point(
        solver: &MoveSolver,
        home: &[(u32, LocationAddr)],
        num_qubits: usize,
        seed: u64,
        cz_layers: &[CzLayer],
        opts: &SolveOptions,
        max_exp: Option<u32>,
        per_variant_time_limit_ms: f64,
        enabled_intervals: &[u32],
    ) -> BenchPoint {
        let depth = cz_layers.len();
        let mut approaches = Vec::new();

        // Baseline (palindrome).
        approaches.push(ApproachResult {
            name: "baseline_palindrome".to_string(),
            result: run_baseline_palindrome(solver, home, cz_layers, opts, max_exp),
        });

        // Loose-goal static.
        let loose_static = run_loose_goal(solver, home, cz_layers, opts, max_exp, false);
        approaches.push(ApproachResult {
            name: "loose_static".to_string(),
            result: loose_static,
        });

        // Loose-goal dynamic: each enabled interval runs independently
        // with its own time limit.
        for &interval in enabled_intervals {
            let dyn_opts = SolveOptions {
                dynamic_targets: true,
                recompute_interval: interval,
                ..opts.clone()
            };
            let result = run_loose_goal(solver, home, cz_layers, &dyn_opts, max_exp, true);
            approaches.push(ApproachResult {
                name: interval_name(interval),
                result,
            });
        }

        BenchPoint {
            num_qubits,
            seed,
            depth,
            approaches,
        }
    }

    /// Write a BenchPoint to CSV.
    fn write_point(writer: &mut impl std::io::Write, p: &BenchPoint) {
        for a in &p.approaches {
            let cpl = if p.depth > 0 {
                a.result.total_cost / p.depth as f64
            } else {
                0.0
            };
            writeln!(
                writer,
                "{},{},{},{:.1},{},{:.2},{},{:.2}",
                p.num_qubits,
                p.seed,
                a.name,
                a.result.total_cost,
                a.result.total_expansions,
                a.result.elapsed_ms,
                a.result.failures,
                cpl,
            )
            .unwrap();
        }
    }

    /// Large-scale random benchmark with parallel execution.
    ///
    /// Approaches: baseline (palindrome), loose-goal static, and loose-goal
    /// dynamic with multiple recompute intervals (1, 5, 10, 50).
    /// Dynamic variants are skipped if static takes > 500ms.
    ///
    /// Seeds run in parallel via rayon. Results flushed to CSV after each batch.
    /// Run with: `cargo test -p bloqade-lanes-search benchmark_sweep_random -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn benchmark_sweep_random() {
        use rayon::prelude::*;
        use std::io::Write;

        let csv_path = "benchmark_results.csv";
        let mut file = std::fs::File::create(csv_path).expect("cannot create CSV file");
        writeln!(
            file,
            "qubits,seed,approach,total_cost,total_expanded,time_ms,failures,cost_per_layer"
        )
        .unwrap();
        file.flush().unwrap();

        let opts = default_opts();
        let depth = 5;
        let n_seeds = 3u64;
        let per_variant_time_limit_ms = 2000.0; // 2s per variant per point
        let recompute_intervals: Vec<u32> = vec![0, 50, 10, 1];

        // Helper: run one arch config with per-variant timeout tracking.
        // If a variant times out or fails at qubit count N, it's disabled
        // for all larger counts on this arch.
        let run_arch = |file: &mut std::fs::File,
                        label: &str,
                        _arch: &str,
                        solver: &MoveSolver,
                        n_pairs: usize,
                        sites: usize,
                        qubit_counts: &[usize]| {
            eprintln!("\n--- {label}: {n_pairs} pairs, {sites} sites ---");
            let mut enabled: Vec<u32> = recompute_intervals.clone();

            for &nq in qubit_counts {
                let home = build_home(nq, n_pairs, sites, 0);
                let max_exp = Some((5000 * nq) as u32);
                let enabled_snapshot = enabled.clone();

                let points: Vec<BenchPoint> = (1..=n_seeds)
                    .into_par_iter()
                    .map(|seed| {
                        let layers = make_random_layers(nq, n_pairs, depth, seed);
                        run_point(
                            solver,
                            &home,
                            nq,
                            seed,
                            &layers,
                            &opts,
                            max_exp,
                            per_variant_time_limit_ms,
                            &enabled_snapshot,
                        )
                    })
                    .collect();

                // Disable variants that timed out or failed on any seed.
                for &interval in &enabled_snapshot {
                    let name = interval_name(interval);
                    let should_disable = points.iter().any(|p| {
                        p.approaches.iter().any(|a| {
                            a.name == name
                                && (a.result.elapsed_ms > per_variant_time_limit_ms
                                    || !a.result.all_solved)
                        })
                    });
                    if should_disable {
                        enabled.retain(|&i| i != interval);
                        eprintln!("  [!] disabled {} for larger qubit counts", name);
                    }
                }

                for p in &points {
                    write_point(file, p);
                    let summary: Vec<String> = p
                        .approaches
                        .iter()
                        .map(|a| {
                            let short = a
                                .name
                                .replace("baseline_palindrome", "base")
                                .replace("loose_static", "static")
                                .replace("loose_dyn_", "dyn");
                            format!("{}={:.0}", short, a.result.total_cost)
                        })
                        .collect();
                    eprintln!(
                        "  {:>3}q seed={}: {}",
                        p.num_qubits,
                        p.seed,
                        summary.join(" ")
                    );
                }
                file.flush().unwrap();
            }
        };

        // Single-zone architectures.
        let configs: Vec<(&str, usize, usize, Vec<usize>)> =
            vec![("small", 2, 10, vec![4, 10]), ("medium", 5, 10, vec![10])];

        for (label, n_pairs, sites, qubit_counts) in &configs {
            let arch = build_gemini_arch(*n_pairs, *sites);
            let solver = MoveSolver::from_json(&arch).unwrap();
            run_arch(
                &mut file,
                label,
                &arch,
                &solver,
                *n_pairs,
                *sites,
                qubit_counts,
            );
        }

        // Gemini full physical arch.
        {
            let arch = build_gemini_full_arch();
            let solver = MoveSolver::from_json(&arch).unwrap();
            let gemini_qubit_counts: Vec<usize> = vec![10];
            run_arch(
                &mut file,
                "gemini_full",
                &arch,
                &solver,
                10,
                17,
                &gemini_qubit_counts,
            );
        }

        eprintln!("\nResults written to {csv_path}");
    }

    /// Diagnostic: compare strategy combinations on the same layers.
    #[ignore]
    #[test]
    fn diag_strategy_comparison() {
        use crate::generators::heuristic::DeadlockPolicy;

        let arch = build_gemini_arch(5, 10);
        let solver = MoveSolver::from_json(&arch).unwrap();
        let home = build_home(10, 5, 10, 0);

        // Test across multiple seeds for robustness.
        let n_seeds = 3;
        let depth = 5;
        let max_exp = Some(50_000u32);

        // Strategy combinations to test.
        struct StrategyConfig {
            name: &'static str,
            strategy: Strategy,
            restarts: u32,
            deadlock_policy: DeadlockPolicy,
            weight: f64,
        }

        let strategies = vec![
            StrategyConfig {
                name: "Cascade/IDS r=1 Skip tc=3",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 1,
                deadlock_policy: DeadlockPolicy::Skip,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=1 MoveBlockers tc=3",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 1,
                deadlock_policy: DeadlockPolicy::MoveBlockers,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=1 AllMoves tc=3",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 1,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=3 AllMoves tc=3",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 3,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=5 AllMoves tc=3",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 5,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=1 AllMoves tc=5",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 1,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=3 AllMoves tc=5",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 3,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=1 Skip tc=5",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 1,
                deadlock_policy: DeadlockPolicy::Skip,
                weight: 1.0,
            },
            StrategyConfig {
                name: "Cascade/IDS r=3 MoveBlockers tc=5",
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                restarts: 3,
                deadlock_policy: DeadlockPolicy::MoveBlockers,
                weight: 1.0,
            },
            StrategyConfig {
                name: "A* w=2.0 r=1 AllMoves tc=3",
                strategy: Strategy::AStar,
                restarts: 1,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 2.0,
            },
            StrategyConfig {
                name: "IDS r=3 AllMoves tc=3",
                strategy: Strategy::Ids,
                restarts: 3,
                deadlock_policy: DeadlockPolicy::AllMoves,
                weight: 1.0,
            },
        ];

        eprintln!(
            "\n=== Strategy comparison: 10q, medium arch, {} layers, {} seeds ===\n",
            depth, n_seeds
        );
        eprintln!(
            "{:<40} | {:>6} | {:>8} | {:>8} | {:>8}",
            "Strategy", "Solved", "Cost", "Expanded", "Time(ms)"
        );
        eprintln!(
            "{:─<40}─┼─{:─>6}─┼─{:─>8}─┼─{:─>8}─┼─{:─>8}",
            "", "", "", "", ""
        );

        for sc in &strategies {
            let opts = SolveOptions {
                strategy: sc.strategy,
                w_t: 0.0,
                restarts: sc.restarts,
                deadlock_policy: sc.deadlock_policy,
                weight: sc.weight,
                ..SolveOptions::default()
            };

            let mut total_solved = 0usize;
            let mut total_layers = 0usize;
            let mut total_cost = 0.0f64;
            let mut total_expanded = 0u32;
            let start = Instant::now();

            for seed in 1..=n_seeds as u64 {
                let layers = make_random_layers(10, 5, depth, seed);
                let result = run_loose_goal(&solver, &home, &layers, &opts, max_exp, false);
                total_layers += depth;
                total_cost += result.total_cost;
                total_expanded += result.total_expansions;
                total_solved += result.layers.iter().filter(|l| l.solved).count();
            }

            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "{:<40} | {:>3}/{:<2} | {:>8.1} | {:>8} | {:>8.0}",
                sc.name, total_solved, total_layers, total_cost, total_expanded, elapsed,
            );
        }
        eprintln!();
    }

    // ── Bottleneck diagnostics ────────────────────────────────────

    /// Core comparison: baseline palindrome vs loose-goal (IDS-only).
    ///
    /// Uses pure IDS (no Cascade A*) for both baseline and loose-goal to
    /// isolate the cost difference without A* overhead dominating runtime.
    /// Sweeps qubit counts from 4 to 50.
    ///
    /// Run with: `cargo test -p bloqade-lanes-search diag_ids_comparison -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_ids_comparison() {
        let depth = 5usize;
        let n_seeds = 3u64;

        let ids_opts = SolveOptions {
            strategy: Strategy::Ids,
            w_t: 0.0,
            ..SolveOptions::default()
        };

        eprintln!("\n=== Baseline Palindrome vs Loose-Goal (IDS-only, depth={depth}, {n_seeds} seeds) ===\n");
        eprintln!(
            "{:>4} | {:>14} {:>10} {:>10} | {:>14} {:>10} {:>10} | {:>8}",
            "Nq", "Base cost", "Solved", "Time(ms)",
            "Loose cost", "Solved", "Time(ms)", "Savings"
        );
        eprintln!(
            "{:─>4}─┼─{:─>14}─{:─>10}─{:─>10}─┼─{:─>14}─{:─>10}─{:─>10}─┼─{:─>8}",
            "", "", "", "", "", "", "", ""
        );

        for &(nq, n_pairs, sites) in &[
            (4usize, 2usize, 10usize),
            (10, 2, 10),
            (10, 5, 10),
            (20, 5, 10),
            (35, 5, 10),
            (50, 5, 10),
        ] {
            let arch = build_gemini_arch(n_pairs, sites);
            let solver = MoveSolver::from_json(&arch).unwrap();
            let home = build_home(nq, n_pairs, sites, 0);
            let max_exp = Some((10_000 * nq) as u32);

            let mut base_cost = 0.0f64;
            let mut base_solved = 0usize;
            let mut base_total = 0usize;
            let mut base_time = 0.0f64;
            let mut loose_cost = 0.0f64;
            let mut loose_solved = 0usize;
            let mut loose_total = 0usize;
            let mut loose_time = 0.0f64;

            for seed in 1..=n_seeds {
                let layers = make_random_layers(nq, n_pairs, depth, seed);

                // Baseline palindrome with IDS.
                let start = Instant::now();
                let bp = run_baseline_palindrome(&solver, &home, &layers, &ids_opts, max_exp);
                base_time += start.elapsed().as_secs_f64() * 1000.0;
                base_total += depth;
                base_solved += bp.layers.iter().filter(|l| l.solved).count();
                base_cost += bp.total_cost;

                // Loose-goal dynamic with IDS.
                let loose_opts = SolveOptions {
                    dynamic_targets: true,
                    recompute_interval: 0,
                    ..ids_opts.clone()
                };
                let start = Instant::now();
                let lg = run_loose_goal(&solver, &home, &layers, &loose_opts, max_exp, true);
                loose_time += start.elapsed().as_secs_f64() * 1000.0;
                loose_total += depth;
                loose_solved += lg.layers.iter().filter(|l| l.solved).count();
                loose_cost += lg.total_cost;
            }

            let savings = if base_cost > 0.0 {
                (1.0 - loose_cost / base_cost) * 100.0
            } else {
                0.0
            };

            eprintln!(
                "{:>4} | {:>14.1} {:>7}/{:<2} {:>10.0} | {:>14.1} {:>7}/{:<2} {:>10.0} | {:>7.0}%",
                nq,
                base_cost, base_solved, base_total, base_time,
                loose_cost, loose_solved, loose_total, loose_time,
                savings,
            );
        }
        eprintln!();
    }

    /// Core comparison: baseline vs loose-goal with multiple search strategies.
    ///
    /// Compares: baseline palindrome (IDS), loose-goal IDS (1 restart),
    /// loose-goal IDS (5 restarts), loose-goal Entropy (1 restart),
    /// loose-goal Entropy (5 restarts).
    ///
    /// Run with: `cargo test -p bloqade-lanes-search diag_strategy_sweep -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_strategy_sweep() {
        let depth = 5usize;
        let n_seeds = 3u64;

        struct StrategyVariant {
            name: &'static str,
            opts: SolveOptions,
            is_baseline: bool,
        }

        let variants = vec![
            StrategyVariant {
                name: "Base(pal/IDS)",
                opts: SolveOptions {
                    strategy: Strategy::Ids,
                    w_t: 0.0,
                    ..SolveOptions::default()
                },
                is_baseline: true,
            },
            StrategyVariant {
                name: "Loose IDS r=1",
                opts: SolveOptions {
                    strategy: Strategy::Ids,
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    restarts: 1,
                    ..SolveOptions::default()
                },
                is_baseline: false,
            },
            StrategyVariant {
                name: "Loose IDS r=5",
                opts: SolveOptions {
                    strategy: Strategy::Ids,
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    restarts: 5,
                    ..SolveOptions::default()
                },
                is_baseline: false,
            },
            StrategyVariant {
                name: "Loose IDS r=10",
                opts: SolveOptions {
                    strategy: Strategy::Ids,
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    restarts: 10,
                    ..SolveOptions::default()
                },
                is_baseline: false,
            },
            StrategyVariant {
                name: "Loose Entropy r=1",
                opts: SolveOptions {
                    strategy: Strategy::Entropy,
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    restarts: 1,
                    ..SolveOptions::default()
                },
                is_baseline: false,
            },
            StrategyVariant {
                name: "Loose Entropy r=5",
                opts: SolveOptions {
                    strategy: Strategy::Entropy,
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    restarts: 5,
                    ..SolveOptions::default()
                },
                is_baseline: false,
            },
            StrategyVariant {
                name: "Loose Entropy r=10",
                opts: SolveOptions {
                    strategy: Strategy::Entropy,
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    restarts: 10,
                    ..SolveOptions::default()
                },
                is_baseline: false,
            },
        ];

        eprintln!("\n=== Strategy Sweep: baseline vs loose-goal (depth={depth}, {n_seeds} seeds) ===\n");

        for &(nq, n_pairs, sites) in &[
            (10usize, 5usize, 10usize),
            (20, 5, 10),
            (35, 5, 10),
            (50, 5, 10),
        ] {
            let arch = build_gemini_arch(n_pairs, sites);
            let solver = MoveSolver::from_json(&arch).unwrap();
            let home = build_home(nq, n_pairs, sites, 0);
            let max_exp = Some((10_000 * nq) as u32);

            eprintln!("--- {nq}q, {n_pairs} pairs, {sites} sites ---\n");
            eprintln!(
                "{:<22} | {:>8} {:>7} {:>10} {:>10} {:>8}",
                "Strategy", "Cost", "Solved", "Expanded", "Time(ms)", "Savings"
            );
            eprintln!(
                "{:─<22}─┼─{:─>8}─{:─>7}─{:─>10}─{:─>10}─{:─>8}",
                "", "", "", "", "", ""
            );

            let mut base_cost = 0.0f64;

            for variant in &variants {
                let mut total_cost = 0.0f64;
                let mut total_solved = 0usize;
                let mut total_layers = 0usize;
                let mut total_expanded = 0u32;
                let start = Instant::now();

                for seed in 1..=n_seeds {
                    let layers = make_random_layers(nq, n_pairs, depth, seed);

                    if variant.is_baseline {
                        let bp = run_baseline_palindrome(
                            &solver, &home, &layers, &variant.opts, max_exp,
                        );
                        total_layers += depth;
                        total_solved += bp.layers.iter().filter(|l| l.solved).count();
                        total_cost += bp.total_cost;
                        total_expanded += bp.total_expansions;
                    } else {
                        let lg = run_loose_goal(
                            &solver, &home, &layers, &variant.opts, max_exp, true,
                        );
                        total_layers += depth;
                        total_solved += lg.layers.iter().filter(|l| l.solved).count();
                        total_cost += lg.total_cost;
                        total_expanded += lg.total_expansions;
                    }
                }

                let elapsed = start.elapsed().as_secs_f64() * 1000.0;

                if variant.is_baseline {
                    base_cost = total_cost;
                }
                let savings = if base_cost > 0.0 {
                    (1.0 - total_cost / base_cost) * 100.0
                } else {
                    0.0
                };

                eprintln!(
                    "{:<22} | {:>8.1} {:>4}/{:<2} {:>10} {:>10.0} {:>7.0}%",
                    variant.name,
                    total_cost,
                    total_solved,
                    total_layers,
                    total_expanded,
                    elapsed,
                    savings,
                );
            }
            eprintln!();
        }
    }

    /// Diagnostic: measure per-layer statistics to identify bottlenecks.
    ///
    /// For each qubit count, measures:
    /// - Number of CZ pairs vs spectators per layer
    /// - Greedy assignment cost (sum of per-qubit distances)
    /// - Number of accidental CZ conflicts at start of each layer
    /// - Nodes expanded per layer (loose-goal dynamic)
    /// - Whether solution was found
    /// - Cost per layer
    ///
    /// Run with: `cargo test -p bloqade-lanes-search diag_bottleneck -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_bottleneck_analysis() {
        use std::collections::HashSet;
        use std::sync::Arc;

        use crate::entangling;
        use crate::heuristic::DistanceTable;

        let depth = 5usize;
        let n_seeds = 3u64;

        eprintln!("\n{}", "=".repeat(80));
        eprintln!("=== Bottleneck Analysis ===\n");

        for &(nq, n_pairs, sites) in &[
            (10usize, 5usize, 10usize),
            (20, 5, 10),
            (35, 5, 10),
            (50, 5, 10),
        ] {
            let arch_json = build_gemini_arch(n_pairs, sites);
            let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
                serde_json::from_str(&arch_json).unwrap();
            let solver = MoveSolver::from_json(&arch_json).unwrap();
            let home = build_home(nq, n_pairs, sites, 0);

            let ent_locs = entangling::all_entangling_locations(&arch);
            let ent_set = entangling::build_entangling_set(&arch);
            let partner_map = entangling::build_partner_map(&ent_set);
            let dist_table = Arc::new(DistanceTable::new(&ent_locs, solver.index()));

            eprintln!("--- {nq}q, {n_pairs} pairs, {sites} sites ---\n");
            eprintln!(
                "{:<8} {:<6} {:<8} {:<10} {:<12} {:<10} {:<10} {:<8} {:<10}",
                "Seed", "Layer", "CZpairs", "Spectators", "AccidentCZ", "AssignCost",
                "Expanded", "Solved", "Cost"
            );

            for seed in 1..=n_seeds {
                let layers = make_random_layers(nq, n_pairs, depth, seed);
                let mut current: Vec<(u32, LocationAddr)> = home.clone();

                for (li, layer) in layers.iter().enumerate() {
                    let n_cz_pairs = layer.pairs.len();
                    let cz_qubits: HashSet<u32> =
                        layer.pairs.iter().flat_map(|&(a, b)| [a, b]).collect();
                    let n_spectators = nq - cz_qubits.len();

                    // Count accidental CZ conflicts in current config.
                    let config =
                        crate::config::Config::new(current.iter().copied()).unwrap();
                    let accidental =
                        entangling::find_accidental_cz(&config, &cz_qubits, &partner_map);
                    let n_accidental = accidental.len();

                    // Measure greedy assignment cost.
                    let greedy_targets = entangling::greedy_assign_pairs(
                        &layer.pairs,
                        &config,
                        &arch,
                        &dist_table,
                        0,
                    );
                    let assign_cost =
                        entangling::assignment_cost(&config, &greedy_targets, &dist_table);

                    // Run loose-goal dynamic solve for this layer.
                    let opts = SolveOptions {
                        strategy: Strategy::Cascade {
                            inner: InnerStrategy::Ids,
                        },
                        w_t: 0.0,
                        dynamic_targets: true,
                        recompute_interval: 0,
                        ..SolveOptions::default()
                    };
                    let max_exp = Some((5000 * nq) as u32);
                    let result = solver
                        .solve_entangling(
                            current.iter().copied(),
                            &layer.pairs,
                            std::iter::empty(),
                            max_exp,
                            &opts,
                        )
                        .unwrap();

                    let solved = result.status == SolveStatus::Solved;
                    eprintln!(
                        "{:<8} {:<6} {:<8} {:<10} {:<12} {:<10} {:<10} {:<8} {:<10.1}",
                        seed,
                        li + 1,
                        n_cz_pairs,
                        n_spectators,
                        n_accidental,
                        assign_cost,
                        result.nodes_expanded,
                        if solved { "yes" } else { "NO" },
                        result.cost,
                    );

                    // Chain: use result config for next layer.
                    if solved {
                        current = result.goal_config.iter().collect();
                    } else {
                        // Reset to home on failure.
                        current = home.clone();
                    }
                }
                eprintln!();
            }
            eprintln!();
        }
    }

    /// Diagnostic: measure how budget affects solve rate and cost.
    ///
    /// Run with: `cargo test -p bloqade-lanes-search diag_budget_scaling -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_budget_scaling() {
        let depth = 5usize;

        eprintln!("\n=== Budget Scaling Analysis ===\n");

        for &nq in &[20usize, 35, 50] {
            let arch_json = build_gemini_arch(5, 10);
            let solver = MoveSolver::from_json(&arch_json).unwrap();
            let home = build_home(nq, 5, 10, 0);

            eprintln!("--- {nq}q ---\n");
            eprintln!(
                "{:<12} {:<8} {:<10} {:<10} {:<10} {:<10}",
                "Budget/q", "Solved", "Total", "AvgCost", "AvgExpand", "Time(ms)"
            );

            for &budget_per_q in &[100u32, 500, 1000, 2000, 5000, 10000] {
                let max_exp = Some(budget_per_q * nq as u32);
                let opts = SolveOptions {
                    strategy: Strategy::Cascade {
                        inner: InnerStrategy::Ids,
                    },
                    w_t: 0.0,
                    dynamic_targets: true,
                    recompute_interval: 0,
                    ..SolveOptions::default()
                };

                let mut total_solved = 0usize;
                let mut total_layers = 0usize;
                let mut total_cost = 0.0f64;
                let mut total_expanded = 0u32;
                let start = Instant::now();

                for seed in 1..=3u64 {
                    let layers = make_random_layers(nq, 5, depth, seed);
                    let result =
                        run_loose_goal(&solver, &home, &layers, &opts, max_exp, true);
                    total_layers += depth;
                    total_cost += result.total_cost;
                    total_expanded += result.total_expansions;
                    total_solved += result.layers.iter().filter(|l| l.solved).count();
                }

                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let avg_cost = if total_solved > 0 {
                    total_cost / total_solved as f64
                } else {
                    0.0
                };
                let avg_exp = if total_layers > 0 {
                    total_expanded / total_layers as u32
                } else {
                    0
                };
                eprintln!(
                    "{:<12} {:>3}/{:<4} {:>10.1} {:>10.1} {:>10} {:>10.0}",
                    budget_per_q, total_solved, total_layers, total_cost, avg_cost,
                    avg_exp, elapsed,
                );
            }
            eprintln!();
        }
    }

    /// Diagnostic: compare loose-goal with and without chaining (return home).
    ///
    /// Tests whether chaining (keeping positions between layers) helps or hurts.
    /// Run with: `cargo test -p bloqade-lanes-search diag_chaining -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_chaining_vs_return_home() {
        let depth = 5usize;

        eprintln!("\n=== Chaining vs Return Home (loose-goal dynamic) ===\n");

        for &nq in &[10usize, 20, 35, 50] {
            let arch_json = build_gemini_arch(5, 10);
            let solver = MoveSolver::from_json(&arch_json).unwrap();
            let home = build_home(nq, 5, 10, 0);
            let max_exp = Some((5000 * nq) as u32);

            let opts = SolveOptions {
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                w_t: 0.0,
                dynamic_targets: true,
                recompute_interval: 0,
                ..SolveOptions::default()
            };

            let mut chain_solved = 0usize;
            let mut chain_cost = 0.0f64;
            let mut chain_time = 0.0f64;
            let mut home_solved = 0usize;
            let mut home_cost = 0.0f64;
            let mut home_time = 0.0f64;
            let mut total_layers = 0usize;

            for seed in 1..=3u64 {
                let layers = make_random_layers(nq, 5, depth, seed);
                total_layers += depth;

                // Chained: keep positions between layers.
                let start = Instant::now();
                let chain_result =
                    run_loose_goal(&solver, &home, &layers, &opts, max_exp, true);
                chain_time += start.elapsed().as_secs_f64() * 1000.0;
                chain_solved += chain_result.layers.iter().filter(|l| l.solved).count();
                chain_cost += chain_result.total_cost;

                // Return home: reset to home between layers.
                let start = Instant::now();
                let mut rh_total_cost = 0.0f64;
                let mut rh_solved_count = 0usize;
                for layer in &layers {
                    // Forward: loose-goal from home.
                    let fwd = solver
                        .solve_entangling(
                            home.iter().copied(),
                            &layer.pairs,
                            std::iter::empty(),
                            max_exp,
                            &opts,
                        )
                        .unwrap();

                    if fwd.status == SolveStatus::Solved {
                        // Return: solve back to home.
                        let ret = solver
                            .solve(
                                fwd.goal_config.iter(),
                                home.iter().copied(),
                                std::iter::empty(),
                                max_exp,
                                &opts,
                            )
                            .unwrap();
                        if ret.status == SolveStatus::Solved {
                            rh_total_cost += fwd.cost + ret.cost;
                            rh_solved_count += 1;
                        } else {
                            rh_total_cost += fwd.cost;
                        }
                    }
                }
                home_time += start.elapsed().as_secs_f64() * 1000.0;
                home_solved += rh_solved_count;
                home_cost += rh_total_cost;
            }

            eprintln!("--- {nq}q ---");
            eprintln!(
                "  Chained:     {:>3}/{} solved, cost={:>8.1}, time={:>8.0}ms",
                chain_solved, total_layers, chain_cost, chain_time,
            );
            eprintln!(
                "  Return home: {:>3}/{} solved, cost={:>8.1}, time={:>8.0}ms",
                home_solved, total_layers, home_cost, home_time,
            );
            eprintln!();
        }
    }

    /// Diagnostic: measure greedy assignment quality vs optimal.
    ///
    /// Compares greedy assignment cost with multiple perturbation seeds.
    /// Run with: `cargo test -p bloqade-lanes-search diag_assignment -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_assignment_quality() {
        use std::sync::Arc;

        use crate::entangling;
        use crate::heuristic::DistanceTable;

        let depth = 5usize;

        eprintln!("\n=== Greedy Assignment Quality ===\n");

        for &nq in &[10usize, 20, 35, 50] {
            let arch_json = build_gemini_arch(5, 10);
            let arch: bloqade_lanes_bytecode_core::arch::types::ArchSpec =
                serde_json::from_str(&arch_json).unwrap();
            let solver = MoveSolver::from_json(&arch_json).unwrap();
            let home = build_home(nq, 5, 10, 0);

            let ent_locs = entangling::all_entangling_locations(&arch);
            let dist_table = Arc::new(DistanceTable::new(&ent_locs, solver.index()));

            eprintln!("--- {nq}q ---\n");
            eprintln!(
                "{:<6} {:<6} {:<8} {:<10} {:<10} {:<10} {:<10}",
                "Seed", "Layer", "Pairs", "Best/10", "Worst/10", "Mean/10", "Spread"
            );

            for seed in 1..=3u64 {
                let layers = make_random_layers(nq, 5, depth, seed);
                let config =
                    crate::config::Config::new(home.iter().copied()).unwrap();

                for (li, layer) in layers.iter().enumerate() {
                    // Try 10 perturbation seeds and measure spread.
                    let mut costs: Vec<u32> = (0..10u64)
                        .map(|s| {
                            let targets = entangling::greedy_assign_pairs(
                                &layer.pairs,
                                &config,
                                &arch,
                                &dist_table,
                                s,
                            );
                            entangling::assignment_cost(&config, &targets, &dist_table)
                        })
                        .collect();
                    costs.sort();
                    let best = costs[0];
                    let worst = *costs.last().unwrap();
                    let mean = costs.iter().sum::<u32>() as f64 / costs.len() as f64;
                    let spread = worst - best;

                    eprintln!(
                        "{:<6} {:<6} {:<8} {:<10} {:<10} {:<10.1} {:<10}",
                        seed,
                        li + 1,
                        layer.pairs.len(),
                        best,
                        worst,
                        mean,
                        spread,
                    );
                }
                eprintln!();
            }
            eprintln!();
        }
    }

    /// Diagnostic: measure per-expansion move generation statistics.
    ///
    /// Instruments a single layer solve to count how many candidates are
    /// CZ-routing vs spectator-escape vs deadlock-escape per expansion.
    /// Run with: `cargo test -p bloqade-lanes-search diag_movegen -- --nocapture --ignored`
    #[ignore]
    #[test]
    fn diag_movegen_stats() {
        eprintln!("\n=== Move Generation Analysis ===\n");

        for &nq in &[10usize, 20, 50] {
            let arch_json = build_gemini_arch(5, 10);
            let solver = MoveSolver::from_json(&arch_json).unwrap();
            let home = build_home(nq, 5, 10, 0);
            let layers = make_random_layers(nq, 5, 3, 1);

            let opts_ids = SolveOptions {
                strategy: Strategy::Ids,
                w_t: 0.0,
                dynamic_targets: true,
                recompute_interval: 0,
                ..SolveOptions::default()
            };
            let opts_cascade = SolveOptions {
                strategy: Strategy::Cascade {
                    inner: InnerStrategy::Ids,
                },
                w_t: 0.0,
                dynamic_targets: true,
                recompute_interval: 0,
                ..SolveOptions::default()
            };

            let max_exp_ids = Some((2000 * nq) as u32);
            let max_exp_cascade = Some((5000 * nq) as u32);

            eprintln!("--- {nq}q, layer 1 ---");

            // IDS only.
            let start = Instant::now();
            let result_ids = solver
                .solve_entangling(
                    home.iter().copied(),
                    &layers[0].pairs,
                    std::iter::empty(),
                    max_exp_ids,
                    &opts_ids,
                )
                .unwrap();
            let ids_time = start.elapsed().as_secs_f64() * 1000.0;

            // Cascade.
            let start = Instant::now();
            let result_cascade = solver
                .solve_entangling(
                    home.iter().copied(),
                    &layers[0].pairs,
                    std::iter::empty(),
                    max_exp_cascade,
                    &opts_cascade,
                )
                .unwrap();
            let cascade_time = start.elapsed().as_secs_f64() * 1000.0;

            eprintln!(
                "  IDS:     expanded={:>6}, cost={:>5.1}, deadlocks={:>3}, solved={}, time={:.0}ms",
                result_ids.nodes_expanded,
                result_ids.cost,
                result_ids.deadlocks,
                result_ids.status == SolveStatus::Solved,
                ids_time,
            );
            eprintln!(
                "  Cascade: expanded={:>6}, cost={:>5.1}, deadlocks={:>3}, solved={}, time={:.0}ms",
                result_cascade.nodes_expanded,
                result_cascade.cost,
                result_cascade.deadlocks,
                result_cascade.status == SolveStatus::Solved,
                cascade_time,
            );
            eprintln!();
        }
    }
}
