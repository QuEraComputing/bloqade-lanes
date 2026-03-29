use std::fs;
use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

use bloqade_lanes_bytecode_core::arch::ArchSpec;
use bloqade_lanes_bytecode_core::bytecode::program::Program;
use bloqade_lanes_bytecode_core::bytecode::text;
use bloqade_lanes_bytecode_core::bytecode::validate;

#[derive(Parser)]
#[command(
    name = "bloqade-bytecode",
    about = "Lane-move bytecode assembler and tools"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Assemble a text program into binary format.
    Assemble {
        /// Input text file.
        input: PathBuf,
        /// Output binary file.
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Disassemble a binary program into text format.
    Disassemble {
        /// Input binary file.
        input: PathBuf,
        /// Output text file (omit to print to stdout).
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Validate a program (text or binary).
    Validate {
        /// Input file (text .sst or binary .bin).
        input: PathBuf,
        /// Architecture spec JSON file for address validation.
        #[arg(long)]
        arch: Option<PathBuf>,
        /// Run stack type simulation.
        #[arg(long)]
        simulate_stack: bool,
    },
    /// Architecture spec commands (pretty-print or validate).
    #[command(args_conflicts_with_subcommands = true)]
    Arch {
        /// ArchSpec JSON file to pretty-print.
        input: Option<PathBuf>,

        #[command(subcommand)]
        command: Option<ArchCommand>,
    },
}

#[derive(Subcommand)]
enum ArchCommand {
    /// Validate an ArchSpec JSON file.
    Validate {
        /// Input ArchSpec JSON file.
        input: PathBuf,
    },
}

fn main() {
    let cli = Cli::parse();

    let result = match cli.command {
        Command::Assemble { input, output } => cmd_assemble(&input, &output),
        Command::Disassemble { input, output } => cmd_disassemble(&input, output.as_deref()),
        Command::Validate {
            input,
            arch,
            simulate_stack,
        } => cmd_validate(&input, arch.as_deref(), simulate_stack),
        Command::Arch { command, input } => match (command, input) {
            (Some(ArchCommand::Validate { input }), _) => cmd_validate_arch_spec(&input),
            (None, Some(input)) => cmd_show_arch_spec(&input),
            (None, None) => Err("provide an ArchSpec JSON file or use a subcommand".to_string()),
        },
    };

    if let Err(e) = result {
        eprintln!("error: {}", e);
        process::exit(1);
    }
}

fn cmd_assemble(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    let source =
        fs::read_to_string(input).map_err(|e| format!("reading {}: {}", input.display(), e))?;
    let program = text::parse(&source).map_err(|e| e.to_string())?;
    let binary = program.to_binary();
    fs::write(output, &binary).map_err(|e| format!("writing {}: {}", output.display(), e))?;
    eprintln!(
        "assembled {} instructions -> {}",
        program.instructions.len(),
        output.display()
    );
    Ok(())
}

fn cmd_disassemble(input: &PathBuf, output: Option<&std::path::Path>) -> Result<(), String> {
    let bytes = fs::read(input).map_err(|e| format!("reading {}: {}", input.display(), e))?;
    let program = Program::from_binary(&bytes).map_err(|e| e.to_string())?;
    let text_out = text::print(&program);
    match output {
        Some(path) => {
            fs::write(path, &text_out).map_err(|e| format!("writing {}: {}", path.display(), e))?;
            eprintln!(
                "disassembled {} instructions -> {}",
                program.instructions.len(),
                path.display()
            );
        }
        None => print!("{}", text_out),
    }
    Ok(())
}

fn cmd_validate(
    input: &PathBuf,
    arch_path: Option<&std::path::Path>,
    simulate_stack: bool,
) -> Result<(), String> {
    let program = load_program(input)?;

    let arch = match arch_path {
        Some(path) => {
            let json = fs::read_to_string(path)
                .map_err(|e| format!("reading {}: {}", path.display(), e))?;
            Some(ArchSpec::from_json_validated(&json).map_err(|e| e.to_string())?)
        }
        None => None,
    };

    let mut all_errors = Vec::new();
    all_errors.extend(validate::validate_structure(&program));

    if let Some(arch) = &arch {
        all_errors.extend(validate::validate_arch_constraints(&program, arch));
    }

    if simulate_stack {
        all_errors.extend(validate::simulate_stack(&program, arch.as_ref()));
    }

    if all_errors.is_empty() {
        eprintln!("valid ({} instructions)", program.instructions.len());
        Ok(())
    } else {
        for e in &all_errors {
            eprintln!("  {}", e);
        }
        Err(format!("{} validation error(s)", all_errors.len()))
    }
}

fn cmd_show_arch_spec(input: &PathBuf) -> Result<(), String> {
    let json =
        fs::read_to_string(input).map_err(|e| format!("reading {}: {}", input.display(), e))?;
    let spec = ArchSpec::from_json(&json).map_err(|e| e.to_string())?;
    print_arch_spec(&spec);
    Ok(())
}

fn print_arch_spec(spec: &ArchSpec) {
    println!("ArchSpec v{}", spec.version);
    println!();

    // Geometry
    let geom = &spec.geometry;
    println!(
        "Geometry: {} word(s), {} sites/word",
        geom.words.len(),
        geom.sites_per_word
    );
    for (word_idx, word) in geom.words.iter().enumerate() {
        let grid = &word.positions;
        println!(
            "  Word {}: {}x{} grid, {} sites",
            word_idx,
            grid.num_x(),
            grid.num_y(),
            word.sites.len()
        );
        println!(
            "    x: start={}, spacing={:?}",
            grid.x_start, grid.x_spacing
        );
        println!(
            "    y: start={}, spacing={:?}",
            grid.y_start, grid.y_spacing
        );
        let sites_str: Vec<String> = word
            .sites
            .iter()
            .map(|s| format!("({},{})", s[0], s[1]))
            .collect();
        println!("    sites: {}", sites_str.join(" "));
        if let Some(cz) = &word.cz_pairs {
            let cz_str: Vec<String> = cz.iter().map(|p| format!("({},{})", p[0], p[1])).collect();
            println!("    cz_pairs: {}", cz_str.join(" "));
        }
    }
    println!();

    // Buses
    println!(
        "Buses: {} site bus(es), {} word bus(es)",
        spec.buses.site_buses.len(),
        spec.buses.word_buses.len()
    );
    for (bus_idx, bus) in spec.buses.site_buses.iter().enumerate() {
        println!(
            "  Site bus {}: src={:?} dst={:?}",
            bus_idx, bus.src, bus.dst
        );
    }
    for (bus_idx, bus) in spec.buses.word_buses.iter().enumerate() {
        println!(
            "  Word bus {}: src={:?} dst={:?}",
            bus_idx, bus.src, bus.dst
        );
    }
    if !spec.words_with_site_buses.is_empty() {
        println!("  words_with_site_buses: {:?}", spec.words_with_site_buses);
    }
    if !spec.sites_with_word_buses.is_empty() {
        println!("  sites_with_word_buses: {:?}", spec.sites_with_word_buses);
    }
    println!();

    // Zones
    println!("Zones: {} zone(s)", spec.zones.len());
    for (zone_idx, zone) in spec.zones.iter().enumerate() {
        println!("  Zone {}: words={:?}", zone_idx, zone.words);
    }
    println!("  entangling_zones: {:?}", spec.entangling_zones);
    println!(
        "  measurement_mode_zones: {:?}",
        spec.measurement_mode_zones
    );

    // Capabilities
    println!();
    println!("Capabilities:");
    println!("  feed_forward: {}", spec.feed_forward);
    println!("  atom_reloading: {}", spec.atom_reloading);

    // Paths
    if let Some(paths) = &spec.paths
        && !paths.is_empty()
    {
        println!();
        println!("Paths: {} path(s)", paths.len());
        for path in paths {
            println!(
                "  0x{:08X}: {} waypoint(s)",
                path.lane,
                path.waypoints.len()
            );
            for wp in &path.waypoints {
                println!("    [{}, {}]", wp[0], wp[1]);
            }
        }
    }
}

fn cmd_validate_arch_spec(input: &PathBuf) -> Result<(), String> {
    let json =
        fs::read_to_string(input).map_err(|e| format!("reading {}: {}", input.display(), e))?;
    let _spec = ArchSpec::from_json_validated(&json).map_err(|e| e.to_string())?;
    eprintln!("arch spec is valid: {}", input.display());
    Ok(())
}

/// Load a program from either text (.sst) or binary format.
fn load_program(path: &PathBuf) -> Result<Program, String> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "sst" => {
            let source = fs::read_to_string(path)
                .map_err(|e| format!("reading {}: {}", path.display(), e))?;
            text::parse(&source).map_err(|e| e.to_string())
        }
        _ => {
            let bytes = fs::read(path).map_err(|e| format!("reading {}: {}", path.display(), e))?;
            Program::from_binary(&bytes).map_err(|e| e.to_string())
        }
    }
}
