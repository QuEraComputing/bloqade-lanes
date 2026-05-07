//! `policies-primer` — autogenerator for `policies/primer.md`.
//!
//! Plan C — see `docs/superpowers/specs/2026-05-01-move-policy-dsl-plan-c-design.md` §8.
//!
//! Modes:
//!   default      — write `policies/primer.md`
//!   --check      — exit 1 with a diff if `policies/primer.md` differs from
//!                  the regenerated content
//!   --stdout     — write to stdout instead of `policies/primer.md`
//!   --input-dir  — read registration files from a path on disk instead of
//!                  the compile-time `include_str!` constants
//!
//! When both `--stdout` and `--check` are set, `--stdout` takes precedence
//! (just print, no comparison).
//!
//! Source files are embedded via `include_str!` at compile time;
//! generation involves no runtime path resolution against the cargo
//! manifest layout (unless `--input-dir` is specified).

use std::collections::BTreeMap;
use std::path::PathBuf;

use bloqade_lanes_search::fixture;

const ACTIONS_SRC: &str = include_str!("../move_policy_dsl/actions.rs");
const LIB_MOVE_SRC: &str = include_str!("../move_policy_dsl/lib_move.rs");
const GRAPH_HANDLE_SRC: &str = include_str!("../move_policy_dsl/graph_handle.rs");
const LIB_TARGET_SRC: &str = include_str!("../target_generator_dsl/lib_target.rs");

const PRIMER_PATH: &str = "policies/primer.md";

#[derive(Default)]
struct ParsedPrimer {
    prose: BTreeMap<String, String>,
}

struct StarlarkMethod {
    name: String,
    sig: String,
    summary: String,
}

fn main() {
    let mut input_dir: Option<PathBuf> = None;
    let mut to_stdout = false;
    let mut check = false;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        if a == "--check" {
            check = true;
        } else if a == "--stdout" {
            to_stdout = true;
        } else if a == "--input-dir" {
            input_dir = args.next().map(PathBuf::from);
        } else if let Some(path) = a.strip_prefix("--input-dir=") {
            input_dir = Some(path.into());
        } else {
            eprintln!("policies-primer: unknown arg: {a}");
            std::process::exit(2);
        }
    }

    let (actions_src, lib_move_src, graph_src, lib_target_src) = match &input_dir {
        Some(d) => (
            std::fs::read_to_string(d.join("actions.rs")).expect("read actions.rs"),
            std::fs::read_to_string(d.join("lib_move.rs")).expect("read lib_move.rs"),
            std::fs::read_to_string(d.join("graph_handle.rs")).expect("read graph_handle.rs"),
            std::fs::read_to_string(d.join("lib_target.rs")).expect("read lib_target.rs"),
        ),
        None => (
            ACTIONS_SRC.to_string(),
            LIB_MOVE_SRC.to_string(),
            GRAPH_HANDLE_SRC.to_string(),
            LIB_TARGET_SRC.to_string(),
        ),
    };

    let existing = if to_stdout {
        // When writing to stdout we don't read the on-disk primer for prose
        // (the output is fully deterministic from the input sources).
        String::new()
    } else {
        std::fs::read_to_string(PRIMER_PATH).unwrap_or_default()
    };
    let parsed = parse_existing(&existing);

    let regenerated = render(
        &parsed,
        &actions_src,
        &lib_move_src,
        &graph_src,
        &lib_target_src,
    );

    if to_stdout {
        print!("{regenerated}");
    } else if check {
        let on_disk = std::fs::read_to_string(PRIMER_PATH).unwrap_or_default();
        if on_disk.trim() != regenerated.trim() {
            let diff = unified_diff(&on_disk, &regenerated);
            eprintln!("{PRIMER_PATH} is stale. Diff:\n{diff}");
            eprintln!("Run `just generate-primer` to update.");
            std::process::exit(1);
        }
        eprintln!("{PRIMER_PATH} is up to date.");
    } else {
        std::fs::write(PRIMER_PATH, &regenerated).expect("write primer");
        eprintln!("wrote {PRIMER_PATH}");
    }
}

fn parse_existing(src: &str) -> ParsedPrimer {
    let mut prose: BTreeMap<String, String> = BTreeMap::new();
    let mut in_block: Option<String> = None;
    let mut buf = String::new();
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("<!-- BEGIN PROSE: ") {
            let name = rest.trim_end_matches(" -->").to_string();
            in_block = Some(name);
            buf.clear();
            continue;
        }
        if line.starts_with("<!-- END PROSE: ") {
            if let Some(name) = in_block.take() {
                prose.insert(name, std::mem::take(&mut buf));
            }
            continue;
        }
        if in_block.is_some() {
            buf.push_str(line);
            buf.push('\n');
        }
    }
    ParsedPrimer { prose }
}

fn render(
    parsed: &ParsedPrimer,
    actions_src: &str,
    lib_move_src: &str,
    graph_src: &str,
    lib_target_src: &str,
) -> String {
    let mut out = String::new();
    out.push_str("<!-- AUTOGEN: DO NOT EDIT BY HAND.\n");
    out.push_str("     Regenerate with `just generate-primer`. -->\n\n");
    out.push_str("# Move Policy & Target Generator DSL — Primer\n\n");
    out.push_str(&prose_block("intro", parsed));
    out.push_str("\n## Move Policy surface\n\n");
    out.push_str(&autogen_block("actions", &render_actions(actions_src)));
    out.push_str(&autogen_block("lib_move", &render_lib_move(lib_move_src)));
    out.push_str(&autogen_block(
        "graph_handle",
        &render_graph_handle(graph_src),
    ));
    out.push_str(&prose_block("move-tour", parsed));
    out.push_str("\n## Target Generator surface\n\n");
    out.push_str(&autogen_block(
        "lib_target",
        &render_lib_target(lib_target_src),
    ));
    out.push_str(&prose_block("target-tour", parsed));
    out.push_str("\n## Problem fixture schema\n\n");
    out.push_str(&autogen_block("schema", &render_schema()));
    out
}

fn prose_block(name: &str, parsed: &ParsedPrimer) -> String {
    let body = parsed
        .prose
        .get(name)
        .cloned()
        .unwrap_or_else(|| format!("TODO: write prose for {name}\n"));
    format!("<!-- BEGIN PROSE: {name} -->\n{body}<!-- END PROSE: {name} -->\n")
}

fn autogen_block(name: &str, body: &str) -> String {
    format!("<!-- BEGIN AUTOGEN: {name} -->\n{body}\n<!-- END AUTOGEN: {name} -->\n\n")
}

fn render_actions(src: &str) -> String {
    let methods = parse_starlark_methods(src, "register_actions");
    render_starlark_section("actions.* — kernel-driven verbs", &methods)
}

fn render_lib_move(src: &str) -> String {
    let methods = parse_starlark_methods(src, "register_lib_methods");
    render_starlark_section("lib_move.* — query primitives", &methods)
}

fn render_graph_handle(src: &str) -> String {
    let methods = parse_starlark_methods(src, "register_graph_methods");
    render_starlark_section("graph.* — read-only graph accessors", &methods)
}

fn render_lib_target(src: &str) -> String {
    let methods = parse_starlark_methods(src, "register_lib_target_methods");
    render_starlark_section("lib_target.* — placement query primitives", &methods)
}

fn render_schema() -> String {
    let mut out = String::from("### Problem fixture schema\n\n");
    out.push_str("Problem fixtures are JSON files with one of two top-level shapes, ");
    out.push_str("discriminated by a `\"kind\"` field.\n\n");
    out.push_str("```json\n");
    out.push_str(&fixture::json_schema_pretty());
    out.push_str("\n```\n");
    out
}

fn render_starlark_section(title: &str, methods: &[StarlarkMethod]) -> String {
    let mut out = String::new();
    out.push_str(&format!("### `{title}`\n\n"));
    if methods.is_empty() {
        out.push_str(
            "(no methods found — generator may have failed to parse the registration site)\n\n",
        );
        return out;
    }
    for m in methods {
        out.push_str(&format!(
            "#### `{}`\n\n{}\n\n```rust\n{}\n```\n\n",
            m.name, m.summary, m.sig
        ));
    }
    out
}

/// Parse a Rust source file looking for an outer `pub fn <fn_filter>(...)`
/// or `fn <fn_filter>(...)` (the `#[starlark_module]`-wrapped registration
/// function) and return metadata for every inner `fn` item inside its body.
fn parse_starlark_methods(src: &str, fn_filter: &str) -> Vec<StarlarkMethod> {
    let file: syn::File = match syn::parse_str(src) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for item in file.items {
        if let syn::Item::Fn(f) = item {
            if f.sig.ident != fn_filter {
                continue;
            }
            for stmt in &f.block.stmts {
                if let syn::Stmt::Item(syn::Item::Fn(inner)) = stmt {
                    let name = inner.sig.ident.to_string();
                    let inner_sig = &inner.sig;
                    let sig = quote::quote!(#inner_sig).to_string();
                    let summary = inner
                        .attrs
                        .iter()
                        .find_map(|a| {
                            if a.path().is_ident("doc")
                                && let syn::Meta::NameValue(nv) = &a.meta
                                && let syn::Expr::Lit(syn::ExprLit {
                                    lit: syn::Lit::Str(s),
                                    ..
                                }) = &nv.value
                            {
                                return Some(s.value().trim().to_string());
                            }
                            None
                        })
                        .unwrap_or_else(|| "(undocumented)".into());
                    out.push(StarlarkMethod { name, sig, summary });
                }
            }
        }
    }
    out
}

fn unified_diff(a: &str, b: &str) -> String {
    let diff = similar::TextDiff::from_lines(a, b);
    let mut buf = String::new();
    for change in diff.iter_all_changes() {
        let sign = match change.tag() {
            similar::ChangeTag::Delete => "-",
            similar::ChangeTag::Insert => "+",
            similar::ChangeTag::Equal => " ",
        };
        buf.push_str(&format!("{sign}{change}"));
    }
    buf
}
