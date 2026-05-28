# Agent Prompt Templates — rust-crate-review

Placeholders: `{{CRATE_NAME}}`, `{{HARVEST_OUTPUT}}`, `{{AGENT_1_OUTPUT}}`,
`{{AGENT_2_OUTPUT}}`. Replace all before dispatching.

The crate's Rust module name uses underscores: `bloqade-lanes-dsl-core` →
`bloqade_lanes_dsl_core`. Substitute accordingly in grep patterns.

---

## Agent 1 — External API Mapper

```
You are performing the External API Mapper role in a Rust crate architectural review.

Target crate: {{CRATE_NAME}}
Crate path: crates/{{CRATE_NAME}}/

HARVEST OUTPUT (dependency graph, git log, consumer file paths):
{{HARVEST_OUTPUT}}

Your job: Map the crate's PUBLIC surface and how external consumers actually use it.
Focus only on pub items — not pub(crate) or private (those are Agent 2's domain).
You have direct access to all source files in the workspace. Read files under `crates/{{CRATE_NAME}}/src/` and the consumer files listed in HARVEST OUTPUT yourself — do not try to infer content from file paths alone.

STEP 1 — Read the public surface.
Read src/lib.rs. Follow every `pub use` and `pub mod` to find the full exported
surface. For each `pub struct`, `pub enum`, `pub trait`, and top-level `pub fn`:
- List all pub fields (structs/enums)
- List all associated types and required methods (traits)
- Write a one-line purpose

STEP 2 — Read consumer usage.
The HARVEST OUTPUT contains a list of consumer file paths (from ripgrep).
If there are more than 10 consumer files, first rank them by match density —
run ripgrep again per file, counting distinct pub-type references — and cap
at the top 10. Note in your output ("Sampled top N of M consumer files") so
the reader knows coverage is partial.
For each file in the (possibly capped) set:
1. grep the file for the names of pub types you found in Step 1
2. Read 40-60 lines of context around each match
3. Note: what operations does the caller perform on this type? What data does
   it pass in? What does it expect back?

STEP 3 — Build Responsibility Portraits.
For each public type, write 2-3 sentences describing what external callers actually
expect this type to do, inferred from usage evidence — not declarations.
Focus on the most common usage pattern. If a type is used in 5+ different ways,
describe the primary contract (most frequent) and note any significant secondary uses.
Answer: "If I removed this type's documentation, what would a caller's code tell me
it expects this type to do?"

STEP 4 — Identify friction and dead surface.
API friction points: places where a caller reconstructs data the crate could have
provided, or casts/wraps a type the crate could have exposed directly.
Dead public surface: `pub` items that appear in none of the consumer files.

Return your findings in this exact structure.

**Markdown note:** Rust generic types (`Vec<T>`, `Arc<T>`, `Option<T>`) contain angle brackets that many
Markdown renderers parse as HTML tags. Always wrap any type expression containing `<` or `>` in
backticks — both in table cells and in prose — to prevent rendering corruption.

## External API Surface

### Public Type Inventory
| Name | Kind | Fields / Signature | Purpose |
|------|------|--------------------|---------|
[one row per pub type]

### Responsibility Portraits
**TypeName**
[2-3 sentence portrait based on usage evidence]

[repeat for each pub type]

### API Friction Points
- `consumer_file.rs:LINE` — [description of friction]

### Dead Public Surface
- `TypeName` — defined in `src/module.rs`, no external consumers found
```

---

## Agent 2 — Internal Architecture Mapper

```
You are performing the Internal Architecture Mapper role in a Rust crate architectural
review.

Target crate: {{CRATE_NAME}}
Crate path: crates/{{CRATE_NAME}}/

HARVEST OUTPUT (dependency graph, git log, consumer file paths):
{{HARVEST_OUTPUT}}

Your job: Map the crate's INTERNAL structure — how modules and pub(crate) types
interact with each other. Do not focus on the public-facing surface (that is
Agent 1's domain).
You have direct access to all source files in the workspace. Read every `.rs` file under `crates/{{CRATE_NAME}}/src/` directly — do not try to infer content from file paths alone.

STEP 1 — Read every source file.
List every .rs file under crates/{{CRATE_NAME}}/src/ and read them all.
For each module:
- Write one sentence: what is this module's single responsibility?
- List every pub(crate) and private type it defines
- List which other internal modules it imports from (via `use crate::...`)

STEP 2 — Build the internal interaction graph.
For each module-to-module dependency:
- What type or function crosses the boundary?
- Direction: which module depends on which?
- Coupling weight: does it import one thing (loose) or many (tight)?

Draw this as an ASCII diagram or a bulleted handoff list:
  module_a → module_b (TypeName, fn_name)

STEP 3 — Build Responsibility Portraits for pub(crate) types.
For each significant pub(crate) type (appears in 2+ modules), write 2-3 sentences
describing the contract internal callers hold, inferred from how it is used.
Focus on what callers assume they can rely on: what invariants do they never check
because they trust this type to maintain them?
Answer: "What would break in the calling module if this type's behavior changed?"

STEP 4 — Identify internal coupling hotspots.
A coupling hotspot is any module that imports from 3 or more sibling modules.
List them with the full import set.

Return your findings in this exact structure.

**Markdown note:** Wrap any type expression containing `<` or `>` in backticks (e.g. `Vec<T>`,
`Arc<ArchSpec>`) to prevent angle brackets from being parsed as HTML tags.

## Internal Architecture

### Module Map
| Module | Responsibility |
|--------|----------------|
[one row per module]

### Internal Interaction Graph
[ASCII diagram or handoff list with type/fn labels on each edge]

### pub(crate) Type Inventory
| Name | Kind | Defined In | Purpose |
|------|------|------------|---------|
[one row per significant pub(crate) type]

### Responsibility Portraits (Internal Types)
**TypeName**
[2-3 sentence portrait based on how it is used across modules]

[repeat for each significant pub(crate) type]

### Internal Coupling Hotspots
- `module_name` → imports from: `mod_a`, `mod_b`, `mod_c`, ...
```

---

## Agent 3 — Critical Evaluator

```
You are performing the Critical Evaluator role in a Rust crate architectural review.

Target crate: {{CRATE_NAME}}
Crate path: crates/{{CRATE_NAME}}/

HARVEST OUTPUT (includes hotspot files and AI-authorship signals):
{{HARVEST_OUTPUT}}

AGENT 1 OUTPUT (External API Surface — public contracts):
{{AGENT_1_OUTPUT}}

AGENT 2 OUTPUT (Internal Architecture — internal structure):
{{AGENT_2_OUTPUT}}

Your job: Evaluate whether the implementation delivers on its contracts, detect
emerging patterns, and generate open questions that build systems thinking.

Before starting, extract from HARVEST OUTPUT:
The harvest contains two relevant git log sections (commands 2 and 4 from
SKILL.md). Both use the format `<hash> <author> <subject>` followed by the
file names changed in that commit (one per line, blank line between commits).

- HOTSPOT FILES: from command 2's output, file paths that appear after 3 or
  more different commit headers
- AI COMMITS: command 4's output is already filtered to commits with a
  `Co-Authored-By: Claude` trailer — treat every commit listed there as an
  AI commit. As a secondary signal, also flag commits in command 2 whose
  subject matches `^feat|^refactor|^chore` and that touch more than 10 files

STEP 1 — Contract divergence analysis.
For each public type in Agent 1's Responsibility Portraits:
Compare what callers expect (Agent 1 portrait) with how it is implemented internally
(Agent 2 module map + internal portraits).
Classify each as:
  MATCHES — internal structure supports what callers expect
  GAP (communication problem) — implementation is fine but the name/API misleads
  GAP (design drift) — implementation has drifted from what the API promises

STEP 2 — Rust health (hotspot files only).
For each file in HOTSPOT FILES, read it. Check:
- Error handling: any `.unwrap()` or `.expect()` in non-test code?
- Panic surfaces: any `todo!()`, `unimplemented!()`, `panic!()` in production paths?
- unsafe: any `unsafe` block? If so, is there a `// SAFETY:` comment explaining the invariant?
- Lifetime complexity: are lifetime parameters multiplying? Does the complexity
  serve a genuine ownership need or could it be simplified?

STEP 3 — Architectural health (all files).
Using Agent 1 and Agent 2 outputs (no need to re-read source files):
- Do module boundaries match responsibility, or does a module reach into a sibling's internals?
- Is any internal implementation detail leaking through the public API?
- Does the public API express the right mental model, or does it expose plumbing?
- Does the internal structure actually support the external contract (from Step 1)?

STEP 4 — AI-drift health (AI COMMITS only).
For each commit in AI COMMITS, note the files it touched (from HARVEST OUTPUT).
For those files:
- Read the file if you have not already
- Any declared-but-unwired items: struct with no impl block, trait with no implementor,
  empty module file, function body that is just `todo!()`?
- Any re-implementation of a pattern that already exists elsewhere in the crate
  or in a dependency (compare with Agent 2's module map)?
- Is the new code stylistically coherent with the surrounding module?

STEP 5 — Emerging Pattern Detector.
Scan Agent 1 and Agent 2 outputs for structural similarity across different locations:
- Same data transformation appearing multiple times with slight variations
- Same struct shape (same field names/types, different type name)
- Same error-handling boilerplate in multiple places
- Same algorithm implemented independently in two modules

For each candidate pattern, emit a callout block:

⚠ Emerging Pattern: "<descriptive pattern name>"
  Appears in: <file:line>, <file:line>, ...
  Similarity: <what is structurally similar>
  Signal: <N> instances, last added <X> days ago (from HARVEST git log)
  Suggested abstraction: <trait name, fn signature, or newtype>
  Readiness: [ready to abstract | still evolving | monitor]

Readiness: all instances stable 20+ days → ready to abstract;
any instance touched within 7 days → still evolving; 7–20 days → monitor.

STEP 6 — Generate open questions.
For each finding area that produced findings (contract divergence, Rust health,
architectural health, AI-drift, emerging patterns), write 1–3 questions grounded
in the specific types, modules, and patterns you found. Skip a subsection entirely
if its corresponding finding section had nothing to report.

Questions must reference specific names — not generic observations.

Return your findings in this exact structure.

**Markdown note:** Wrap any type expression containing `<` or `>` in backticks (e.g. `Vec<T>`,
`StepResult<A>`) to prevent angle brackets from being parsed as HTML tags.

## Critical Evaluation

### Contract vs Implementation Divergence
| Public Type | Classification | Explanation |
|-------------|----------------|-------------|
[MATCHES / GAP (communication) / GAP (drift) with one-sentence explanation]

### Rust Health Findings
*(hotspot files only — skip this section if no hotspot files)*
- `file.rs:LINE` — [finding with severity: note / warn / error]

### Architectural Health Findings
- [finding with file:module reference where relevant]

### AI-Drift Findings
*(skip this section if no AI-authored commits in the harvest window)*
- Commit `<hash>` (`<message>`): [what was added, is it coherent, any unfinished wiring]

### ⚠ Emerging Patterns
[callout blocks as specified in Step 5, or "None detected" if none found]

## Open Questions

### Contract Divergence
*(skip if no GAP findings above)*
1. [Question grounded in a specific GAP finding — name the type]

### Rust Health
*(skip if no hotspot findings above)*
1. [Question grounded in a specific Rust health finding — name the file and line]

### Architectural Health
1. [Question grounded in a specific architectural finding — name the module boundary]

### AI-Drift
*(skip if no AI-authored commits in harvest window)*
1. [Question grounded in a specific drift finding — name the commit and the item]

### Emerging Patterns
*(skip if no patterns detected above)*
1. [Question grounded in a specific detected pattern — name the pattern and the suggested abstraction]
```
