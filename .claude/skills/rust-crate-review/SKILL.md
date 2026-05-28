---
name: rust-crate-review
description: Use when orienting to a Rust crate after significant changes, conducting
  periodic maintainability checks, or onboarding a contributor who needs to build
  a systems-level understanding of a crate and its boundaries within the workspace.
---

# Rust Crate Review

Architectural health check for a single Rust crate and its one-hop neighborhood
(direct dependencies + consumers). Produces a saved review document and a set of
open questions for systems-thinking dialogue.

**Not a PR review** — use `requesting-code-review` for diff-scoped review. This
skill reasons about design intent and responsibility boundaries over time, not diffs.

## When to Use

- After a period of heavy AI-authored changes to a crate
- Periodic maintenance check (monthly or per-release cycle)
- Onboarding a junior contributor who needs to build a mental model of a crate

## Invocation

```
/rust-crate-review <crate-name>
```

`<crate-name>` is the directory name under `crates/`
(e.g., `bloqade-lanes-dsl-core`, `bloqade-lanes-search`).

## Step 1 — Automated Harvest (no agents)

Run these four commands and collect all output before spawning any agents.
This is fast (seconds) and gives every agent a reliable structural foundation.

Before running command 3, derive `<crate-module>` from `<crate-name>` by
replacing dashes with underscores
(e.g., `bloqade-lanes-search` → `bloqade_lanes_search`).

```bash
# 1. Workspace dependency graph
cargo metadata --format-version 1 --no-deps

# 2. Change hotspots — 30-day window scoped to the crate directory
git log --since="30 days ago" --name-only --pretty=format:"%h %an %s" \
  -- crates/<crate-name>/

# 3. Consumer file paths across the workspace
#    Matches `use <crate-module>`, `<crate-module>::Foo`, and
#    `use <crate-module> as bar` forms; excludes the crate's own sources.
rg "\b<crate-module>\b" --type rust -l \
  | grep -v "^crates/<crate-name>/"

# 4. AI-authored commits in the harvest window
#    `Co-Authored-By:` lives in commit trailers, so this needs its own pass.
git log --since="30 days ago" --grep="Co-Authored-By: Claude" \
  --name-only --pretty=format:"%h %an %s" -- crates/<crate-name>/
```

**Default to `--since="30 days ago"` for commands 2 and 4. After running command 2,
check the count and rerun both if needed:**
- Output empty or under 5 commits → rerun with `--since="90 days ago"`
- Output exceeds 30 commits → rerun with `--since="14 days ago"` to focus on
  recent drift

**From the harvest output, extract before proceeding:**
- **Hotspot files**: from command 2, files appearing in 3+ commits →
  pass to Agent 3 for weighted scrutiny
- **AI-authored commits**: command 4 already filters to these — pass the
  commit list (with file names) to Agent 3 for the AI-drift lens
- **Large non-AI commits** (secondary signal): from command 2, any commit
  touching 10+ files whose subject matches `^feat|^refactor|^chore` →
  also flag to Agent 3

## Step 2 — Phase 1: Parallel Agents

Dispatch Agent 1 (External API Mapper) and Agent 2 (Internal Architecture Mapper)
simultaneously using the `dispatching-parallel-agents` skill.

Pass to both agents:
- The target crate name
- The full harvest output from Step 1

Use the prompt templates in `agent-prompts.md` — fill in `{{CRATE_NAME}}` and
`{{HARVEST_OUTPUT}}` before dispatching.

## Step 3 — Phase 2: Sequential Agent

Once **both** Phase 1 agents have returned, dispatch Agent 3 (Critical Evaluator).

Pass to Agent 3:
- The target crate name
- The full harvest output from Step 1
- The complete output from Agent 1
- The complete output from Agent 2

Use the Agent 3 prompt template in `agent-prompts.md` — fill in all four
placeholders: `{{CRATE_NAME}}`, `{{HARVEST_OUTPUT}}`, `{{AGENT_1_OUTPUT}}`,
`{{AGENT_2_OUTPUT}}`.

## Step 4 — Synthesis

Assemble the final review document from the three agent outputs following this
structure:

```
# Crate Review: <crate-name> (<YYYY-MM-DD>)

## 1. Context
   One-paragraph summary of what the crate does, its dependency position
   (what it depends on, what depends on it), and change activity level.

## 2. External API Surface
   From Agent 1: public type inventory, responsibility portraits,
   API friction points, dead public surface.

## 3. Internal Architecture
   From Agent 2: module map, internal interaction graph,
   pub(crate) type inventory, coupling hotspots, responsibility portraits.

## 4. Critical Evaluation
   From Agent 3: contract divergence, Rust health findings (hotspot-weighted),
   architectural health, AI-drift findings, ⚠ emerging pattern callouts.

## 5. Open Questions
   One subsection per finding area that produced findings.
   ### Contract Divergence
   ### Rust Health
   ### Architectural Health
   ### AI-Drift
   ### Emerging Patterns
```

Save to: `docs/superpowers/reviews/YYYY-MM-DD-<crate-name>-review.md`
(create the `docs/superpowers/reviews/` directory if it does not yet exist —
on first run, only `plans/` and `specs/` are siblings).

After saving, surface the Section 5 questions directly in the conversation as a
prompt to the user — do not just leave them buried in the document.

## Emerging Pattern Callout Format

Agent 3 emits these as structured blocks. Preserve them verbatim in Section 4:

```
⚠ Emerging Pattern: "<pattern name>"
  Appears in: <file:line>, <file:line>, <file:line>
  Similarity: <description of structural resemblance>
  Signal: <N> instances, last added <X> days ago
  Suggested abstraction: <trait name / fn signature>
  Readiness: [ready to abstract | still evolving | monitor]
```

Readiness thresholds (from git log recency):
- All instances stable 20+ days → **ready to abstract**
- Any instance touched within 7 days → **still evolving**
- Between 7–20 days → **monitor**

## Prompt Templates

See `agent-prompts.md` for the complete Agent 1, Agent 2, and Agent 3 prompt
templates with placeholder conventions.
