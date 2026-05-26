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

Run these three commands and collect all output before spawning any agents.
This is fast (seconds) and gives every agent a reliable structural foundation.

```bash
# 1. Workspace dependency graph
cargo metadata --format-version 1 --no-deps

# 2. Change hotspots — 30-day window scoped to the crate directory
git log --since="30 days ago" --name-only --pretty=format:"%h %an %s" \
  -- crates/<crate-name>/

# 3. Consumer file paths across the workspace
rg "use bloqade_lanes_<crate_name_underscored>" --type rust -l
```

**Adjust the `--since` window:**
- Low-activity crate (< 5 commits in 30 days): use `--since="90 days ago"`
- Heavily active crate (> 30 commits in 30 days): use `--since="14 days ago"`

**From the git log output, extract before proceeding:**
- **Hotspot files**: files appearing in 3+ commits → pass to Agent 3 for weighted scrutiny
- **AI-authorship signals**: commits containing `Co-Authored-By: Claude` or
  messages matching `^feat|^refactor|^chore` with 10+ files touched → flag for
  AI-drift lens

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
   ### Design Philosophy
   ### Ownership & Maintainability
   ### Forward-looking Risk
```

Save to: `docs/superpowers/reviews/YYYY-MM-DD-<crate-name>-review.md`

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
