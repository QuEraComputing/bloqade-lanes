# Fix: builder must emit shared-template word IDs (LocationAddress convention)

**Status:** planned (not started)
**Date:** 2026-07-29
**Author:** Phillip Weinberg (with Claude)

## Problem

The zone-centric ArchSpec model has one correct convention for `word_id`, but
the Python architecture *builder* implements a different, legacy one. They only
coexist today because every bundled spec is single-zone (all offsets are 0).

### The correct convention (LocationAddress / shared template)

`words` is ONE global template list. `word_id` indexes it and is **reused
identically by every zone** (each zone addresses `0..Nw-1`); `zone_id`
disambiguates. A `LocationAddr` is `(zone_id, word_id, site_id)`.

Evidence this is the intended model (all consumers already assume it):

- Rust `Zone` struct has **no `words` field** — `crates/bloqade-lanes-bytecode-core/src/arch/types.rs:49-68`. Zones cannot own slicing.
- `ArchSpec.words: Vec<Word>`, "A word's ID is its index in this list" — `types.rs:96`.
- `Word.sites` are `[x_idx, y_idx]` index pairs into the **parent zone's grid** — `types.rs:40-43`. One template applied per-zone against each zone's own grid.
- `query.rs:415-422` `location_position()` / `query.rs:199-201` `word_by_id()` resolve `self.words.get(loc.word_id)` **ignoring `zone_id`**.
- `query.rs:366-377` `zone_location_index()` uses `wid * spw + sid` and bounds-checks against the single global `words.len()`.
- `atom_state.rs:225`: *"In the zone-centric model, all zones share the same words."*
- Rust test helper `make_valid_two_zone_spec()` (`validate.rs:522-583`) is a 2-zone spec with a **2-word global list** and a `zone_bus` whose dst is `(zone_id 1, word_id 0)` — a second zone re-addressing `word_id 0` against a 2-word template is only valid under the shared-template convention (concatenation would need 4 words).
- Docs: `docs/src/arch/zone-centric-concepts.md` ("Words — The Shared Template", Invariant #5).

### The legacy/buggy convention (offset / concatenation)

The builder concatenates each zone's words into one flat list and **offsets
`word_id` per zone** (zone 0 = `0..Nw0-1`, zone 1 = `Nw0..`). For a 2-zone,
2-word device it emits `words=[W0,W1,W2,W3]` (4) with zone 1 owning IDs 2,3 —
where the model wants `words=[W0,W1]` (2) with both zones addressing 0,1.

Offending sites:

- `python/bloqade/lanes/arch/build/imperative.py`
  - `ArchBuilder._word_id_offsets` / `_total_words` (`:1066`, `:1071`)
  - `add_zone()` accumulates offsets (`:1091-1092`)
  - `build()` concatenates `all_words` (`:1200-1204`)
  - `build()` applies `offset + w` to word_buses (`:1211-1219`),
    `words_with_site_buses` (`:1224-1232`), `entangling_pairs` (`:1236-1238`),
    zone_buses (`:1252-1263`), modes/`bitstring_order` (`:1266-1282`)
  - `_compute_paths()` `global_word = word_offset + local_word` (`:968`, `:1034`)
- `python/bloqade/lanes/arch/build/word_factory.py`
  - `WordGrid.word_id_offset` field (`:32`), `word_id_at()` (`:38-40`),
    `all_word_ids` (`:42-45`), `cz_pairs()` (`:47-51`)
- `python/bloqade/lanes/arch/build/blueprint.py`
  - `word_id_offset += zone_spec.num_words` accumulation (`:240`, `:248`)
  - `w - offset` round-trip conversions (`:289-294`, `:298-303`, `:319-329`)

### Why it's redundant (key investigation finding)

The offset machinery is pure round-tripping and the shared template is already
the de-facto reality:

1. `ArchBlueprint.__post_init__` already **forces uniform grid dims** across
   zones — `blueprint.py:137-141` ("All zones must have the same grid
   dimensions").
2. `create_zone_words()` output depends only on `num_rows/num_cols` + `layout`
   + `word_id_offset` — so every zone's word slicing is **byte-for-byte
   identical except for the added offset** (`word_factory.py:70-94`).
3. Flow is local→global→local→global: `create_zone_words` stamps global IDs →
   topology generators emit global via `word_id_at()`
   (`topology.py:140,187,284`) → blueprint subtracts offset back to local
   (`blueprint.py:289-329`) → `ArchBuilder.build()` re-adds it. Setting every
   offset to 0 collapses the whole chain with **no behavioral change on
   existing single-zone specs**.

### Validator gap (and why Rust can't close it)

`crates/.../arch/validate.rs` only bounds-checks referenced `word_id` against
the **global** `words.len()` (`:196-210`, `:266-299`, `:367-403`) — which the
concatenation model satisfies. Crucially, the Rust validator **structurally
cannot** detect concatenation: `Zone` has no per-zone word count (no `words`
field), so a concatenated 2-zone × Nw-word spec is indistinguishable from a
legitimate `2*Nw`-word shared template where zone 0 uses `0..Nw-1` and zone 1
uses `Nw..2Nw-1`. Both are valid ArchSpecs. Therefore the guard against
re-entry cannot live in Rust — it must live on the **Python builder**
(the identical-template check, step 2) plus the inverted tests.

## Blast radius / current state

- **No bundled multi-zone specs, and the builder is not even in the shipped
  path.** Both bundled specs are single-zone AND JSON-loaded —
  `gemini/physical/spec.py:25` and the logical spec both do
  `_RustArchSpec.from_json(...)`, never `ArchBuilder`. So this is a latent
  correctness fix, not a live-data migration, and shipped data provably cannot
  shift.
- **What actually breaks under a concatenated multi-zone spec.** The spec stays
  a *valid* ArchSpec — `location_position`, `word_partner_map`, `word_zone_map`
  all still run. The break is a **contract mismatch**: a consumer that addresses
  zone-1 atoms with zone-local IDs (`0..Nw-1`), as the docs mandate, will miss on
  `zone.words_with_site_buses.contains(&word_id)` (`query.rs:313`),
  `sites_with_word_buses` (`:328`), and get wrong flat indices from
  `zone_location_index`, because the builder wrote *global* IDs into those
  per-zone lists. It is not a blanket "every consumer crashes."
- **Tests currently codify the bug.** `python/tests/arch/test_arch_builder.py`:
  - class `TestArchBuilderMultiZoneOffsets` (`:680-762`) asserts offset/global
    IDs (`>= 4`, `len(words) == 8`).
  - `test_multi_zone_with_connection` (`:541`) asserts `len(words) == 4`.
  - `test_global_word_ids_assigned` (`:597`) asserts `len(words) == 4`.
  These must **invert** to the shared-template contract.

## Design decision

Keep the per-zone `ZoneBuilder.add_word` API (minimal change) but make
`ArchBuilder.build()` **validate that all zones declare an identical word
template and emit that single `Nw`-length list with no offsets.** Chosen over
"words at ArchBuilder level" because the blueprint already guarantees identical
templates, so validation is cheap and the public API is preserved.

## Implementation steps (TDD)

1. **Red test** — add `TestArchBuilderSharedTemplate` to
   `python/tests/arch/test_arch_builder.py` asserting:
   - two zones with identical 4-word templates → `len(spec.words) == 4`
   - zone 1 word_buses / entangling_pairs reference `0..Nw-1`
   - both zones' word_buses are identical
   - zone_bus src/dst word_ids are zone-local (`0..Nw-1`), disambiguated by zone_id
   - mismatched templates across zones → `ValueError(match="same word template")`
   (Draft already written in the working session; re-add it.)

2. **`imperative.py` `ArchBuilder`**
   - delete `_word_id_offsets`, `_total_words`; `add_zone()` stops tracking offsets
   - in `build()`, validate every zone's `_words` equals zone 0's `_words`
     (raise `ValueError("... same word template ...")` otherwise); emit that
     single list as `all_words` (length `Nw`)
   - drop every `offset + w` (word_buses, `words_with_site_buses`,
     `entangling_pairs`, zone_buses, modes `bitstring_order`)
   - `_compute_paths()` uses local `word_id` directly (drop `word_offset` param
     / set to 0)

3. **`word_factory.py`** — remove `WordGrid.word_id_offset`; `word_id_at`,
   `all_word_ids`, `cz_pairs` return template-local IDs. Update signature of
   `create_zone_words` (drop `word_id_offset`).
   - **Public-API change:** `WordGrid` and `create_zone_words` are re-exported
     from `arch/__init__.py` and `arch/build/__init__.py`, and `word_id_at` /
     `all_word_ids` / `cz_pairs` change semantics global→local. Decide
     explicitly: hard-remove (fine for pre-1.0 dev SDK — chosen here) vs. keep a
     deprecated no-op `word_id_offset=0` param. Note the break in the commit body.

4. **`blueprint.py`** — drop the `word_id_offset` accumulation (`:240,248`) and
   the `w - offset` conversions (become identities). Confirm topology
   generators now receive/emit template-local IDs consistently.

5. **Rust validator** (`validate.rs`) — **no effective guard is possible here.**
   The `word_id < words.len()` bound already exists (`:202-209`, `:282-297`) and
   is exactly what concatenation passes; see "Validator gap" above for why Rust
   can't distinguish the two models. Do NOT rely on a Rust check for regression
   protection — the real guards are the builder's identical-template validation
   (step 2) and the inverted tests (step 6). Optional: refresh the doc comment on
   `zone_location_index` / `WordRef` to restate the zone-local contract, purely
   as documentation.

6. **Invert the offset tests** — rewrite `TestArchBuilderMultiZoneOffsets` →
   shared-template assertions; fix `test_multi_zone_with_connection` and
   `test_global_word_ids_assigned` (`len == 4` → `len == 2`). Flag each flipped
   assertion in the commit body.

7. **Verify** — `just test-python` + `just test-rust`. Benchmark metrics
   **cannot** shift: the shipped specs are JSON-loaded (`from_json`), so
   `ArchBuilder` produces none of the benchmarked data, and even for
   builder-made single-zone specs every offset is already 0. Regenerating the
   baselines is unnecessary; if `latest_physical.csv` / `latest_logical.csv` do
   diff, treat it as a signal that something unexpected changed, not as a
   routine baseline refresh.

## Risks / watch-items

- A caller using the **imperative `ArchBuilder` directly** with genuinely
  different per-zone words would now get a `ValueError`. That's correct under
  the model. **Grep done:** the only `ArchBuilder()` caller outside
  `python/tests/` is `blueprint.py:308`, which always feeds identical templates
  (uniform grid dims + deterministic `create_zone_words`), so no production
  caller trips the new check.
- Topology generators (`topology.py`) work in `word_id_at()` space; once
  offsets are 0 they emit template-local IDs — double-check no generator does
  independent offset math (the `s * d + offset` at `:238-252` is site-group
  expansion, unrelated to zone word offsets).
- `visualize/` or `analysis/` code that assumed concatenated global IDs for
  multi-zone rendering. **Grep done:** every `word_id_offset` / `word_offset` /
  `_word_id_offsets` / `_total_words` usage is confined to the three `build/`
  files (`imperative.py`, `word_factory.py`, `blueprint.py`); nothing in
  `visualize/`, `analysis/`, or `heuristics/` references the offset machinery.
