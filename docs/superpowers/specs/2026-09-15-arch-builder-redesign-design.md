# Architecture builder redesign

Replace `ZoneBuilder` / `ArchBuilder` / `blueprint` with a single builder shaped
like the spec it produces, and deprecate the existing API.

## Motivation

The current builder is not merely buggy — it is *inconsistent*, in a way that
cannot be repaired without two compensating mechanisms that both cut across the
object boundary it defines.

### The builder can express architectures the spec cannot hold

`ArchSpec.words` is a single, spec-wide list. A `Word` is a set of
`[x_idx, y_idx]` pairs that index **each zone's own grid**, so every zone applies
the *same* index pattern to its *own* coordinates:

```rust
pub struct ArchSpec { pub words: Vec<Word>, pub zones: Vec<Zone>, ... }
pub struct Word { pub sites: Vec<[u32; 2]> }
```

`ZoneBuilder.add_word` nevertheless lets every zone declare an independent word
list, and `ArchBuilder.build()` then compares them and rejects all but the first:

```python
template = self._zones[0]._words
for zone in self._zones[1:]:
    if zone._words != template:
        raise ValueError(f"zone '{zone.name}' declares a different word template ...")
```

The constraint already exists. It is simply enforced at the last possible moment
instead of being structural. `blueprint` has the same problem one layer up:
`create_zone_words(zone_spec, layout)` derives a word grid per zone from that
zone's `num_rows` / `num_cols`, so a blueprint whose zones differ in shape
assembles cleanly and dies at the final identity check.

### The builder cannot express architectures the spec *can* hold

`Zone.sites_with_word_buses` is a per-zone subset in the spec. `build()`
hardcodes it:

```python
sites_with_word_buses = list(range(zone.sites_per_word)) if word_buses else []
```

So the mismatch runs in both directions.

### Validation is non-local, and no amount of care fixes that

A `ZoneBuilder` cannot validate itself. Every fact that decides whether a zone's
buses are legal is spec-scoped:

- the **word template** is shared across zones;
- the **grid index space** is therefore shared too, since `Word.sites` indexes it;
- the **participant sets** (`words_with_site_buses`, `sites_with_word_buses`) are
  per-zone fields shared by every bus in the zone, so a site bus's carried-atom
  set is not knowable from the bus;
- **`ArchBuilder.connect`** makes two zones' occupancy relevant to each other.

Making the current API sound would require enforcing cross-zone template identity
*and* freezing words once any bus exists. The freeze is the tell: `connect` lives
on `ArchBuilder` but would have to reach into `ZoneBuilder` objects the caller
still holds a reference to and mutate them. A method on one object silently
disabling methods on another is the shape of an ownership error, not a missing
guard.

### Scope

Both bundled Gemini specs contain exactly **one zone**, so the entire multi-zone
and shared-template apparatus is unexercised by anything shipped. The blast
radius is `blueprint.py` and tests.

## Ground truth: what the spec actually says

| concept | scope | field |
|---|---|---|
| word / site slicing | **spec** | `ArchSpec.words: Vec<Word>` (`[x_idx, y_idx]` pairs) |
| grid index space | **spec** (implied) | every zone's grid must cover the template's indices |
| grid coordinates | zone | `Zone.grid` (`x_start` + `x_spacing[]`) |
| site / word buses | zone | `Zone.site_buses`, `Zone.word_buses` |
| transport participation | zone | `Zone.words_with_site_buses`, `Zone.sites_with_word_buses` |
| entangling pairs | zone | `Zone.entangling_pairs` |
| blockade radius | **spec** | `ArchSpec.blockade_radius: Option<f64>` |
| inter-zone buses, modes, paths, capabilities | **spec** | `ArchSpec.*` |

Note the deliberate asymmetry in the last-but-one row: the radius is set once at
spec scope, but the pairs it derives are per-zone, because the same radius
against different zone coordinates yields different pairs.

Subset participation is production, not hypothetical — the shipped physical spec
has `words_with_site_buses: [1, 3, 5, ..., 19]`, odd words only.

## Proposed API

```python
b = ArchBuilder(grid_shape=(32, 5), word_shape=(8, 1))

# 1. Word template — spec-wide index patterns, defined once.
b.add_word(x=[0, 4, 8, 12, 16, 20, 24, 28], y=[0])   # -> word 0
b.add_word(x=[1, 5, 9, 13, 17, 21, 25, 29], y=[0])   # -> word 1

# 2. Zones — a name plus coordinates for that shared index space,
#    plus which words/sites take part in transport here.
b.add_zone(
    "gate",
    x=[...32 coordinates...], y=[...5 coordinates...],
    x_clearance=..., y_clearance=...,
    words_with_site_buses=[1, 3, 5, ...],
    sites_with_word_buses=range(8),
)

# 3. Buses — always zone-qualified.
b.add_site_bus("gate", src=[0, 2, 4, 6], dst=[1, 3, 5, 7])
b.add_word_bus("gate", src=[...], dst=[...])
b.connect(("gate", [...]), ("storage", [...]))

# 4. Spec scope.
b.set_blockade_radius(2.0)          # derives per-zone entangling_pairs
b.add_mode("all", ["gate"])
spec = b.build()
```

### What this makes unrepresentable

- Two zones slicing the grid differently — there is one template, declared once.
- A word added after a bus — `add_word` is a phase-1 call; zones and buses are
  phases 2 and 3. No freeze flag, no cross-object mutation: the phase order *is*
  the constraint.
- A bus whose carried-atom set is unknown — participation is declared on the zone,
  before any bus, so every bus's atom set is fixed at the call that adds it.

That last point is the payoff. Every validation question argued in the review of
PR #1010 becomes decidable at the offending call, without a freeze, because the
information exists by the time it is needed.

### Bus validation, once, in the right place

An AOD transport is realizable exactly when it is **separable and
order-preserving**: each atom's x-displacement depends only on its column and its
y-displacement only on its row, and tones keep their relative order. This is
weaker than a uniform translation — compression and expansion are single AOD
operations.

The stowaway rule should test the real hazard rather than a proxy: reject a bus
when a **non-participating atom** sits at one of its tone intersections. An
intersection that no word owns can never hold an atom, so it is not a hazard.
This replaces PR #1010's "bus atoms must *fill* their tone product", which
rejects performable buses — including any staggered participating-word layout.
The destination-side completeness check should be dropped rather than relaxed: it
rejects the same layouts independently, its stowaway justification does not
transfer to a deposit, and the intra-bus collision it might catch is already
impossible once the mapping check passes (that mapping is injective by
construction).

### Transport paths are inherited, not recomputed

The bundled architectures' `paths` are **hardware-derived and baked in**, not
products of the builder's path search. A sweep of clearance values from 0.1 to
5.0 µm reproduces 0 of the logical spec's 110 lanes and at best 240 of the
physical spec's 1120 — the search simply does not generate that geometry.

This matters because move durations are read off those segment lengths, so
`estimated_fidelity` and the committed routing benchmark baselines depend on
them. A `from_spec` → `build()` round-trip that recomputed paths would silently
replace calibrated lane geometry with search output and shift every routing
metric.

So `from_spec` carries the spec's paths over and `build()` reuses them verbatim
while the bus structure is untouched; editing a bus invalidates them and forces
a search, which is why clearances are optional on `from_spec` and required only
then. `build(recompute_paths=True)` is the explicit opt-in.

Clearances themselves are not part of an `ArchSpec` — they are inputs to the
path search, not properties of the architecture — so a round-trip cannot
recover them, and the builder says so rather than guessing.

## Migration

1. Land the new builder alongside the existing one, under a new name.
2. Port `blueprint`'s topology generators — they are independent of the builder
   and worth keeping — then deprecate the `blueprint` entry point.
3. Point `from_spec` / `from_zone` (PR #1006) at the new builder. Rebuilding a
   plain value is a better target than replaying an imperative API.
4. Deprecate `ZoneBuilder` / `ArchBuilder` once the bundled specs and tests are
   migrated.

There is no `ZoneBuilder` replacement. A zone becomes a plain value — name, grid,
clearances, participation, buses — with no validation logic of its own, because
it has no basis for any.

## Open questions

- Should zones be required to share grid *dimensions*, or merely to cover the
  template's index range? Nothing shipped settles this, and nothing currently
  checks it.
- Do the topology generators keep operating on a `WordGrid` abstraction, or
  directly on the spec-wide template?
- Does `set_blockade_radius` stay a scan, or become an assertion against
  explicitly supplied pairs, given it silently overwrites `entangling_pairs`
  today?

## Related

- PR #1010 — carries the bug fixes; the freeze mechanism is explicitly out of
  scope, since this redesign supersedes it.
- PR #1006 — `from_spec` / `from_zone`; should target the new builder.
- Issue #1008 — `ArchSpec::validate()` has no AOD geometry checks, so
  JSON-loaded specs bypass everything the builder enforces. The separability rule
  above is the spec-layer counterpart.
