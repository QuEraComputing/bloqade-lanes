# Motivation: `PhysicalSpec` and `PhysicalNoiseSpec`

## What is `PhysicalSpec`?

`PhysicalSpec` is the hardware description layer of the `ArchSpec` redesign. It captures everything that is physically true about a neutral atom processor — the geometry of the atom arrays, the pre-calibrated transport routes between them, and the device capabilities — without reference to how the compiler addresses or schedules those atoms. See the [ArchSpec redesign spec](./2026-05-08-arch-spec-redesign.md) for the full type definitions.

---

## 1. Human-readable layout using a familiar data structure

Atom positions in a neutral atom processor are arranged in regular 2D grids controlled by spatial light modulators (SLMs). The `Grid` type reflects this directly:

```python
class Grid:
    x_start: float                # x coordinate of the first column (µm)
    y_start: float                # y coordinate of the first row (µm)
    x_spacing: tuple[float, ...]  # per-column x spacings; len = number of columns - 1
    y_spacing: tuple[float, ...]  # per-row y spacings; len = number of rows - 1
```

This is the same structure used in `bloqade-geometry` and `bloqade-lanes-bytecode-core` — a deliberately familiar, human-readable representation. Hardware engineers and compiler authors alike can read a `PhysicalSpec` and immediately understand the physical layout without having to decode an index scheme or look up a mapping table.

`slm_grids` is a `dict[str, Grid]` keyed by human-readable names (e.g. `"storage"`, `"entangling"`). The string keys are stable identifiers that appear throughout the rest of the stack — in `NameBridge`, in `local_gate_grids`, in `rydberg_tophats`, and in downstream dispatch tables — without leaking geometry details across the boundary.

---

## 2. Physical connectivity via pre-defined, calibrated buses

Neutral atom processors move atoms along pre-calibrated routes. In `PhysicalSpec`, each such route is a `PhysicalBus`:

```python
class PhysicalBus:
    src: Grid
    dst: Grid
    waypoints: list[Grid] | None = None   # intermediate stops, excluding src and dst
```

A `PhysicalBus` encodes the full physical path: the starting configuration (`src`), the ending configuration (`dst`), and any intermediate stops (`waypoints`) for multi-hop moves. All `Grid`s in a bus have the same number of atom positions — the i-th atom in `src` moves to the i-th atom in `dst`.

**In the near term, all buses represent SLM↔SLM moves.** Atoms are transported from one SLM zone to another along a known, pre-calibrated trajectory. The `buses: dict[str, PhysicalBus]` field in `PhysicalSpec` enumerates all such routes by name (e.g. `"storage_to_entangling"`, `"entangling_to_storage"`). These names are stable across compiler versions — adding or recalibrating a route means adding or updating a key in this dict, with no changes to the compiler pipeline.

The string bus keys also serve directly as the dispatch handles for the `lanes2flair` rewrite layer: a `dict[str, KernelFn]` keyed by bus name replaces the old hardcoded `{0: ..., 8: ...}` integer dispatch, and new bus types are handled automatically.

---

## 3. Decoupling geometry and bus definitions from the compiler

Before this redesign, bus definitions were embedded inside the compiler's zone and word addressing scheme. This created tight coupling: changing how the compiler addressed atoms required updating bus definitions, and adding a new bus type required modifying compiler internals.

`PhysicalSpec` is owned entirely by `bloqade-flair`. The lanes compiler never writes to it — it only reads `PhysicalSpec` to resolve physical positions during `ArchSpec.derive()`. This means:

- The flair team can add new buses, resize grids, or recalibrate existing routes without touching the lanes compiler.
- The lanes compiler can change how it addresses and schedules atoms without invalidating any bus definition.
- `MachineModel` (the derived artifact) is always a pure function of `PhysicalSpec` + `AddressSpace` — regenerated on demand, never stale.

The `NameBridge` (jointly owned) provides the one coordination point: when flair adds a new bus, its name is appended to `NameBridge.bus_names` to assign it a stable integer ID. Beyond that, each team evolves its layer independently.

---

## 4. Compatibility with non-fixed-lane architectures

Not all current neutral atom processors use a fixed-lane architecture with pre-defined SLM zones and named buses. For machines that don't — where atom movement is more free-form or where bus concepts don't apply — `PhysicalSpec` remains valid: simply leave `buses` empty (`{}`).

The lanes compiler treats an empty `buses` dict as a valid `PhysicalSpec`. The `NameBridge.bus_names` list is correspondingly empty, and `MachineModel.bus_graph.buses` is an empty list. Nothing else in the pipeline breaks. This means `PhysicalSpec` can describe both the current Gemini fixed-lane architecture and earlier or simpler architectures without requiring a separate code path.

---

## 5. Future work: AOD+SLM gate operations

Current hardware uses atom transport exclusively for state preparation and rearrangement between SLM zones. A key upcoming capability is using AOD (acousto-optic deflector) positioning in combination with SLM zones to execute entangling gate operations — where the physical path of an atom move is itself part of the gate protocol.

This will require new data in `PhysicalSpec` to describe:

- Which buses can be used to execute a gate (not just transport atoms)
- The geometric constraints on AOD positioning during a gate move
- Timing and synchronization requirements between AOD and SLM control

**The exact representation is TBD.** The current `PhysicalBus` type encodes only start/end/waypoint geometry — it has no field for gate semantics. Extending it (or introducing a new `GatingBus` type alongside `PhysicalBus`) is the most likely direction, but this requires hardware input on what constraints need to be encoded. The `buses: dict[str, PhysicalBus]` structure is intentionally open-ended — adding AOD+SLM gate buses is a matter of adding new entries with new types, without breaking existing bus definitions.

---

## `PhysicalNoiseSpec`: noise as a peer layer

`PhysicalNoiseSpec` is a peer of `PhysicalSpec` — it shares the same string keys but is not nested inside it. Gate fidelities, coherence times, SPAM errors, and transport noise are physical hardware properties, but they are optional metadata for the compiler: a `PhysicalSpec` without a `PhysicalNoiseSpec` is fully valid for compilation. Noise is only needed when building a `NoiseModel` for noisy simulation or error-aware optimization.

The key design decision is that `PhysicalNoiseSpec` is keyed by the same string identifiers as `PhysicalSpec`:

- `bus_noise: dict[str, BusNoiseParams]` — keyed by bus name (same keys as `PhysicalSpec.buses`)
- `cz_noise: dict[str, TopHatNoiseParams]` — keyed by top-hat name (same keys as `PhysicalSpec.rydberg_tophats`)
- `local_gate_noise: dict[str, LocalGateNoiseParams]` — keyed by grid name (same keys as `PhysicalSpec.slm_grids`)
- `spam_noise: dict[str, SPAMNoiseParams]` — keyed by grid name

This means `PhysicalNoiseSpec` can be updated independently of the compiler — a recalibration that changes fidelity numbers or coherence times touches only `PhysicalNoiseSpec`, not `PhysicalSpec` or any compiler pass.
