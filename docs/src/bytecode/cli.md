# CLI Reference

The `bloqade-bytecode` CLI assembles, disassembles, and validates lane-move bytecode programs.

```
bloqade-bytecode <COMMAND>
```

## Building

### CLI binary

```bash
just build-cli       # Build CLI in release mode (output: target/release/bloqade-bytecode)
```

### Testing

```bash
just test-rust       # Run Rust tests (core + CLI crates)
just cli-smoke-test  # CLI bytecode validation tests
```

### Python wheel with bundled CLI

```bash
just develop         # Dev install with bundled CLI
just build-wheel     # Build distributable wheel
```

## Commands

### `assemble`

Assemble a text program (`.sst`) into binary format (`.bin`).

```
bloqade-bytecode assemble <INPUT> -o <OUTPUT>
```

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input text file (`.sst`) |
| `-o, --output <OUTPUT>` | Output binary file (required) |

**Example:**

```bash
bloqade-bytecode assemble prog.sst -o prog.bin
# assembled 12 instructions -> prog.bin
```

---

### `disassemble`

Disassemble a binary program (`.bin`) into human-readable text format.

```
bloqade-bytecode disassemble <INPUT> [-o <OUTPUT>]
```

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input binary file (`.bin`) |
| `-o, --output <OUTPUT>` | Output text file (omit to print to stdout) |

**Examples:**

```bash
# Print to stdout
bloqade-bytecode disassemble prog.bin

# Write to file
bloqade-bytecode disassemble prog.bin -o prog.sst
# disassembled 12 instructions -> prog.sst
```

The text format round-trips perfectly: `text → binary → text` produces identical output.

---

### `validate`

Validate a program for correctness. Accepts both text (`.sst`) and binary formats — the format is auto-detected from the file extension.

```
bloqade-bytecode validate <INPUT> [--arch <ARCH>] [--simulate-stack]
```

| Argument | Description |
|----------|-------------|
| `<INPUT>` | Input file (`.sst` = text, otherwise binary) |
| `--arch <ARCH>` | ArchSpec JSON file for address validation |
| `--simulate-stack` | Run stack type simulation |

**Validation levels:**

| Level | When | What it checks |
|-------|------|----------------|
| Structural | Always | Arity bounds, `initial_fill` ordering |
| Address | `--arch` provided | Location, lane, zone, and bus validity against the architecture |
| Stack simulation | `--simulate-stack` | Type safety, stack balance |

**Examples:**

```bash
# Structural validation only
bloqade-bytecode validate prog.sst
# valid (12 instructions)

# Full validation with architecture and stack simulation
bloqade-bytecode validate prog.sst --arch gemini-logical.json --simulate-stack
# valid (12 instructions)

# Validation failure
bloqade-bytecode validate bad.sst --arch gemini-logical.json
#   [0] initial_fill: invalid location address ...
# error: 1 validation error(s)
```

---

### `arch`

Pretty-print an architecture specification.

```
bloqade-bytecode arch <INPUT>
```

| Argument | Description |
|----------|-------------|
| `<INPUT>` | ArchSpec JSON file |

**Example output:**

```
ArchSpec v1.0

Geometry: 2 word(s), 10 sites/word
  Word 0: 2x10 grid, 10 sites
    x: start=0, spacing=[5.0]
    y: start=0, spacing=[1.0, 1.0, ...]
    site_indices: (0,0) (0,1) (0,2) ...
    has_cz: (1,0) (1,1) (1,2) ...

Buses: 1 site bus(es), 1 word bus(es)
  Site bus 0: src=[0, 1, 2, 3, 4] dst=[5, 6, 7, 8, 9]
  Word bus 0: src=[0] dst=[1]
  words_with_site_buses: [0, 1]
  sites_with_word_buses: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Zones: 1 zone(s)
  Zone 0: words=[0, 1]
  entangling_zones: [0]
  measurement_mode_zones: [0]

Paths: 2 path(s)
  0x0100000F: 3 waypoint(s)
    [0.0, 0.0]
    [1.0, 0.0]
    [2.0, 0.0]
  0x0100010F: 2 waypoint(s)
    [0.0, 1.0]
    [1.0, 1.0]
```

The Paths section is only shown when the ArchSpec includes path data. Each path is identified by its encoded lane address and lists the AOD waypoints (physical coordinates) that define the transport trajectory.

### `arch validate`

Validate an ArchSpec JSON file for internal consistency.

```
bloqade-bytecode arch validate <INPUT>
```

| Argument | Description |
|----------|-------------|
| `<INPUT>` | ArchSpec JSON file |

**Example:**

```bash
bloqade-bytecode arch validate gemini-logical.json
# arch spec is valid: gemini-logical.json
```

---

## File Formats

| Extension | Format | Description |
|-----------|--------|-------------|
| `.sst` | Text | Human-readable bytecode (one instruction per line, `#` comments) |
| `.bin` | Binary | Compact binary encoding (`BLQD` magic header, 16 bytes per instruction) |
| `.json` | JSON | Architecture specification |

See also the [Instruction Quick Reference](inst-quick-ref.md) for a compact summary of all 24 instructions, or the full [Instruction Set](inst-spec.md) for encoding details.
