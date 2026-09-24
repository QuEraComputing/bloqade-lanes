//! The Bloqade Lanes device: an atom-movement machine, as a vihaco component.
//!
//! This is the lanes half of the instruction set. The stack ops (`const`,
//! `dup`, `halt`, …) are **not** here — they come from vihaco-cpu's own `CPU`
//! component, which is composed alongside this one in
//! [`machine`](super::machine). That is the idiomatic vihaco composition: each
//! device owns its instruction set, and the composite routes between them.
//!
//! [`vihaco::component!`] generates two enums from the `instruction` block:
//!
//! - `lanes::syntax::Instruction` — the surface form, with a derived text
//!   parser under the `lanes` dialect head (`lanes.move 2`);
//! - `lanes::runtime::Instruction` — the executable form.
//!
//! Where the two differ, the `Surface => Runtime` notation names both. Only the
//! address constants differ: they are written in hex ([`HexU64`]/[`HexU32`])
//! and executed as plain integers. vihaco generates no conversion between the
//! pair, so [`lower`] writes it out.
//!
//! ## What "execute" means here
//!
//! A lanes program describes *atom movement*, not quantum state. So this device
//! executes the arrangement ops — `initial_fill`, `fill`, `move` drive an
//! [`AtomStateData`] and fault on an illegal move — and emits the quantum ops
//! (`local_r`, `cz`, `measure`, …) as [`LanesEffect`]s for a downstream
//! consumer to interpret. Simulating the quantum state is explicitly out of
//! scope; that is what PPVM is for.

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::types::ArchSpec;
use crate::atom_state::AtomStateData;

use super::syntax::{HexU32, HexU64};

vihaco::component! {
    /// Atom-movement device: owns where every qubit currently sits.
    #[derive(Debug, Default, Clone)]
    pub component Lanes {
        /// Current qubit → location mapping, advanced by the arrangement ops.
        pub(crate) atoms: AtomStateData,
        /// Architecture the moves are resolved against. `move` needs it to turn
        /// a lane into (src, dst); without one the device cannot execute moves.
        pub(crate) arch: Option<ArchSpec>,
        /// Next qubit id handed out by `initial_fill` / `fill`.
        pub(crate) next_qubit_id: u32,
    }

    instruction {
        // ---- Address constants ----
        // Written in hex, executed as the packed integer. The composite pushes
        // the value onto the CPU stack; keeping them here (rather than using
        // `cpu.const u64`) is what preserves the location/lane/zone distinction
        // that validation relies on.
        #[pattern = "'const_loc $0"]
        ConstLoc(HexU64 => u64),
        #[pattern = "'const_lane $0"]
        ConstLane(HexU64 => u64),
        #[pattern = "'const_zone $0"]
        ConstZone(HexU32 => u32),

        // ---- Atom arrangement ----
        #[pattern = "'initial_fill $0"]
        InitialFill(u32),
        #[pattern = "'fill $0"]
        Fill(u32),
        #[pattern = "'move $0"]
        Move(u32),

        // ---- Quantum gates ----
        // `local_rz` precedes `local_r` (and `global_rz` precedes `global_r`)
        // because the derive tries variants in declaration order and the
        // shorter token is a prefix of the longer one.
        #[pattern = "'local_rz $0"]
        LocalRz(u32),
        #[pattern = "'local_r $0"]
        LocalR(u32),
        #[pattern = "'global_rz"]
        GlobalRz,
        #[pattern = "'global_r"]
        GlobalR,
        #[pattern = "'cz"]
        Cz,

        // ---- Measurement ----
        #[pattern = "'measure $0"]
        Measure(u32),
        #[pattern = "'await_measure"]
        AwaitMeasure,

        // ---- Arrays ----
        // `new_array <type_tag> <dim0> <dim1>`; 1-D arrays use `dim1 = 0`.
        #[pattern = "'new_array $0 $1 $2"]
        NewArray(u32, u32, u32),
        #[pattern = "'get_item $0"]
        GetItem(u32),

        // ---- Detectors / observables ----
        #[pattern = "'set_detector"]
        SetDetector,
        #[pattern = "'set_observable"]
        SetObservable,
    }
}

pub use lanes::{Lanes, runtime, syntax as surface};
pub use runtime::Instruction as LanesInstruction;
pub use surface::Instruction as LanesSurfaceInstruction;

/// Lower a parsed instruction to its executable form.
///
/// vihaco's `component!` generates the surface and runtime enums independently
/// and no conversion between them, so this is written out. Every arm but the
/// three address constants is a straight rename.
pub fn lower(inst: LanesSurfaceInstruction) -> LanesInstruction {
    use LanesSurfaceInstruction as S;
    match inst {
        S::ConstLoc(HexU64(v)) => LanesInstruction::ConstLoc(v),
        S::ConstLane(HexU64(v)) => LanesInstruction::ConstLane(v),
        S::ConstZone(HexU32(v)) => LanesInstruction::ConstZone(v),
        S::InitialFill(a) => LanesInstruction::InitialFill(a),
        S::Fill(a) => LanesInstruction::Fill(a),
        S::Move(a) => LanesInstruction::Move(a),
        S::LocalRz(a) => LanesInstruction::LocalRz(a),
        S::LocalR(a) => LanesInstruction::LocalR(a),
        S::GlobalRz => LanesInstruction::GlobalRz,
        S::GlobalR => LanesInstruction::GlobalR,
        S::Cz => LanesInstruction::Cz,
        S::Measure(a) => LanesInstruction::Measure(a),
        S::AwaitMeasure => LanesInstruction::AwaitMeasure,
        S::NewArray(t, d0, d1) => LanesInstruction::NewArray(t, d0, d1),
        S::GetItem(n) => LanesInstruction::GetItem(n),
        S::SetDetector => LanesInstruction::SetDetector,
        S::SetObservable => LanesInstruction::SetObservable,
    }
}

/// The operands a [`LanesInstruction`] consumes, popped off the CPU stack by
/// the composite before the device runs.
///
/// This mirrors vihaco-cpu's `CPUMessage` and PPVM's `CircuitMessage`: the
/// instruction says *what*, the message carries *which atoms*.
#[derive(Debug, Clone, PartialEq, vihaco::Message)]
pub enum LanesMessage {
    /// No operands (`global_r`, `await_measure`, `set_detector`, …).
    None,
    /// A constant to push (`const_loc`, `const_lane`, `const_zone`).
    Constant(vihaco::Value),
    /// A group of locations (`initial_fill`, `fill`).
    Locations(Vec<LocationAddr>),
    /// A group of lanes (`move`).
    Lanes(Vec<LaneAddr>),
    /// A zone (`cz`) or a group of them (`measure`).
    Zones(Vec<ZoneAddr>),
    /// Rotation angles and the locations they apply to (`local_r`/`local_rz`).
    LocalRotation {
        angles: Vec<f64>,
        locations: Vec<LocationAddr>,
    },
    /// Rotation angles only (`global_r`/`global_rz`).
    GlobalRotation { angles: Vec<f64> },
    /// Raw stack values in **program order** — the order they were pushed, not
    /// the reverse order they were popped in.
    ///
    /// This is what the ops the machine does not interpret consume, and the
    /// only record of it: `new_array`'s elements, `get_item`'s array followed
    /// by its indices, and the single reference taken by `await_measure` /
    /// `set_detector` / `set_observable`.
    Values(Vec<vihaco::Value>),
}

/// What a lanes instruction did, for a consumer downstream of atom movement.
///
/// The arrangement ops are executed here, so they report the resulting state.
/// Everything else reports the request verbatim, for hardware — or a simulator
/// like PPVM — to carry out.
#[derive(Debug, Clone)]
pub enum LanesEffect {
    /// A value to push onto the CPU stack. The device cannot reach the stack
    /// itself; the composite performs the push (the same route PPVM uses to
    /// return measurement outcomes).
    Push(vihaco::Value),
    /// Atoms were placed or moved; carries the state afterwards.
    Arrangement(AtomStateData),
    /// The request was recorded but **not** simulated.
    ///
    /// Covers the quantum ops, which are deliberately out of scope here, and
    /// the array/measurement ops, whose representation is still being decided
    /// in <https://github.com/QuEraComputing/bloqade-lanes/issues/776>.
    /// Executing those would mean inventing the semantics that issue exists to
    /// settle, so they are surfaced rather than interpreted.
    ///
    /// This is the extension point for simulating them: see
    /// <https://github.com/QuEraComputing/bloqade-lanes/issues/1022> for the
    /// planned gate-recording and PPVM-tableau observers. `msg` carries the
    /// operands the instruction consumed, so such an observer has the whole
    /// request — which array `set_detector` referenced, which elements went
    /// into a `new_array` — and not just the opcode.
    NotSimulated {
        inst: LanesInstruction,
        msg: LanesMessage,
    },
}

/// How many values an instruction leaves on the stack.
///
/// The ops that are reported rather than executed still have to keep the
/// stack at the depth [`super::validate::simulate_stack`] predicts, or a
/// program that validates would underflow the moment it ran. They push
/// [`vihaco::Value::Undefined`] placeholders: the right *shape*, and a value
/// that cannot be mistaken for a simulated result.
fn result_count(inst: &LanesInstruction) -> u32 {
    use LanesInstruction as I;
    match inst {
        I::Measure(n) => *n,
        I::AwaitMeasure | I::NewArray(..) | I::GetItem(_) | I::SetDetector | I::SetObservable => 1,
        _ => 0,
    }
}

// ── Execution ─────────────────────────────────────────────────────────────────

#[vihaco::dispatch(instruction = LanesInstruction, message = LanesMessage, effect = LanesEffect)]
impl Lanes {
    fn execute(
        &mut self,
        inst: LanesInstruction,
        msg: LanesMessage,
    ) -> eyre::Result<vihaco::Effects<LanesEffect>> {
        use LanesInstruction as I;
        match (&inst, &msg) {
            // ---- Address constants: hand the value back for the stack ----
            (I::ConstLoc(v) | I::ConstLane(v), _) => Ok(vihaco::Effects::one(LanesEffect::Push(
                vihaco::Value::U64(*v),
            ))),
            (I::ConstZone(v), _) => Ok(vihaco::Effects::one(LanesEffect::Push(
                vihaco::Value::U32(*v),
            ))),

            // ---- Atom arrangement: the part we actually simulate ----
            (I::InitialFill(_) | I::Fill(_), LanesMessage::Locations(locations)) => {
                let qubits: Vec<(u32, LocationAddr)> = locations
                    .iter()
                    .enumerate()
                    .map(|(i, loc)| (self.next_qubit_id + i as u32, *loc))
                    .collect();
                let next = self
                    .atoms
                    .add_atoms(&qubits)
                    .map_err(|e| eyre::eyre!("{inst:?}: {e}"))?;
                self.next_qubit_id += locations.len() as u32;
                self.atoms = next.clone();
                Ok(vihaco::Effects::one(LanesEffect::Arrangement(next)))
            }
            (I::Move(_), LanesMessage::Lanes(lanes)) => {
                let arch = self
                    .arch
                    .as_ref()
                    .ok_or_else(|| eyre::eyre!("move requires an arch spec"))?;
                let next = self.atoms.apply_moves(lanes, arch).ok_or_else(|| {
                    eyre::eyre!("move is not executable against the current atom state")
                })?;
                self.atoms = next.clone();
                Ok(vihaco::Effects::one(LanesEffect::Arrangement(next)))
            }
            (I::InitialFill(_) | I::Fill(_) | I::Move(_), _) => Err(eyre::eyre!(
                "{inst:?} expects a location/lane group, got {msg:?}"
            )),

            // ---- Everything else is reported, not interpreted ----
            // The report comes first, then the placeholders that keep the
            // stack at the depth the static simulator predicts.
            _ => {
                let pushes = result_count(&inst);
                if pushes == 0 {
                    return Ok(vihaco::Effects::one(LanesEffect::NotSimulated {
                        inst,
                        msg,
                    }));
                }
                let mut effects = vec![LanesEffect::NotSimulated { inst, msg }];
                effects.extend(
                    std::iter::repeat_n(vihaco::Value::Undefined, pushes as usize)
                        .map(LanesEffect::Push),
                );
                Ok(vihaco::Effects::many(effects.into()))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chumsky::Parser as _;
    use vihaco_parser::Parse;

    fn parse(input: &str) -> LanesInstruction {
        lower(
            LanesSurfaceInstruction::parser()
                .parse(input)
                .into_result()
                .unwrap_or_else(|e| panic!("parse({input:?}) failed: {e:?}")),
        )
    }

    #[test]
    fn surface_parses_under_the_lanes_head() {
        assert_eq!(parse("lanes.move 2"), LanesInstruction::Move(2));
        assert_eq!(parse("lanes.cz"), LanesInstruction::Cz);
        assert_eq!(
            parse("lanes.new_array 1 3 0"),
            LanesInstruction::NewArray(1, 3, 0)
        );
    }

    #[test]
    fn address_constants_parse_as_hex_and_lower_to_integers() {
        assert_eq!(
            parse("lanes.const_loc 0x0000000001000000"),
            LanesInstruction::ConstLoc(0x0100_0000)
        );
        assert_eq!(
            parse("lanes.const_zone 0x00000007"),
            LanesInstruction::ConstZone(7)
        );
        // Decimal is not accepted where the format specifies hex.
        assert!(
            LanesSurfaceInstruction::parser()
                .parse("lanes.const_loc 16777216")
                .into_result()
                .is_err()
        );
    }

    #[test]
    fn prefix_tokens_disambiguate() {
        assert_eq!(parse("lanes.local_rz 1"), LanesInstruction::LocalRz(1));
        assert_eq!(parse("lanes.local_r 3"), LanesInstruction::LocalR(3));
        assert_eq!(parse("lanes.global_rz"), LanesInstruction::GlobalRz);
        assert_eq!(parse("lanes.global_r"), LanesInstruction::GlobalR);
    }

    #[test]
    fn the_head_is_required_and_cpu_ops_are_not_ours() {
        for bad in ["move 2", "cpu.move 2", "lanes.halt", "lanes.dup"] {
            assert!(
                LanesSurfaceInstruction::parser()
                    .parse(bad)
                    .into_result()
                    .is_err(),
                "{bad:?} should not parse as a lanes instruction"
            );
        }
    }
}
