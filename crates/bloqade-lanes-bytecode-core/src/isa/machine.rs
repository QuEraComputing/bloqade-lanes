//! The Bloqade Lanes machine: vihaco-cpu's `CPU` plus our atom-movement device.
//!
//! This is the idiomatic vihaco composition, and the same shape PPVM uses: each
//! device owns its instruction set, and `#[composite]` generates the combined
//! program type, the text parser (an alternation over the devices' parsers) and
//! the section loaders.
//!
//! ```text
//! cpu::cpu.const u64, 0      <- vihaco-cpu's device
//! lanes::lanes.move 2        <- ours
//! ```
//!
//! The `<field>::` prefix is the device field name; the second half is that
//! device's own dialect head.
//!
//! ## Who does what
//!
//! Instruction operands live on the CPU stack, so a device never reads them
//! directly. [`LanesMachine::resolve_lanes`] pops them and packs them into a
//! [`LanesMessage`]; the device executes; the machine then applies any
//! [`LanesEffect::Push`] back onto the stack. That round trip is the reason
//! address constants can stay lanes instructions while still behaving like
//! stack pushes.
//!
//! The pop rules here are the same ones [`super::validate::simulate_stack`]
//! type-checks statically — that simulator is the static half of this.

use vihaco::traits::StackMemory;
use vihaco::{Effects, GeneratedComponent, ProgramImage, Type, Value, composite};
use vihaco_cpu::CPU;

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::types::ArchSpec;

use super::container::LanesContext;
use super::device::{Lanes, LanesEffect, LanesInstruction, LanesMessage};
use super::program::LanesInfo;

/// The combined instruction set: one variant per device.
pub type MachineInstruction = lanes_machine::runtime::Instruction;

/// The combined surface syntax, parsed from `.sst`.
pub type MachineSurfaceInstruction = lanes_machine::syntax::Instruction;

#[composite]
#[derive(Default)]
pub struct LanesMachine {
    /// Required by `#[composite]`: the loaded program plus its constants and
    /// device info. Unread until the bytecode/text layers are wired onto
    /// `MachineInstruction`.
    #[allow(dead_code)]
    loader: ProgramImage<MachineInstruction, LanesContext, Value, Type, LanesInfo>,

    #[device(0x00)]
    cpu: CPU,

    #[device(0x01)]
    lanes: Lanes,
}

impl LanesMachine {
    /// Point the machine at an architecture. `move` cannot resolve a lane into
    /// (src, dst) endpoints without one, so it fails until this is set.
    pub fn with_arch(mut self, arch: ArchSpec) -> Self {
        self.lanes.arch = Some(arch);
        self
    }

    /// Pop the operands `inst` consumes and pack them into its message.
    ///
    /// Pops are in reverse push order throughout: the last value pushed is the
    /// first popped, so a group read back from the stack is reversed to restore
    /// program order.
    fn resolve_lanes(&mut self, inst: &LanesInstruction) -> eyre::Result<LanesMessage> {
        use LanesInstruction as I;
        Ok(match inst {
            // Constants carry their operand in the instruction word.
            I::ConstLoc(_) | I::ConstLane(_) | I::ConstZone(_) => LanesMessage::None,

            I::InitialFill(n) | I::Fill(n) => LanesMessage::Locations(self.pop_locations(*n)?),
            I::Move(n) => LanesMessage::Lanes(self.pop_lanes(*n)?),

            // Angles sit above the locations: `local_r` pops axis then
            // rotation, then the location group beneath them.
            I::LocalR(n) => LanesMessage::LocalRotation {
                angles: self.pop_floats(2)?,
                locations: self.pop_locations(*n)?,
            },
            I::LocalRz(n) => LanesMessage::LocalRotation {
                angles: self.pop_floats(1)?,
                locations: self.pop_locations(*n)?,
            },
            I::GlobalR => LanesMessage::GlobalRotation {
                angles: self.pop_floats(2)?,
            },
            I::GlobalRz => LanesMessage::GlobalRotation {
                angles: self.pop_floats(1)?,
            },

            I::Cz => LanesMessage::Zones(self.pop_zones(1)?),
            I::Measure(n) => LanesMessage::Zones(self.pop_zones(*n)?),

            // `new_array` consumes dim0×dim1 elements (dim1 = 0 means 1-D);
            // `get_item` consumes its indices plus the array reference. Both
            // are reported rather than executed — see `LanesEffect::NotSimulated`.
            I::NewArray(_, dim0, dim1) => {
                let count = dim0 * if *dim1 == 0 { 1 } else { *dim1 };
                for _ in 0..count {
                    self.cpu.stack_pop()?;
                }
                LanesMessage::Arity(count)
            }
            I::GetItem(n) => {
                for _ in 0..*n + 1 {
                    self.cpu.stack_pop()?;
                }
                LanesMessage::Arity(*n)
            }

            I::AwaitMeasure | I::SetDetector | I::SetObservable => {
                self.cpu.stack_pop()?;
                LanesMessage::None
            }
        })
    }

    fn pop_u64(&mut self) -> eyre::Result<u64> {
        match self.cpu.stack_pop()? {
            Value::U64(v) => Ok(v),
            v => Err(eyre::eyre!("expected a packed u64 address, got {v:?}")),
        }
    }

    fn pop_floats(&mut self, n: u32) -> eyre::Result<Vec<f64>> {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match self.cpu.stack_pop()? {
                Value::F64(v) => out.push(v),
                v => return Err(eyre::eyre!("expected an angle (f64), got {v:?}")),
            }
        }
        out.reverse();
        Ok(out)
    }

    fn pop_locations(&mut self, n: u32) -> eyre::Result<Vec<LocationAddr>> {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            out.push(LocationAddr::decode(self.pop_u64()?));
        }
        out.reverse();
        Ok(out)
    }

    fn pop_lanes(&mut self, n: u32) -> eyre::Result<Vec<LaneAddr>> {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            out.push(LaneAddr::decode_u64(self.pop_u64()?));
        }
        out.reverse();
        Ok(out)
    }

    fn pop_zones(&mut self, n: u32) -> eyre::Result<Vec<ZoneAddr>> {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            match self.cpu.stack_pop()? {
                Value::U32(v) => out.push(ZoneAddr::decode(v)),
                v => return Err(eyre::eyre!("expected a zone address (u32), got {v:?}")),
            }
        }
        out.reverse();
        Ok(out)
    }

    /// Apply a device effect. Only [`LanesEffect::Push`] touches the machine;
    /// the rest are observations for the caller.
    fn apply(&mut self, effects: &Effects<LanesEffect>) -> eyre::Result<()> {
        let mut push = |effect: &LanesEffect| {
            if let LanesEffect::Push(value) = effect {
                self.cpu.stack_push(*value);
            }
        };
        match effects {
            Effects::None => {}
            Effects::One(effect) => push(effect),
            Effects::Many(effects) => effects.iter().for_each(push),
        }
        Ok(())
    }

    /// Run one lanes instruction end to end: pop its operands, execute, then
    /// apply whatever comes back.
    pub fn step_lanes(&mut self, inst: LanesInstruction) -> eyre::Result<Effects<LanesEffect>> {
        let msg = self.resolve_lanes(&inst)?;
        let effects = self.lanes.execute_generated(inst, msg)?;
        self.apply(&effects)?;
        Ok(effects)
    }

    /// The atom arrangement as it currently stands.
    pub fn atoms(&self) -> &crate::atom_state::AtomStateData {
        &self.lanes.atoms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::addr::LocationAddr;
    use crate::isa::device::LanesInstruction as I;

    const SIMPLE_ARCH_JSON: &str = include_str!("../../../../examples/arch/simple.json");

    fn machine() -> LanesMachine {
        LanesMachine::default()
            .with_arch(ArchSpec::from_json(SIMPLE_ARCH_JSON).expect("simple.json should parse"))
    }

    fn loc(zone_id: u32, word_id: u32, site_id: u32) -> u64 {
        LocationAddr {
            zone_id,
            word_id,
            site_id,
        }
        .encode()
    }

    #[test]
    fn const_push_then_initial_fill_places_atoms() {
        // The whole point of the composite: address constants land on the CPU's
        // stack, and the lanes device reads them back off it.
        let mut m = machine();
        m.step_lanes(I::ConstLoc(loc(0, 0, 0))).unwrap();
        m.step_lanes(I::ConstLoc(loc(0, 0, 1))).unwrap();

        let effects = m.step_lanes(I::InitialFill(2)).unwrap();
        assert!(matches!(effects, Effects::One(LanesEffect::Arrangement(_))));

        // Both sites are now occupied, in program order.
        assert_eq!(
            m.atoms().get_qubit(&LocationAddr::decode(loc(0, 0, 0))),
            Some(0)
        );
        assert_eq!(
            m.atoms().get_qubit(&LocationAddr::decode(loc(0, 0, 1))),
            Some(1)
        );
    }

    #[test]
    fn initial_fill_underflows_without_enough_constants() {
        let mut m = machine();
        m.step_lanes(I::ConstLoc(loc(0, 0, 0))).unwrap();
        assert!(m.step_lanes(I::InitialFill(2)).is_err());
    }

    #[test]
    fn a_location_where_a_zone_belongs_is_rejected() {
        // `cz` wants a zone (u32); a location constant pushes a u64, so the
        // type mismatch surfaces at resolve time rather than silently decoding.
        let mut m = machine();
        m.step_lanes(I::ConstLoc(loc(0, 0, 0))).unwrap();
        let err = m.step_lanes(I::Cz).unwrap_err().to_string();
        assert!(err.contains("zone address"), "got {err}");
    }

    #[test]
    fn quantum_ops_are_reported_not_simulated() {
        let mut m = machine();
        m.step_lanes(I::ConstZone(0)).unwrap();
        let effects = m.step_lanes(I::Cz).unwrap();
        assert!(matches!(
            effects,
            Effects::One(LanesEffect::NotSimulated {
                inst: I::Cz,
                msg: LanesMessage::Zones(_)
            })
        ));
    }

    #[test]
    fn move_without_an_arch_is_an_error() {
        // `move` cannot resolve a lane into endpoints without a spec.
        let mut m = LanesMachine::default();
        m.step_lanes(I::ConstLane(0)).unwrap();
        let err = m.step_lanes(I::Move(1)).unwrap_err().to_string();
        assert!(err.contains("arch spec"), "got {err}");
    }
}
