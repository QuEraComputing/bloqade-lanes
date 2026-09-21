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
use vihaco_cpu::{CPU, SurfaceInstruction as CpuSurfaceInstruction};

use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::types::ArchSpec;

use super::container::LanesContext;
use super::device::{Lanes, LanesEffect, LanesInstruction, LanesMessage};
use super::program::LanesInfo;

/// The combined instruction set: one variant per device.
pub type MachineInstruction = lanes_machine::runtime::Instruction;

/// The combined surface syntax, parsed from `.sst`.
pub type MachineSurfaceInstruction = lanes_machine::syntax::Instruction;

/// `#[composite]` derives only `Debug` and `Clone` on its runtime enum, but both
/// device instruction sets *are* `PartialEq`, so comparing programs only needs
/// the two arms spelled out. Round-trip tests and the Python `__eq__` both rely
/// on it.
impl PartialEq for MachineInstruction {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cpu(a), Self::Cpu(b)) => a == b,
            (Self::Lanes(a), Self::Lanes(b)) => a == b,
            _ => false,
        }
    }
}

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

            // The device cannot reach the stack, so the machine does the
            // popping for it and the pushes come back as effects.
            I::Pop => {
                self.cpu.stack_pop()?;
                LanesMessage::None
            }
            I::Swap => LanesMessage::Values(vec![self.cpu.stack_pop()?, self.cpu.stack_pop()?]),

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

// ── Text rendering and introspection ──────────────────────────────────────────

/// vihaco-cpu's own `Display` emits bare mnemonics (`halt`, `const.f64 1.5`)
/// that its *parser* does not accept, so rendering is written here against the
/// surface grammar instead. The round-trip tests pin the two together.
fn cpu_type_text(ty: Type) -> &'static str {
    match ty {
        Type::Undefined => "undef",
        Type::String => "str",
        Type::Bool => "bool",
        Type::I64 => "i64",
        Type::U32 => "u32",
        Type::U64 => "u64",
        Type::F64 => "f64",
        Type::FunctionRef => "fn_ref",
        Type::HeapRef => "heap_ref",
    }
}

fn cpu_value_text(value: &Value) -> String {
    match value {
        // `{:?}` on f64 is round-trip exact; `{}` drops the `.0` on integral
        // floats, which would then re-parse as an integer.
        Value::F64(v) => format!("{v:?}"),
        Value::Bool(v) => v.to_string(),
        Value::I64(v) => v.to_string(),
        Value::U32(v) => v.to_string(),
        Value::U64(v) => v.to_string(),
        Value::String(v) | Value::FunctionRef(v) | Value::HeapRef(v) => v.to_string(),
        Value::Undefined => "undef".to_string(),
    }
}

fn cpu_text(inst: &vihaco_cpu::RuntimeInstruction) -> String {
    use vihaco_cpu::RuntimeInstruction as C;
    let typed = |op: &str, ty: Type| format!("{op} {}", cpu_type_text(ty));
    match inst {
        C::Span(a, b, c) => format!("span {a} {b} {c}"),
        C::Label(name) => format!("label @{}", name.as_str()),
        C::FunctionStart => "func_start".into(),
        C::FunctionEnd => "func_end".into(),
        C::Breakpoint => "breakpoint".into(),
        C::Branch(t) => format!("br @{t}"),
        C::ConditionalBranch(t, f) => format!("cond_br @{t}, @{f}"),
        C::Return(n) => format!("ret {n}"),
        C::IndirectCall => "call_indirect".into(),
        C::Call(arity, target) => format!("call {arity}, {target}"),
        C::Halt => "halt".into(),
        C::Print => "print".into(),
        C::Load(ty, addr) => format!("load {}, {addr}", cpu_type_text(*ty)),
        C::Store(ty, addr) => format!("store {}, {addr}", cpu_type_text(*ty)),
        C::Dup => "dup".into(),
        C::HeapAlloc(n) => format!("heap_alloc {n}"),
        C::GetItem => "get_item".into(),
        C::HeapDealloc => "heap_dealloc".into(),
        C::Const(ty, v) => format!("const {}, {}", cpu_type_text(*ty), cpu_value_text(v)),
        C::Add(t) => typed("add", *t),
        C::Sub(t) => typed("sub", *t),
        C::Mul(t) => typed("mul", *t),
        C::Div(t) => typed("div", *t),
        C::Rem(t) => typed("rem", *t),
        C::Neg(t) => typed("neg", *t),
        C::Shl(t) => typed("shl", *t),
        C::Shr(t) => typed("shr", *t),
        C::Rol(t) => typed("rol", *t),
        C::Ror(t) => typed("ror", *t),
        C::BitAnd(t) => typed("bitand", *t),
        C::BitOr(t) => typed("bitor", *t),
        C::BitXor(t) => typed("bitxor", *t),
        C::Not => "not".into(),
        C::And => "and".into(),
        C::Or => "or".into(),
        C::Xor => "xor".into(),
        C::Eq(t) => typed("eq", *t),
        C::Ne(t) => typed("ne", *t),
        C::Lt(t) => typed("lt", *t),
        C::Gt(t) => typed("gt", *t),
        C::Le(t) => typed("le", *t),
        C::Ge(t) => typed("ge", *t),
    }
}

fn lanes_text(inst: &LanesInstruction) -> String {
    use LanesInstruction as L;
    match inst {
        L::Pop => "pop".into(),
        L::Swap => "swap".into(),
        // Hex, fixed width, so addresses line up by eye.
        L::ConstLoc(v) => format!("const_loc 0x{v:016x}"),
        L::ConstLane(v) => format!("const_lane 0x{v:016x}"),
        L::ConstZone(v) => format!("const_zone 0x{v:08x}"),
        L::InitialFill(a) => format!("initial_fill {a}"),
        L::Fill(a) => format!("fill {a}"),
        L::Move(a) => format!("move {a}"),
        L::LocalRz(a) => format!("local_rz {a}"),
        L::LocalR(a) => format!("local_r {a}"),
        L::GlobalRz => "global_rz".into(),
        L::GlobalR => "global_r".into(),
        L::Cz => "cz".into(),
        L::Measure(a) => format!("measure {a}"),
        L::AwaitMeasure => "await_measure".into(),
        L::NewArray(t, d0, d1) => format!("new_array {t} {d0} {d1}"),
        L::GetItem(n) => format!("get_item {n}"),
        L::SetDetector => "set_detector".into(),
        L::SetObservable => "set_observable".into(),
    }
}

/// Render an instruction in the `.sst` surface grammar, including both the
/// device prefix and the dialect head — e.g. `lanes::lanes.move 2`.
pub fn to_sst_text(inst: &MachineInstruction) -> String {
    match inst {
        MachineInstruction::Cpu(i) => format!("cpu::cpu.{}", cpu_text(i)),
        MachineInstruction::Lanes(i) => format!("lanes::lanes.{}", lanes_text(i)),
    }
}

/// The device this instruction belongs to: `"cpu"` or `"lanes"`.
pub fn device_of(inst: &MachineInstruction) -> &'static str {
    match inst {
        MachineInstruction::Cpu(_) => "cpu",
        MachineInstruction::Lanes(_) => "lanes",
    }
}

/// Canonical opcode name, without any prefix — the key the Python decoder
/// dispatches on (`_visit_{op_name}`).
///
/// Two names deliberately differ from the text mnemonic, because the decoder
/// depends on them:
///
/// - the constants stay `const_float` / `const_int` rather than vihaco-cpu's
///   single typed `const`, since the decoder pushes a different value type for
///   each and the mnemonic alone would not say which;
/// - `return` keeps its spelling rather than vihaco-cpu's `ret`.
pub fn op_name(inst: &MachineInstruction) -> String {
    match inst {
        MachineInstruction::Cpu(vihaco_cpu::RuntimeInstruction::Return(_)) => "return".into(),
        MachineInstruction::Cpu(vihaco_cpu::RuntimeInstruction::Const(ty, _)) => match ty {
            Type::F64 => "const_float".into(),
            Type::I64 => "const_int".into(),
            other => format!("const_{}", cpu_type_text(*other)),
        },
        MachineInstruction::Cpu(i) => cpu_text(i)
            .split([' ', ','])
            .next()
            .unwrap_or_default()
            .to_string(),
        MachineInstruction::Lanes(i) => lanes_text(i)
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string(),
    }
}

/// Lower a parsed instruction to its executable form.
///
/// `#[composite]` generates the surface and runtime enums independently and no
/// conversion between them, so — as with the device — this is written out.
pub fn lower(inst: MachineSurfaceInstruction) -> eyre::Result<MachineInstruction> {
    Ok(match inst {
        MachineSurfaceInstruction::Cpu(i) => MachineInstruction::Cpu(lower_cpu(i)?),
        MachineSurfaceInstruction::Lanes(i) => MachineInstruction::Lanes(super::device::lower(i)),
    })
}

/// Lower a parsed vihaco-cpu instruction to its runtime form.
///
/// The surface form is lexical — `SurfaceValue` is a raw token and branch
/// targets are identifiers — so this is where a constant becomes a typed
/// [`Value`]. Symbolic control flow is rejected: resolving a label to an address
/// needs a symbol table for the whole function, which a per-instruction lowering
/// does not have. A lanes program has no control flow, so nothing we emit hits
/// this path.
fn lower_cpu(inst: CpuSurfaceInstruction) -> eyre::Result<vihaco_cpu::RuntimeInstruction> {
    use CpuSurfaceInstruction as S;
    use vihaco_cpu::RuntimeInstruction as R;

    let ty = |t: vihaco_cpu::SurfaceType| -> Type {
        match t {
            vihaco_cpu::SurfaceType::Undefined => Type::Undefined,
            vihaco_cpu::SurfaceType::String => Type::String,
            vihaco_cpu::SurfaceType::Bool => Type::Bool,
            vihaco_cpu::SurfaceType::I64 => Type::I64,
            vihaco_cpu::SurfaceType::U32 => Type::U32,
            vihaco_cpu::SurfaceType::U64 => Type::U64,
            vihaco_cpu::SurfaceType::F64 => Type::F64,
            vihaco_cpu::SurfaceType::FunctionRef => Type::FunctionRef,
            vihaco_cpu::SurfaceType::HeapRef => Type::HeapRef,
        }
    };

    Ok(match inst {
        S::Const(t, v) => {
            let text = match &v {
                vihaco_cpu::SurfaceValue::Bare(token) => token.0.clone(),
                vihaco_cpu::SurfaceValue::Quoted(s) => s.0.clone(),
            };
            let ty = ty(t);
            let value = match ty {
                Type::F64 => Value::F64(text.parse()?),
                Type::I64 => Value::I64(text.parse()?),
                Type::U64 => Value::U64(text.parse()?),
                Type::U32 => Value::U32(text.parse()?),
                Type::Bool => Value::Bool(text.parse()?),
                other => {
                    return Err(eyre::eyre!(
                        "const of type {} is not supported in a lanes program",
                        cpu_type_text(other)
                    ));
                }
            };
            R::Const(ty, value)
        }
        S::Span(a, b, c) => R::Span(a, b, c),
        S::FunctionStart => R::FunctionStart,
        S::FunctionEnd => R::FunctionEnd,
        S::Breakpoint => R::Breakpoint,
        S::Return(n) => R::Return(n),
        S::IndirectCall => R::IndirectCall,
        S::Halt => R::Halt,
        S::Print => R::Print,
        S::Load(t, addr) => R::Load(ty(t), addr),
        S::Store(t, addr) => R::Store(ty(t), addr),
        S::Dup => R::Dup,
        S::HeapAlloc(n) => R::HeapAlloc(n),
        S::GetItem => R::GetItem,
        S::HeapDealloc => R::HeapDealloc,
        S::Add(t) => R::Add(ty(t)),
        S::Sub(t) => R::Sub(ty(t)),
        S::Mul(t) => R::Mul(ty(t)),
        S::Div(t) => R::Div(ty(t)),
        S::Rem(t) => R::Rem(ty(t)),
        S::Neg(t) => R::Neg(ty(t)),
        S::Shl(t) => R::Shl(ty(t)),
        S::Shr(t) => R::Shr(ty(t)),
        S::Rol(t) => R::Rol(ty(t)),
        S::Ror(t) => R::Ror(ty(t)),
        S::BitAnd(t) => R::BitAnd(ty(t)),
        S::BitOr(t) => R::BitOr(ty(t)),
        S::BitXor(t) => R::BitXor(ty(t)),
        S::Not => R::Not,
        S::And => R::And,
        S::Or => R::Or,
        S::Xor => R::Xor,
        S::Eq(t) => R::Eq(ty(t)),
        S::Ne(t) => R::Ne(ty(t)),
        S::Lt(t) => R::Lt(ty(t)),
        S::Gt(t) => R::Gt(ty(t)),
        S::Le(t) => R::Le(ty(t)),
        S::Ge(t) => R::Ge(ty(t)),
        S::Label(_) | S::Branch(_) | S::ConditionalBranch(_, _) | S::Call(_, _) => {
            return Err(eyre::eyre!(
                "symbolic control flow needs a label table and is not supported \
                 in a lanes program"
            ));
        }
    })
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
