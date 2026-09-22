//! Architecture-dependent validation for the vihaco-backed ISA.
//!
//! These checks gate a [`Program`] against an [`ArchSpec`]. They are
//! **arch-dependent by design**: when no arch spec is supplied ([`validate`]
//! called with `None`), every check here is skipped and an empty error list is
//! returned — an arch-free program is considered fine at this layer.
//!
//! [`validate`] runs the arch-dependent capability + address checks below.
//! [`validate_structure`] adds arch-independent structural checks, and
//! [`simulate_stack`] adds optional stack-type simulation (underflow, type
//! mismatches, and lane/location group validation).
//!
//! ## Capability checks
//!
//! - **`feed_forward` → multiple measurements.** Without mid-circuit classical
//!   feedback the hardware can only run straight-line code, so at most one
//!   `measure` may appear.
//!
//!   The companion rule against branch/call instructions
//!   ([`ControlFlowRequiresFeedForward`](ValidationError::ControlFlowRequiresFeedForward))
//!   rejects control flow for the same reason. It was unreachable for as long
//!   as the ISA had no such instructions; adopting the calling convention
//!   brought it back.
//! - **`atom_reloading` → `fill`.** Without atom reloading, refilling atoms
//!   after the initial fill is unsupported.
//!
//! ## What the stack simulation does not cover
//!
//! [`simulate_stack`] is a linear pass over each function from an empty stack,
//! so it stops at the first branch or call and skips any function a `call`
//! passes operands to. Both are conservative: it reports nothing rather than
//! something it cannot justify from a state it does not have. Restoring that
//! coverage needs a CFG walk that merges state at join points and models calls
//! against the frame — see
//! <https://github.com/QuEraComputing/bloqade-lanes/issues/1026>.
//!
//! ## Address checks
//!
//! Every `const_loc` / `const_lane` / `const_zone` operand is decoded and
//! checked against the architecture's topology via
//! [`ArchSpec::check_location`] / [`check_lane`](ArchSpec::check_lane) /
//! [`check_zone`](ArchSpec::check_zone) — invalid zones, words, sites, lanes,
//! and AOD constraints are reported with the arch layer's own message.

use std::collections::HashSet;
use std::fmt;

use super::device::LanesInstruction as L;
use super::machine::MachineInstruction as M;
use super::program::Program;
use crate::arch::addr::{LaneAddr, LocationAddr, ZoneAddr};
use crate::arch::query::{LaneGroupError, LocationGroupError};
use crate::arch::types::ArchSpec;
use vihaco::{Type, Value};
use vihaco_cpu::RuntimeInstruction as C;

/// Value type tags tracked by the [`simulate_stack`] type simulator. These
/// mirror the stack value kinds the runtime distinguishes.
pub mod tag {
    pub const FLOAT: u8 = 0x0;
    pub const INT: u8 = 0x1;
    pub const ARRAY_REF: u8 = 0x2;
    pub const LOCATION: u8 = 0x3;
    pub const LANE: u8 = 0x4;
    pub const ZONE: u8 = 0x5;
    pub const MEASURE_FUTURE: u8 = 0x6;
    pub const DETECTOR_REF: u8 = 0x7;
    pub const OBSERVABLE_REF: u8 = 0x8;
    pub const MEASUREMENT_RESULT: u8 = 0x9;
}

/// An arch-dependent validation failure, tagged with the offending
/// instruction's program counter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    /// A `br`, `cond_br`, `call` or `call_indirect` appears but
    /// `feed_forward` is disabled.
    ///
    /// Without mid-circuit classical feedback the hardware can only run
    /// straight-line code. This was documented as unreachable for as long as
    /// the ISA had no control flow to reject; adopting the calling convention
    /// brought it back.
    ControlFlowRequiresFeedForward {
        pc: usize,
        /// The offending mnemonic (e.g. `"cond_br"`).
        mnemonic: &'static str,
    },
    /// More than one `measure` appears but `feed_forward` is disabled.
    MultipleMeasuresRequireFeedForward { pc: usize },
    /// A `fill` appears but `atom_reloading` is disabled.
    FillRequiresAtomReloading { pc: usize },
    /// A `const_loc` operand does not name a valid location in the arch spec.
    InvalidLocation { pc: usize, message: String },
    /// A `const_lane` operand does not name a valid lane in the arch spec.
    InvalidLane { pc: usize, message: String },
    /// A `const_zone` operand does not name a valid zone in the arch spec.
    InvalidZone { pc: usize, message: String },

    // ---- structural (arch-independent) ----
    /// `new_array` dim0 must be greater than zero.
    NewArrayZeroDim0 { pc: usize },
    /// `new_array` type_tag exceeds the maximum value tag.
    NewArrayInvalidTypeTag { pc: usize, type_tag: u32 },
    /// `new_array` declares more elements than [`MAX_ARRAY_ELEMENTS`].
    NewArrayTooManyElements { pc: usize, count: u64 },
    /// `get_item` takes an index count outside `1..=`[`MAX_GET_ITEM_DIMS`].
    GetItemInvalidDims { pc: usize, ndims: u32 },
    /// `load`/`store` names a local past [`MAX_LOCAL_INDEX`].
    LocalIndexOutOfRange {
        pc: usize,
        /// The offending mnemonic (`"load"` or `"store"`).
        mnemonic: &'static str,
        index: u32,
    },
    /// An instruction sits outside every function's extent — after the last
    /// `func_end`, or in a stream with no markers at all. Unreachable, and
    /// unrenderable: [`super::text::to_text`] emits it after the closing brace.
    CodeOutsideFunction { pc: usize },
    /// A `br`/`cond_br` names an address outside the code, or a `call` names
    /// one that does not begin a function.
    InvalidControlFlowTarget {
        pc: usize,
        target: u32,
        /// What the target should have been: `"an address in the code"` or
        /// `"a function entry"`.
        expected: &'static str,
    },
    /// `initial_fill` is not the first non-constant instruction.
    InitialFillNotFirst { pc: usize },
    /// The program has no instructions (and therefore no terminator).
    EmptyProgram,
    /// Some path through a function runs off its end instead of reaching
    /// `return` or `halt`. Per function, not per program: `func_end` is a
    /// no-op, so falling off it runs whatever was laid out next.
    MissingTerminator { pc: usize },
    /// An instruction follows a `return`/`halt` and is unreachable.
    UnreachableInstruction { pc: usize },

    // ---- stack-type simulation (only via `simulate_stack`) ----
    /// An instruction popped from an empty stack.
    StackUnderflow { pc: usize },
    /// A popped value had the wrong type tag (see [`tag`]).
    TypeMismatch { pc: usize, expected: u8, got: u8 },
    /// A `local_r`/`local_rz`/`fill`/`initial_fill` location group is invalid.
    LocationGroupValidation {
        pc: usize,
        error: LocationGroupError,
    },
    /// A `move` lane group is invalid (duplicate, inconsistent, AOD, …).
    LaneGroupValidation { pc: usize, error: LaneGroupError },
}

/// Maximum valid `new_array` element type tag ([`tag::MEASUREMENT_RESULT`]).
const MAX_TYPE_TAG: u32 = tag::MEASUREMENT_RESULT as u32;

/// Maximum number of elements a `new_array` may declare.
///
/// `dim0` and `dim1` are attacker-controlled `u32`s read straight out of the
/// instruction word, and their product drives a pop loop. Without a bound, a
/// 28-byte program can make the validator pop four billion times, and the
/// product itself overflows `u32`. An array holds one element per measured
/// site, so a million is already orders of magnitude past any physical
/// architecture — the bound exists to make a malformed word a diagnosis
/// rather than a hang.
pub const MAX_ARRAY_ELEMENTS: u64 = 1 << 20;

/// Maximum number of indices a `get_item` may take.
///
/// `new_array` carries exactly two dimension fields, so an array is at most
/// 2-D and one or two indices is the only well-formed shape. The Python
/// `stack_move.GetItem` documents the same invariant and defers enforcement
/// here.
pub const MAX_GET_ITEM_DIMS: u32 = 2;

/// Highest local index a `load` or `store` may name, so a frame addresses at
/// most 1024 locals.
///
/// vihaco's locals are a window into the operand stack starting at the current
/// frame's base, and `store` *grows the stack to reach its index*: `op_store`
/// calls `get_local_mut(index)`, which `resize`s to `base + index + 1` and
/// writes `Undefined` into every new slot. The index is therefore a memory
/// request read straight out of the instruction word — at the 16 bytes per slot
/// measured in #1032, `store u64, 4294967295` touches about 68 GB from a
/// 12-byte program. And because `resize` writes rather than reserves, the pages
/// are resident on every platform, so this does not hide behind macOS's lazy
/// commit the way the earlier `Vec::with_capacity` bounds did.
///
/// A lanes function's locals are its arguments, and the pipeline emits no
/// `load`/`store` at all, so every legitimate index is a handful. The bound is
/// set well above that rather than at it: its job is to keep a malformed
/// operand from becoming an allocation, not to impose a calling convention on a
/// hand-written program. The ceiling costs about 16 KB of operand stack, which
/// is nothing, while still being three orders of magnitude past any function
/// the compiler will emit.
pub const MAX_LOCAL_INDEX: u32 = 1023;
/// Record a `br`/`cond_br` target that leaves the branch's own function.
///
/// Testing only `target < code.len()` let a branch land anywhere in the
/// stream: a trailing `cpu::cpu.label` resolves to the function's own
/// `func_end`, and a target past it enters the *next* function's body without
/// a frame — the caller's locals still in place, no `ret` to come back to.
/// A branch is intra-procedural; leaving the function is what `call` is for.
fn check_branch_target(
    errors: &mut Vec<ValidationError>,
    owner: &[Option<usize>],
    pc: usize,
    target: u32,
) {
    let same_function = owner
        .get(target as usize)
        .copied()
        .flatten()
        .is_some_and(|t| owner.get(pc).copied().flatten() == Some(t));
    if !same_function {
        errors.push(ValidationError::InvalidControlFlowTarget {
            pc,
            target,
            expected: "an address in the same function",
        });
    }
}

/// Element count of a `new_array`, in `u64` so the product cannot overflow.
/// `dim1 == 0` means a 1-D array.
pub(crate) fn array_element_count(dim0: u32, dim1: u32) -> u64 {
    dim0 as u64 * if dim1 == 0 { 1 } else { dim1 as u64 }
}

/// Record a `load`/`store` local index past [`MAX_LOCAL_INDEX`].
fn check_local_index(
    errors: &mut Vec<ValidationError>,
    pc: usize,
    mnemonic: &'static str,
    index: u32,
) {
    if index > MAX_LOCAL_INDEX {
        errors.push(ValidationError::LocalIndexOutOfRange {
            pc,
            mnemonic,
            index,
        });
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ValidationError::ControlFlowRequiresFeedForward { pc, mnemonic } => write!(
                f,
                "pc {pc}: control-flow instruction '{mnemonic}' requires feed_forward capability"
            ),
            ValidationError::MultipleMeasuresRequireFeedForward { pc } => write!(
                f,
                "pc {pc}: multiple measure instructions require feed_forward capability"
            ),
            ValidationError::FillRequiresAtomReloading { pc } => {
                write!(
                    f,
                    "pc {pc}: fill instruction requires atom_reloading capability"
                )
            }
            ValidationError::InvalidLocation { pc, message } => {
                write!(f, "pc {pc}: invalid location: {message}")
            }
            ValidationError::InvalidLane { pc, message } => {
                write!(f, "pc {pc}: invalid lane: {message}")
            }
            ValidationError::InvalidZone { pc, message } => {
                write!(f, "pc {pc}: invalid zone: {message}")
            }
            ValidationError::NewArrayZeroDim0 { pc } => {
                write!(f, "pc {pc}: new_array dim0 must be > 0")
            }
            ValidationError::NewArrayInvalidTypeTag { pc, type_tag } => {
                write!(f, "pc {pc}: invalid new_array type tag {type_tag}")
            }
            ValidationError::NewArrayTooManyElements { pc, count } => write!(
                f,
                "pc {pc}: new_array declares {count} elements, more than the \
                 maximum of {MAX_ARRAY_ELEMENTS}"
            ),
            ValidationError::GetItemInvalidDims { pc, ndims } => write!(
                f,
                "pc {pc}: get_item takes 1..={MAX_GET_ITEM_DIMS} indices, got {ndims}"
            ),
            ValidationError::LocalIndexOutOfRange {
                pc,
                mnemonic,
                index,
            } => write!(
                f,
                "pc {pc}: {mnemonic} takes a local index 0..={MAX_LOCAL_INDEX}, got {index}"
            ),
            ValidationError::CodeOutsideFunction { pc } => {
                write!(f, "pc {pc}: instruction is outside any function")
            }
            ValidationError::InvalidControlFlowTarget {
                pc,
                target,
                expected,
            } => write!(f, "pc {pc}: control-flow target {target} is not {expected}"),
            ValidationError::InitialFillNotFirst { pc } => write!(
                f,
                "pc {pc}: initial_fill must be the first non-constant instruction"
            ),
            ValidationError::EmptyProgram => {
                write!(
                    f,
                    "program has no instructions: missing return or halt terminator"
                )
            }
            ValidationError::MissingTerminator { pc } => {
                write!(f, "pc {pc}: function must end with return or halt")
            }
            ValidationError::UnreachableInstruction { pc } => {
                write!(f, "pc {pc}: unreachable instruction after return or halt")
            }
            ValidationError::StackUnderflow { pc } => write!(f, "pc {pc}: stack underflow"),
            ValidationError::TypeMismatch { pc, expected, got } => write!(
                f,
                "pc {pc}: type mismatch: expected tag 0x{expected:x}, got 0x{got:x}"
            ),
            ValidationError::LocationGroupValidation { pc, error } => {
                write!(f, "pc {pc}: {error}")
            }
            ValidationError::LaneGroupValidation { pc, error } => write!(f, "pc {pc}: {error}"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Validate a program's arch-dependent constraints (capabilities + addresses).
///
/// When `arch` is `None`, all checks are skipped and an empty list is returned.
/// Otherwise every violation is collected in program order (the validator does
/// not stop at the first error), so callers can report them all at once.
pub fn validate(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
    let Some(arch) = arch else {
        return Vec::new();
    };

    let mut errors = Vec::new();
    let mut measure_count = 0u32;
    let mut initial_fill_count = 0u32;

    // The capability rules describe *the run*, so they are scoped to the code
    // that runs. With `feed_forward` off the control-flow arm below rejects
    // every `call`, so only the entry function can execute — counting within
    // its span is exact wherever the measure rule is active, and stops an
    // uncallable function's `measure` from condemning a program that measures
    // once.
    let entry = program
        .main_function
        .and_then(|i| program.functions.get(i as usize))
        .map(|f| (f.start_address as usize, f.end_address as usize));
    let runs = |pc: usize| entry.is_none_or(|(start, end)| pc >= start && pc < end);

    for (pc, inst) in program.code.iter().enumerate() {
        match inst {
            // ---- capability checks ----
            // Without mid-circuit classical feedback the hardware runs
            // straight-line code only, so a branch or a call is unrunnable.
            // This arm did not exist while the ISA had no such instructions;
            // now that a program can declare functions and branch between
            // them, its absence let every one of them through.
            //
            // Guarded on [`is_control_flow`] rather than re-listing the
            // variants, so a new control-flow op is gated here the moment it
            // joins that predicate, instead of silently passing the rule that
            // exists to reject it.
            inst if is_control_flow(inst) && !arch.feed_forward => {
                errors.push(ValidationError::ControlFlowRequiresFeedForward {
                    pc,
                    mnemonic: super::machine::op_name(inst),
                });
            }
            M::Lanes(L::Measure(_)) => {
                // A textual count over the code stream, which is exact only
                // because the arm above rejects every branch and call when
                // `feed_forward` is false: without control flow the code runs
                // once, top to bottom. With `feed_forward` on, repeats are
                // allowed anyway, so the count does not have to be exact.
                if runs(pc) {
                    measure_count += 1;
                    if !arch.feed_forward && measure_count > 1 {
                        errors.push(ValidationError::MultipleMeasuresRequireFeedForward { pc });
                    }
                }
            }
            M::Lanes(L::Fill(_)) if !arch.atom_reloading => {
                errors.push(ValidationError::FillRequiresAtomReloading { pc });
            }
            // A *second* `initial_fill` is a refill, whatever it is spelled.
            // The whole-program "must be first" ordering rule used to make
            // that unreachable; scoping it to a function did not, so a helper
            // could reload atoms on hardware that cannot.
            //
            // Counted across the whole program, unlike `measure` above. The
            // measure rule is only active when `feed_forward` is off, which
            // bans `call` and leaves the entry function as the only code that
            // runs — so entry-scoping is exact there. This rule is active with
            // `feed_forward` *on*, where a called function does run, and
            // deciding which functions are callable needs a call-graph walk.
            // Counting all of them can only over-report, which is the right
            // way for a capability gate to be wrong.
            M::Lanes(L::InitialFill(_)) => {
                initial_fill_count += 1;
                if !arch.atom_reloading && initial_fill_count > 1 {
                    errors.push(ValidationError::FillRequiresAtomReloading { pc });
                }
            }

            // ---- address checks ----
            M::Lanes(L::ConstLoc(bits)) => {
                if let Some(message) = arch.check_location(&LocationAddr::decode(*bits)) {
                    errors.push(ValidationError::InvalidLocation { pc, message });
                }
            }
            M::Lanes(L::ConstLane(bits)) => {
                for message in arch.check_lane(&LaneAddr::decode_u64(*bits)) {
                    errors.push(ValidationError::InvalidLane { pc, message });
                }
            }
            M::Lanes(L::ConstZone(bits)) => {
                if let Some(message) = arch.check_zone(&ZoneAddr::decode(*bits)) {
                    errors.push(ValidationError::InvalidZone { pc, message });
                }
            }

            _ => {}
        }
    }

    errors
}

/// True if `inst` transfers control somewhere the linear walk cannot follow.
fn is_control_flow(inst: &M) -> bool {
    matches!(
        inst,
        M::Cpu(C::Branch(_) | C::ConditionalBranch(_, _) | C::Call(_, _) | C::IndirectCall)
    )
}

/// True if `inst` terminates execution (`return` or `halt`).
fn is_terminator(inst: &M) -> bool {
    matches!(inst, M::Cpu(C::Return(_)) | M::Cpu(C::Halt))
}

/// True if `inst` only pushes a constant (and so may precede `initial_fill`).
fn is_constant_push(inst: &M) -> bool {
    matches!(
        inst,
        M::Lanes(L::ConstLoc(_))
            | M::Lanes(L::ConstLane(_))
            | M::Lanes(L::ConstZone(_))
            | M::Cpu(C::Const(Type::F64, Value::F64(_)))
            | M::Cpu(C::Const(Type::I64, Value::I64(_)))
    )
}

/// Validate a program's arch-independent structural rules: `new_array` operand
/// bounds, `initial_fill` ordering, and terminator/reachability. These never
/// consult an arch spec, so they always run.
pub fn validate_structure(program: &Program) -> Vec<ValidationError> {
    let mut errors = Vec::new();
    let mut seen_non_constant = false;
    let spans = function_spans(&program.code);

    // Which function owns each address, so a branch can be constrained to its
    // own. `pc -> span index`; `None` for code outside every function.
    let mut owner = vec![None; program.code.len()];
    for (i, span) in spans.iter().enumerate() {
        for slot in owner.iter_mut().take(span.end).skip(span.start) {
            *slot = Some(i);
        }
    }

    for (pc, inst) in program.code.iter().enumerate() {
        match inst {
            M::Lanes(L::NewArray(type_tag, dim0, dim1)) => {
                if *dim0 == 0 {
                    errors.push(ValidationError::NewArrayZeroDim0 { pc });
                }
                if *type_tag > MAX_TYPE_TAG {
                    errors.push(ValidationError::NewArrayInvalidTypeTag {
                        pc,
                        type_tag: *type_tag,
                    });
                }
                // Bound the element count here, where the operands are read,
                // so neither the stack simulator nor the machine has to guard
                // its own pop loop against a four-billion-element array.
                let count = array_element_count(*dim0, *dim1);
                if count > MAX_ARRAY_ELEMENTS {
                    errors.push(ValidationError::NewArrayTooManyElements { pc, count });
                }
                seen_non_constant = true;
            }
            M::Lanes(L::GetItem(ndims)) => {
                if *ndims == 0 || *ndims > MAX_GET_ITEM_DIMS {
                    errors.push(ValidationError::GetItemInvalidDims { pc, ndims: *ndims });
                }
                seen_non_constant = true;
            }
            // Bound the local index here, where the operand is read, so a
            // `store` cannot turn a one-word instruction into a multi-gigabyte
            // stack `resize` inside vihaco (see [`MAX_LOCAL_INDEX`]). `load`
            // only reads, so it fails cleanly either way — it is checked too
            // because a program naming a local that far out is malformed
            // regardless of which end of it is reached first.
            M::Cpu(C::Load(_, index)) => {
                check_local_index(&mut errors, pc, "load", *index);
                seen_non_constant = true;
            }
            M::Cpu(C::Store(_, index)) => {
                check_local_index(&mut errors, pc, "store", *index);
                seen_non_constant = true;
            }
            M::Lanes(L::InitialFill(_)) => {
                if seen_non_constant {
                    errors.push(ValidationError::InitialFillNotFirst { pc });
                }
                seen_non_constant = true;
            }
            // Branch and call targets are addresses, and nothing has checked
            // them: a `br` past the end of the code, or a `call` to an address
            // that begins no function, is a program the renderer cannot even
            // name — it has to invent `@L9` / `@F1`, producing text that will
            // not re-read.
            M::Cpu(C::Branch(t)) => {
                check_branch_target(&mut errors, &owner, pc, *t);
                seen_non_constant = true;
            }
            M::Cpu(C::ConditionalBranch(t, f)) => {
                check_branch_target(&mut errors, &owner, pc, *t);
                check_branch_target(&mut errors, &owner, pc, *f);
                seen_non_constant = true;
            }
            M::Cpu(C::Call(_, target)) => {
                if !matches!(
                    program.code.get(*target as usize),
                    Some(M::Cpu(C::FunctionStart))
                ) {
                    errors.push(ValidationError::InvalidControlFlowTarget {
                        pc,
                        target: *target,
                        expected: "a function entry",
                    });
                }
                seen_non_constant = true;
            }
            // A function boundary resets the ordering rule: `initial_fill`
            // must lead its own function, not the whole code stream.
            M::Cpu(C::FunctionStart) => seen_non_constant = false,
            M::Cpu(C::FunctionEnd) => {}
            inst if is_constant_push(inst) => {}
            _ => seen_non_constant = true,
        }
    }

    // Reachability and terminators are per *function*, and reachability
    // follows branches rather than latching on the first terminator.
    //
    // The latch this replaces was right while a program was one flat `@main`.
    // It condemned `cond_br`'s second arm — the `halt` ending the first arm
    // set the latch — and it could not see code a `br` jumped over. Following
    // the edges gets both right, and the edge extraction already exists for
    // the target checks above.
    let spans = function_spans(&program.code);

    // An instruction outside every function is unreachable *and* unrenderable:
    // `to_text` emits it after the last `}`, producing text that will not
    // re-read.
    let mut covered = vec![false; program.code.len()];
    for span in &spans {
        for slot in covered.iter_mut().take(span.end).skip(span.start) {
            *slot = true;
        }
    }
    for (pc, inside) in covered.iter().enumerate() {
        if !inside {
            errors.push(ValidationError::CodeOutsideFunction { pc });
        }
    }

    // A label names a position in a function's *body*. One sitting on a
    // `func_start` cannot be written down: the only spot before a function's
    // first instruction is the header line, so `to_text` emits it as the first
    // body label and it re-reads one address later — a different binary, no
    // error. One on a `func_end`, or outside every function, has no
    // instruction to mark at all.
    for label in &program.labels {
        let body = spans.iter().any(|span| {
            let body_end = if span.closed { span.end - 1 } else { span.end };
            (label.address as usize) > span.start && (label.address as usize) < body_end
        });
        if !body {
            errors.push(ValidationError::InvalidControlFlowTarget {
                pc: label.address as usize,
                target: label.address,
                expected: "an instruction inside a function body",
            });
        }
    }

    // "Empty" means no *body*: a program that is nothing but markers has
    // nothing to run, however many functions it declares.
    if !program.code.iter().any(|i| !is_marker(i)) {
        errors.push(ValidationError::EmptyProgram);
        return errors;
    }

    for span in &spans {
        walk_function(program, span, &mut errors);
    }

    errors
}

/// One function's extent in the code stream.
struct Span {
    /// Address of the `func_start`.
    start: usize,
    /// One past the last address the function owns.
    end: usize,
    /// Whether a `func_end` closed it. An unclosed final function still owns
    /// the rest of the code, and falls off its own end.
    closed: bool,
}

fn is_marker(inst: &M) -> bool {
    matches!(inst, M::Cpu(C::FunctionStart) | M::Cpu(C::FunctionEnd))
}

/// Entry addresses of every function some `call` passes operands to.
///
/// The arity lives at the call site, not on the callee — `resolve` records an
/// empty [`vihaco::module::Signature`] for every function — so the only way to
/// learn that a function receives operands is to read the calls that reach it.
/// A function called from several sites with different arities appears here if
/// *any* of them is nonzero, because one caller passing operands is enough to
/// make the empty-stack simulation wrong.
fn callee_arities(code: &[M]) -> HashSet<u32> {
    code.iter()
        .filter_map(|inst| match inst {
            M::Cpu(C::Call(arity, target)) if *arity > 0 => Some(*target),
            _ => None,
        })
        .collect()
}

/// Split the code into function extents, delimited by the markers.
fn function_spans(code: &[M]) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut open: Option<usize> = None;
    for (pc, inst) in code.iter().enumerate() {
        match inst {
            M::Cpu(C::FunctionStart) => {
                // A nested `func_start` closes nothing; the outer function
                // runs up to it and `resolve` rejects the construct anyway.
                if let Some(start) = open.replace(pc) {
                    spans.push(Span {
                        start,
                        end: pc,
                        closed: false,
                    });
                }
            }
            M::Cpu(C::FunctionEnd) => {
                if let Some(start) = open.take() {
                    spans.push(Span {
                        start,
                        end: pc + 1,
                        closed: true,
                    });
                }
            }
            _ => {}
        }
    }
    if let Some(start) = open {
        spans.push(Span {
            start,
            end: code.len(),
            closed: false,
        });
    }
    spans
}

/// Walk one function from its entry, following branch edges.
///
/// Reports every body address no path reaches, and a missing terminator when
/// any path runs off the end instead of hitting `ret`/`halt`. Both are
/// per-function: a second function's dead code no longer hides a first
/// function's missing terminator, which the old single either/or gate did.
fn walk_function(program: &Program, span: &Span, errors: &mut Vec<ValidationError>) {
    let body_end = if span.closed { span.end - 1 } else { span.end };
    let mut seen = vec![false; span.end - span.start];
    let mut work = vec![span.start];
    let mut falls_off = false;

    while let Some(pc) = work.pop() {
        // A target outside this function is reported by the target checks;
        // following it would attribute the callee's code to the caller.
        if pc < span.start || pc >= span.end || seen[pc - span.start] {
            continue;
        }
        seen[pc - span.start] = true;

        let inst = &program.code[pc];
        let fall_through = |work: &mut Vec<usize>, falls_off: &mut bool| {
            if pc + 1 >= body_end {
                *falls_off = true;
            }
            if pc + 1 < span.end {
                work.push(pc + 1);
            }
        };
        match inst {
            M::Cpu(C::FunctionEnd) => falls_off = true,
            M::Cpu(C::Branch(t)) => work.push(*t as usize),
            M::Cpu(C::ConditionalBranch(t, f)) => {
                work.push(*t as usize);
                work.push(*f as usize);
            }
            inst if is_terminator(inst) => {}
            _ => fall_through(&mut work, &mut falls_off),
        }
    }

    for pc in span.start + 1..body_end {
        if !seen[pc - span.start] {
            errors.push(ValidationError::UnreachableInstruction { pc });
        }
    }
    if falls_off {
        // Point at the last body instruction where there is one; an empty
        // function has only its marker to name. An empty function is not
        // exempt: `func_end` is a no-op, so falling off it runs whatever was
        // laid out next rather than returning.
        errors.push(ValidationError::MissingTerminator {
            pc: body_end.saturating_sub(1).max(span.start),
        });
    }
}

// ── Stack-type simulation ──────────────────────────────────────────────────

/// One tracked stack value: its type tag and (when known) concrete bits.
#[derive(Debug, Clone)]
struct SimEntry {
    tag: u8,
    value: Option<u64>,
}

/// Type-level stack simulator: walks the instruction stream tracking value
/// types, reporting underflow and type mismatches, and — when given an
/// [`ArchSpec`] — validating `move` lane groups and `fill`/`local_*` location
/// groups (duplicates only without an arch).
struct StackSimulator<'a> {
    stack: Vec<SimEntry>,
    errors: Vec<ValidationError>,
    arch: Option<&'a ArchSpec>,
    pc: usize,
}

impl<'a> StackSimulator<'a> {
    fn new(arch: Option<&'a ArchSpec>) -> Self {
        Self {
            stack: Vec::new(),
            errors: Vec::new(),
            arch,
            pc: 0,
        }
    }

    /// Record that a pop found the stack empty.
    ///
    /// `StackUnderflow` carries nothing but the `pc`, so a second one for the
    /// same instruction is a literally identical value and says nothing new.
    /// Group pops (`fill 40`, `new_array`) would otherwise emit one per
    /// missing operand — a million lines of `pc 0: stack underflow` for a
    /// single malformed word. All pops for one instruction are consecutive, so
    /// checking the last error is enough.
    fn underflow(&mut self) {
        let already = matches!(
            self.errors.last(),
            Some(ValidationError::StackUnderflow { pc }) if *pc == self.pc
        );
        if !already {
            self.errors
                .push(ValidationError::StackUnderflow { pc: self.pc });
        }
    }

    fn pop_any(&mut self) {
        if self.stack.pop().is_none() {
            self.underflow();
        }
    }

    fn pop_typed(&mut self, expected: u8) {
        match self.stack.pop() {
            Some(entry) if entry.tag != expected => {
                self.errors.push(ValidationError::TypeMismatch {
                    pc: self.pc,
                    expected,
                    got: entry.tag,
                })
            }
            Some(_) => {}
            None => self.underflow(),
        }
    }

    fn pop_typed_n(&mut self, expected: u8, count: u32) {
        for _ in 0..count {
            self.pop_typed(expected);
        }
    }

    /// Pop one value expected to have `expected` tag, returning its concrete
    /// bits when the type matches.
    fn pop_addr(&mut self, expected: u8) -> Option<u64> {
        match self.stack.pop() {
            Some(entry) if entry.tag == expected => entry.value,
            Some(entry) => {
                self.errors.push(ValidationError::TypeMismatch {
                    pc: self.pc,
                    expected,
                    got: entry.tag,
                });
                None
            }
            None => {
                self.underflow();
                None
            }
        }
    }

    fn push(&mut self, tag: u8, value: Option<u64>) {
        self.stack.push(SimEntry { tag, value });
    }

    fn sim_dup(&mut self) {
        if let Some(top) = self.stack.last().cloned() {
            self.stack.push(top);
        } else {
            self.underflow();
        }
    }

    fn sim_swap(&mut self) {
        let len = self.stack.len();
        if len >= 2 {
            self.stack.swap(len - 1, len - 2);
        } else {
            self.underflow();
        }
    }

    /// Report each uniquely-duplicated location once (no-arch fallback).
    fn check_duplicate_locations(&mut self, locations: &[LocationAddr]) {
        let mut seen = HashSet::new();
        let mut reported = HashSet::new();
        for loc in locations {
            let bits = loc.encode();
            if !seen.insert(bits) && reported.insert(bits) {
                self.errors.push(ValidationError::LocationGroupValidation {
                    pc: self.pc,
                    error: LocationGroupError::DuplicateAddress { address: bits },
                });
            }
        }
    }

    /// Report each uniquely-duplicated lane once (no-arch fallback).
    fn check_duplicate_lanes(&mut self, lanes: &[LaneAddr]) {
        let mut seen = HashSet::new();
        let mut reported = HashSet::new();
        for lane in lanes {
            let (d0, d1) = lane.encode();
            let combined = (d0 as u64) | ((d1 as u64) << 32);
            if !seen.insert(combined) && reported.insert(combined) {
                self.errors.push(ValidationError::LaneGroupValidation {
                    pc: self.pc,
                    error: LaneGroupError::DuplicateAddress { address: (d0, d1) },
                });
            }
        }
    }

    /// Pop `arity` locations and validate them as a group.
    fn pop_and_validate_locations(&mut self, arity: u32) {
        let bits: Vec<Option<u64>> = (0..arity).map(|_| self.pop_addr(tag::LOCATION)).collect();
        let locations: Vec<LocationAddr> = bits
            .iter()
            .filter_map(|v| v.map(LocationAddr::decode))
            .collect();
        let pc = self.pc;
        if let Some(arch) = self.arch {
            for error in arch.check_locations(&locations) {
                self.errors
                    .push(ValidationError::LocationGroupValidation { pc, error });
            }
        } else {
            self.check_duplicate_locations(&locations);
        }
    }

    /// Pop `arity` lanes and validate them as a group.
    fn sim_move(&mut self, arity: u32) {
        let bits: Vec<Option<u64>> = (0..arity).map(|_| self.pop_addr(tag::LANE)).collect();
        let lanes: Vec<LaneAddr> = bits
            .iter()
            .filter_map(|v| v.map(LaneAddr::decode_u64))
            .collect();
        let pc = self.pc;
        if let Some(arch) = self.arch {
            for error in arch.check_lanes(&lanes) {
                self.errors
                    .push(ValidationError::LaneGroupValidation { pc, error });
            }
        } else {
            self.check_duplicate_lanes(&lanes);
        }
    }

    fn dispatch(&mut self, inst: &M) {
        match inst {
            // constants push a typed value
            M::Cpu(C::Const(Type::F64, Value::F64(v))) => self.push(tag::FLOAT, Some(v.to_bits())),
            M::Cpu(C::Const(Type::I64, Value::I64(v))) => self.push(tag::INT, Some(*v as u64)),
            M::Lanes(L::ConstLoc(v)) => self.push(tag::LOCATION, Some(*v)),
            M::Lanes(L::ConstLane(v)) => self.push(tag::LANE, Some(*v)),
            M::Lanes(L::ConstZone(v)) => self.push(tag::ZONE, Some(*v as u64)),

            // stack manipulation
            M::Lanes(L::Pop) => self.pop_any(),
            M::Cpu(C::Dup) => self.sim_dup(),
            M::Lanes(L::Swap) => self.sim_swap(),

            // atom arrangement
            M::Lanes(L::InitialFill(arity)) | M::Lanes(L::Fill(arity)) => {
                self.pop_and_validate_locations(*arity)
            }
            M::Lanes(L::Move(arity)) => self.sim_move(*arity),

            // gates
            M::Lanes(L::LocalR(arity)) => {
                self.pop_typed_n(tag::FLOAT, 2);
                self.pop_and_validate_locations(*arity);
            }
            M::Lanes(L::LocalRz(arity)) => {
                self.pop_typed_n(tag::FLOAT, 1);
                self.pop_and_validate_locations(*arity);
            }
            M::Lanes(L::GlobalR) => self.pop_typed_n(tag::FLOAT, 2),
            M::Lanes(L::GlobalRz) => self.pop_typed_n(tag::FLOAT, 1),
            M::Lanes(L::Cz) => self.pop_typed(tag::ZONE),

            // measurement
            M::Lanes(L::Measure(arity)) => {
                self.pop_typed_n(tag::ZONE, *arity);
                for _ in 0..*arity {
                    self.push(tag::MEASURE_FUTURE, None);
                }
            }
            // `await_measure` yields an *array* of measurement results — the
            // same value `set_detector`/`set_observable` consume, and the type
            // `stack_move.AwaitMeasure` is declared to produce. The distinct
            // `MEASUREMENT_RESULT` tag is that array's *element* type, which a
            // one-tag-per-slot simulator cannot express; it exists so
            // `new_array` can name the element type (#547). Giving the whole
            // array that tag broke `measure -> await_measure -> set_detector`.
            M::Lanes(L::AwaitMeasure) => {
                self.pop_typed(tag::MEASURE_FUTURE);
                self.push(tag::ARRAY_REF, None);
            }

            // arrays
            M::Lanes(L::NewArray(_type_tag, dim0, dim1)) => {
                // `validate_structure` rejects counts past the cap, so the
                // clamp here only keeps a rejected program from also driving
                // an unbounded loop before its diagnosis is reported.
                let count = array_element_count(*dim0, *dim1).min(MAX_ARRAY_ELEMENTS);
                for _ in 0..count {
                    self.pop_any();
                }
                self.push(tag::ARRAY_REF, None);
            }
            M::Lanes(L::GetItem(ndims)) => {
                // Clamped for the same reason as `new_array`: an out-of-range
                // index count is already reported by `validate_structure`.
                self.pop_typed_n(tag::INT, (*ndims).min(MAX_GET_ITEM_DIMS));
                self.pop_typed(tag::ARRAY_REF);
                // Element type is not tracked; assume float.
                self.push(tag::FLOAT, None);
            }

            // detectors / observables
            M::Lanes(L::SetDetector) => {
                self.pop_typed(tag::ARRAY_REF);
                self.push(tag::DETECTOR_REF, None);
            }
            M::Lanes(L::SetObservable) => {
                self.pop_typed(tag::ARRAY_REF);
                self.push(tag::OBSERVABLE_REF, None);
            }

            // control
            // `ret <keep>` keeps the top `keep` values as the function's
            // return values; it does not pop exactly one. Modelling it as one
            // made `ret 0` — the spelling every function the compiler emits
            // uses — report a spurious underflow on an empty stack.
            //
            // Bounded by the stack depth plus one: `keep` is an operand, and
            // the extra pop is what records the single underflow when it
            // exceeds what the function has.
            M::Cpu(C::Return(keep)) => {
                for _ in 0..(*keep as usize).min(self.stack.len() + 1) {
                    self.pop_any();
                }
            }
            M::Cpu(C::Halt) => {}

            // Every other vihaco-cpu op — arithmetic, comparisons, control
            // flow, the heap ops — is reachable in a decoded program but is
            // never emitted by the lanes pipeline, so its stack effect is not
            // modelled here. Treating it as a no-op means the simulator does
            // not invent underflows for code it does not understand.
            M::Cpu(_) => {}
        }
    }

    fn run(mut self, program: &Program) -> Vec<ValidationError> {
        // Simulate each function from an empty stack, independently.
        //
        // This used to walk the whole stream and `break` at the first control
        // flow, which lost two things. A `call` is the *only* way to reach a
        // second function, so breaking there meant every multi-function
        // program went unchecked from its first call onwards — a callee's
        // unconditional underflow, behind no branch at all, was never
        // reported. And with no control flow anywhere the markers fell through
        // as no-ops, so one function's leftover operands were still on the
        // stack when the next began, and its errors surfaced an instruction
        // late or not at all.
        //
        // A branch still stops the *containing* function: past it the stack
        // state depends on which edge was taken, and merging those needs the
        // CFG walk tracked in
        // <https://github.com/QuEraComputing/bloqade-lanes/issues/1026>. The
        // linear prefix of every function is checked, which is all of every
        // function the compiler emits today.
        //
        // Starting from empty is the part that is only right for a function
        // nothing passes operands to. `call <arity>` does not clear the stack:
        // it sets `base = stack.len() - arity`, and the callee's locals *alias*
        // the operand stack from there up. So a callee with nonzero arity
        // legitimately consumes values this simulation cannot see, and every
        // such function reported a `StackUnderflow` that execution disproves.
        //
        // Skipping them trades coverage for correctness. Checking them properly
        // means modelling the frame rather than absolute depth — a callee owns
        // exactly `arity` values and popping below `base` corrupts its caller —
        // which is the frame-aware walk in #1026, not something this linear
        // pass can express. This is a placeholder for that, not the end state.
        let takes_operands = callee_arities(&program.code);
        for span in function_spans(&program.code) {
            if takes_operands.contains(&(span.start as u32)) {
                continue;
            }
            self.stack.clear();
            for (pc, inst) in program.code[span.start..span.end]
                .iter()
                .enumerate()
                .map(|(i, inst)| (span.start + i, inst))
            {
                if is_control_flow(inst) {
                    break;
                }
                self.pc = pc;
                self.dispatch(inst);
            }
        }
        self.errors
    }
}

/// Run the type-level stack simulation over a program. Collects underflow and
/// type-mismatch errors, plus lane/location group errors (validated against
/// `arch` when provided, else duplicate-only).
///
/// Conservative around control flow: each function is simulated from an empty
/// stack up to its first branch or call, and a function some `call` passes
/// operands to is skipped entirely. An empty result therefore means "nothing
/// this pass can see is wrong", not "the stack discipline is sound" — see the
/// module docs.
pub fn simulate_stack(program: &Program, arch: Option<&ArchSpec>) -> Vec<ValidationError> {
    StackSimulator::new(arch).run(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::version::Version;

    /// A small, valid arch: one word of 5 sites, one zone.
    const SIMPLE_ARCH_JSON: &str = include_str!("../../../../examples/arch/simple.json");

    fn simple_arch() -> ArchSpec {
        ArchSpec::from_json(SIMPLE_ARCH_JSON).expect("examples/arch/simple.json should parse")
    }

    /// Minimal arch carrying only capability flags; topology is empty (the
    /// capability checks never touch it).
    fn caps_arch(feed_forward: bool, atom_reloading: bool) -> ArchSpec {
        ArchSpec {
            version: Version::new(2, 0),
            words: vec![],
            zones: vec![],
            zone_buses: vec![],
            modes: vec![],
            paths: None,
            feed_forward,
            atom_reloading,
            blockade_radius: None,
        }
    }

    fn program(instructions: Vec<M>) -> Program {
        crate::isa::program::from_code(Version::new(1, 0), instructions).unwrap()
    }

    fn loc(zone_id: u32, word_id: u32, site_id: u32) -> u64 {
        LocationAddr {
            zone_id,
            word_id,
            site_id,
        }
        .encode()
    }

    // ---- capability checks ----

    #[test]
    fn no_arch_skips_all_checks() {
        let p = program(vec![
            M::Lanes(L::Fill(1)),
            M::Lanes(L::Measure(1)),
            M::Lanes(L::Measure(1)),
            M::Lanes(L::ConstZone(99)), // also an invalid address
        ]);
        assert!(validate(&p, None).is_empty());
    }

    #[test]
    fn repeated_measures_allowed_with_feed_forward() {
        let p = program(vec![M::Lanes(L::Measure(1)), M::Lanes(L::Measure(1))]);
        assert!(validate(&p, Some(&caps_arch(true, false))).is_empty());
    }

    #[test]
    fn stack_ops_need_no_capability() {
        let p = program(vec![
            M::Cpu(C::Const(Type::I64, Value::I64(1))),
            M::Cpu(C::Dup),
            M::Cpu(C::Halt),
        ]);
        assert!(validate(&p, Some(&caps_arch(false, false))).is_empty());
    }

    /// Control flow needs `feed_forward`, and the arm that says so exists.
    ///
    /// The variant was declared, formatted, mapped to a Python exception and
    /// unit-tested — but never constructed, because it was written when the
    /// ISA had no control flow. Adopting the calling convention made every
    /// `br`/`cond_br`/`call` reachable from `.sst` with the gate wide open.
    #[test]
    fn control_flow_requires_feed_forward() {
        use vihaco_parser::Ident;

        for (inst, mnemonic) in [
            (M::Cpu(C::Branch(1)), "br"),
            (M::Cpu(C::ConditionalBranch(1, 1)), "cond_br"),
            (M::Cpu(C::Call(0, 0)), "call"),
            (M::Cpu(C::IndirectCall), "call_indirect"),
        ] {
            let p = program(vec![inst, M::Cpu(C::Halt)]);
            assert!(
                validate(&p, Some(&caps_arch(false, false)))
                    .contains(&ValidationError::ControlFlowRequiresFeedForward { pc: 1, mnemonic }),
                "{mnemonic}: got {:?}",
                validate(&p, Some(&caps_arch(false, false)))
            );
            // With the capability, it is allowed.
            assert!(
                !validate(&p, Some(&caps_arch(true, false)))
                    .iter()
                    .any(|e| matches!(e, ValidationError::ControlFlowRequiresFeedForward { .. })),
                "{mnemonic} should pass with feed_forward"
            );
        }

        // A label is not control flow, and neither is `ret`.
        let p = program(vec![
            M::Cpu(C::Label(Ident("x".into()))),
            M::Cpu(C::Return(0)),
        ]);
        assert!(
            !validate(&p, Some(&caps_arch(false, false)))
                .iter()
                .any(|e| matches!(e, ValidationError::ControlFlowRequiresFeedForward { .. }))
        );
    }

    #[test]
    fn single_measure_ok_but_second_rejected_without_feed_forward() {
        let p = program(vec![M::Lanes(L::Measure(1)), M::Lanes(L::Measure(1))]);
        assert_eq!(
            validate(&p, Some(&caps_arch(false, false))),
            vec![ValidationError::MultipleMeasuresRequireFeedForward { pc: 2 }]
        );
    }

    #[test]
    fn fill_requires_atom_reloading() {
        let p = program(vec![M::Lanes(L::Fill(1))]);
        assert_eq!(
            validate(&p, Some(&caps_arch(false, false))),
            vec![ValidationError::FillRequiresAtomReloading { pc: 1 }]
        );
        assert!(validate(&p, Some(&caps_arch(false, true))).is_empty());
    }

    // ---- address checks ----

    #[test]
    fn valid_addresses_pass() {
        let arch = simple_arch();
        // zone 0, word 0, sites 0 and 4 are in range; zone 0 exists.
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::ConstLoc(loc(0, 0, 4))),
            M::Lanes(L::ConstZone(ZoneAddr { zone_id: 0 }.encode())),
        ]);
        assert!(validate(&p, Some(&arch)).is_empty(), "expected no errors");
    }

    #[test]
    fn invalid_location_rejected() {
        let arch = simple_arch();
        // site 99 is out of range for a 5-site word.
        let p = program(vec![M::Lanes(L::ConstLoc(loc(0, 0, 99)))]);
        let errors = validate(&p, Some(&arch));
        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::InvalidLocation { pc: 1, .. }]
            ),
            "got {errors:?}"
        );
    }

    #[test]
    fn invalid_zone_rejected() {
        let arch = simple_arch();
        // zone 5 does not exist (only zone 0).
        let p = program(vec![M::Lanes(L::ConstZone(
            ZoneAddr { zone_id: 5 }.encode(),
        ))]);
        let errors = validate(&p, Some(&arch));
        assert!(
            matches!(
                errors.as_slice(),
                [ValidationError::InvalidZone { pc: 1, .. }]
            ),
            "got {errors:?}"
        );
    }

    #[test]
    fn invalid_lane_rejected() {
        let arch = simple_arch();
        // A lane in a nonexistent zone is invalid.
        let bad = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 9,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let p = program(vec![M::Lanes(L::ConstLane(bad.encode_u64()))]);
        let errors = validate(&p, Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidLane { pc: 1, .. })),
            "got {errors:?}"
        );
    }

    // ---- structural checks ----

    #[test]
    fn well_formed_program_has_no_structural_errors() {
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::InitialFill(1)),
            M::Cpu(C::Return(0)),
        ]);
        assert!(validate_structure(&p).is_empty());
    }

    #[test]
    fn empty_program_is_rejected() {
        assert_eq!(
            validate_structure(&program(vec![])),
            vec![ValidationError::EmptyProgram]
        );
    }

    #[test]
    fn missing_terminator_rejected() {
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::InitialFill(1)),
        ]);
        assert_eq!(
            validate_structure(&p),
            vec![ValidationError::MissingTerminator { pc: 2 }]
        );
    }

    #[test]
    fn halt_is_a_valid_terminator() {
        let p = program(vec![M::Cpu(C::Halt)]);
        assert!(validate_structure(&p).is_empty());
    }

    #[test]
    fn unreachable_after_terminator_rejected() {
        let p = program(vec![M::Cpu(C::Return(0)), M::Cpu(C::Halt)]);
        assert_eq!(
            validate_structure(&p),
            vec![ValidationError::UnreachableInstruction { pc: 2 }]
        );
    }

    #[test]
    fn initial_fill_must_be_first_non_constant() {
        // A const push before initial_fill is fine; a gate before it is not.
        let ok = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::InitialFill(1)),
            M::Cpu(C::Return(0)),
        ]);
        assert!(validate_structure(&ok).is_empty());

        let bad = program(vec![
            M::Lanes(L::GlobalR),
            M::Lanes(L::InitialFill(1)),
            M::Cpu(C::Return(0)),
        ]);
        assert!(validate_structure(&bad).contains(&ValidationError::InitialFillNotFirst { pc: 2 }));
    }

    #[test]
    fn new_array_bounds_checked() {
        let p = program(vec![M::Lanes(L::NewArray(99, 0, 0)), M::Cpu(C::Return(0))]);
        let errors = validate_structure(&p);
        assert!(errors.contains(&ValidationError::NewArrayZeroDim0 { pc: 1 }));
        assert!(errors.contains(&ValidationError::NewArrayInvalidTypeTag {
            pc: 1,
            type_tag: 99
        }));
    }

    #[test]
    fn structure_and_arch_checks_compose() {
        // The composition consumers run: structural + arch-dependent checks
        // both fire and collect. Missing terminator (structural) + bad zone (arch).
        let arch = simple_arch();
        let p = program(vec![M::Lanes(L::ConstZone(
            ZoneAddr { zone_id: 5 }.encode(),
        ))]);
        let errors: Vec<_> = validate_structure(&p)
            .into_iter()
            .chain(validate(&p, Some(&arch)))
            .collect();
        assert!(errors.contains(&ValidationError::MissingTerminator { pc: 1 }));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidZone { .. }))
        );
    }

    #[test]
    fn capability_and_address_errors_collected_together() {
        let arch = simple_arch(); // feed_forward = false, atom_reloading = false
        let p = program(vec![
            M::Lanes(L::Fill(1)),                                     // pc 0: reloading
            M::Lanes(L::ConstZone(ZoneAddr { zone_id: 5 }.encode())), // pc 1: bad zone
        ]);
        let errors = validate(&p, Some(&arch));
        assert_eq!(
            errors.len(),
            2,
            "one error per violation, in pc order: {errors:?}"
        );
        assert!(matches!(
            errors[0],
            ValidationError::FillRequiresAtomReloading { pc: 1 }
        ));
        assert!(matches!(
            errors[1],
            ValidationError::InvalidZone { pc: 2, .. }
        ));
    }

    // ---- stack simulation ----

    fn cpu_float(v: f64) -> M {
        M::Cpu(C::Const(Type::F64, Value::F64(v)))
    }

    #[test]
    fn stack_well_typed_program_has_no_errors() {
        // const_loc, const_loc, initial_fill 2, const_zone, measure 1,
        // await_measure, return — all types line up.
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::ConstLoc(loc(0, 0, 1))),
            M::Lanes(L::InitialFill(2)),
            M::Lanes(L::ConstZone(0)),
            M::Lanes(L::Measure(1)),
            M::Lanes(L::AwaitMeasure),
            M::Cpu(C::Return(0)),
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_underflow_detected() {
        let p = program(vec![M::Lanes(L::Pop)]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
    }

    #[test]
    fn type_mismatch_detected() {
        // initial_fill expects locations; a float is on the stack instead.
        let p = program(vec![cpu_float(1.0), M::Lanes(L::InitialFill(1))]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::TypeMismatch {
                    pc: 2,
                    expected,
                    got
                } if *expected == tag::LOCATION && *got == tag::FLOAT
            )),
            "got {errors:?}"
        );
    }

    #[test]
    fn measure_pushes_future_consumed_by_await() {
        // A measure future left dangling is fine; awaiting a non-future is not.
        let p = program(vec![cpu_float(1.0), M::Lanes(L::AwaitMeasure)]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors.iter().any(|e| matches!(
                e,
                ValidationError::TypeMismatch { pc: 2, expected, .. } if *expected == tag::MEASURE_FUTURE
            )),
            "got {errors:?}"
        );
    }

    #[test]
    fn local_r_pops_two_floats_then_locations() {
        // const_loc, const_float, const_float, local_r 1 — well typed.
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            cpu_float(1.5),
            cpu_float(0.5),
            M::Lanes(L::LocalR(1)),
        ]);
        assert!(simulate_stack(&p, None).is_empty());
    }

    #[test]
    fn duplicate_locations_flagged_without_arch() {
        // Two identical locations into a single fill group.
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::InitialFill(2)),
        ]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn invalid_lane_group_flagged_with_arch() {
        let arch = simple_arch();
        // A lane in a nonexistent zone forms an invalid move group.
        let bad = crate::arch::addr::LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 9,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let p = program(vec![
            M::Lanes(L::ConstLane(bad.encode_u64())),
            M::Lanes(L::Move(1)),
        ]);
        let errors = simulate_stack(&p, Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LaneGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    /// The whole measurement pipeline must type-check, not just its first half.
    ///
    /// The previous version of this test stopped at `await_measure` and only
    /// asserted the tag's numeric value, so it stayed green while
    /// `set_detector` could no longer consume what `await_measure` produced —
    /// the state the canonical `stack_full_pipeline.sst` fixture shipped in.
    #[test]
    fn measure_await_set_detector_type_checks_end_to_end() {
        let p = program(vec![
            M::Lanes(L::ConstZone(0)),
            M::Lanes(L::Measure(1)),
            M::Lanes(L::AwaitMeasure),
            M::Lanes(L::SetDetector),
            M::Lanes(L::Pop),
            M::Cpu(C::Halt),
        ]);
        assert_eq!(simulate_stack(&p, None), vec![]);
    }

    /// `MEASUREMENT_RESULT` is an array *element* type (#547), so it has to be
    /// accepted where `new_array` names one.
    #[test]
    fn measurement_result_is_a_valid_new_array_element_tag() {
        let p = program(vec![
            M::Cpu(C::Const(Type::I64, Value::I64(1))),
            M::Lanes(L::NewArray(tag::MEASUREMENT_RESULT as u32, 1, 0)),
            M::Lanes(L::SetObservable),
            M::Lanes(L::Pop),
            M::Cpu(C::Halt),
        ]);
        assert_eq!(validate_structure(&p), vec![]);
        assert_eq!(simulate_stack(&p, None), vec![]);
    }

    #[test]
    fn await_measure_pushes_an_array_ref() {
        // Pin the tag the awaited value carries, by observing what rejects it:
        // `cz` wants a zone, and says what it got instead.
        let p = program(vec![
            M::Lanes(L::ConstZone(0)),
            M::Lanes(L::Measure(1)),
            M::Lanes(L::AwaitMeasure),
            M::Lanes(L::Cz),
        ]);
        assert!(
            simulate_stack(&p, None).contains(&ValidationError::TypeMismatch {
                pc: 4,
                expected: tag::ZONE,
                got: tag::ARRAY_REF,
            }),
            "got {:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn new_array_element_count_is_bounded() {
        // dim0 * dim1 = 2^32, which wraps to 0 in `u32` — the operands are
        // read straight out of the instruction word, so nothing stops a
        // program from carrying them.
        let p = program(vec![
            M::Lanes(L::NewArray(0, 65536, 65536)),
            M::Cpu(C::Halt),
        ]);
        assert!(
            validate_structure(&p).contains(&ValidationError::NewArrayTooManyElements {
                pc: 1,
                count: 1 << 32,
            }),
            "got {:?}",
            validate_structure(&p)
        );

        // And a count that does not wrap but would still drive a four-billion
        // iteration pop loop.
        let p = program(vec![M::Lanes(L::NewArray(0, u32::MAX, 0)), M::Cpu(C::Halt)]);
        assert!(
            validate_structure(&p).contains(&ValidationError::NewArrayTooManyElements {
                pc: 1,
                count: u32::MAX as u64,
            }),
            "got {:?}",
            validate_structure(&p)
        );
    }

    /// Simulating a rejected program terminates promptly *and* says something
    /// useful: the element count is clamped, and the group pop reports one
    /// underflow rather than one per missing operand.
    #[test]
    fn simulating_an_oversized_new_array_reports_one_underflow() {
        let p = program(vec![M::Lanes(L::NewArray(0, u32::MAX, 0)), M::Cpu(C::Halt)]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
    }

    /// A group pop off an empty stack is one diagnosis, not `arity` of them.
    #[test]
    fn a_group_pop_reports_a_single_underflow() {
        let p = program(vec![M::Lanes(L::InitialFill(40)), M::Cpu(C::Halt)]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );

        // Distinct instructions still report separately.
        let p = program(vec![M::Lanes(L::Pop), M::Lanes(L::Pop), M::Cpu(C::Halt)]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![
                ValidationError::StackUnderflow { pc: 1 },
                ValidationError::StackUnderflow { pc: 2 },
            ]
        );
    }

    /// Branch and call targets are addresses, and nothing checked them.
    ///
    /// An unnameable target is not just invalid, it is unrenderable: `to_text`
    /// has to invent `@L9` / `@F1`, and the text it emits does not re-read.
    /// Catching it here means the renderer is only ever asked to name targets
    /// that have a name.
    /// Reachability follows branches instead of latching on a terminator.
    ///
    /// The latch condemned `cond_br`'s second arm — the `halt` ending the
    /// first arm set it — and could not see code a `br` jumped over.
    /// Each function is simulated from an empty stack, independently.
    ///
    /// Breaking at the first control flow lost every multi-function program's
    /// tail — `call` is the only way to reach a second function — and with no
    /// control flow at all, one function's leftover operands were still on the
    /// stack when the next began.
    #[test]
    fn the_stack_is_simulated_per_function() {
        use crate::isa::text::parse_text;

        // (a) A callee's unconditional underflow, behind a `call`.
        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n                   .text(root):\nfn @main() {\n  lanes::lanes.const_zone 0x00000000\n                     cpu::cpu.call 0, helper\n  cpu::cpu.halt\n}\n                   fn @helper() {\n  lanes::lanes.pop\n  cpu::cpu.ret 0\n}\n                   .text(root).\n.section(root).\n";
        let p = parse_text(src).unwrap();
        assert!(
            simulate_stack(&p, None)
                .iter()
                .any(|e| matches!(e, ValidationError::StackUnderflow { .. })),
            "@helper's underflow should be reported: {:?}",
            simulate_stack(&p, None)
        );

        // (b) One function must not consume what the previous left behind.
        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n                   .text(root):\nfn @main() {\n  lanes::lanes.const_zone 0x00000000\n                     cpu::cpu.halt\n}\n                   fn @helper() {\n  lanes::lanes.cz\n  cpu::cpu.ret 0\n}\n                   .text(root).\n.section(root).\n";
        let p = parse_text(src).unwrap();
        assert!(
            simulate_stack(&p, None)
                .iter()
                .any(|e| matches!(e, ValidationError::StackUnderflow { .. })),
            "@helper starts from an empty stack: {:?}",
            simulate_stack(&p, None)
        );
    }

    /// `ret <keep>` keeps the top `keep` values; it does not pop exactly one.
    ///
    /// Modelling it as one made `ret 0` — what every function the compiler
    /// emits uses — report a spurious underflow on an empty stack.
    #[test]
    fn ret_pops_its_keep_count() {
        let p = program(vec![M::Cpu(C::Return(0))]);
        assert_eq!(simulate_stack(&p, None), vec![]);

        let p = program(vec![M::Cpu(C::Return(1))]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );

        // An implausible keep count is one diagnosis, not four billion.
        let p = program(vec![M::Cpu(C::Return(u32::MAX))]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
    }

    /// Wrap a multi-function body in the `sst v1` container.
    fn sst_module(body: &str) -> Program {
        use crate::isa::text::parse_text;
        parse_text(&format!(
            "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n\
             .text(root):\n{body}.text(root).\n.section(root).\n"
        ))
        .expect("the module should parse")
    }

    /// A callee that receives operands is not simulated, because this pass
    /// starts every function from an empty stack.
    ///
    /// `call <arity>` does not clear the stack — it sets
    /// `base = stack.len() - arity` and the callee's locals alias the operand
    /// stack from there up. So the argument `@helper` consumes below is one
    /// `@main` legitimately passed, and reporting an underflow for it
    /// contradicts the machine, which runs this program to `Halted`. See
    /// <https://github.com/QuEraComputing/bloqade-lanes/issues/1026>.
    #[test]
    fn a_callee_taking_operands_is_not_simulated() {
        let p = sst_module(
            "fn @main() {\n  lanes::lanes.const_zone 0x00000000\n  \
             cpu::cpu.call 1, helper\n  lanes::lanes.pop\n  cpu::cpu.halt\n}\n\n\
             fn @helper() {\n  cpu::cpu.load u32, 0\n  lanes::lanes.measure 1\n  \
             lanes::lanes.await_measure\n  cpu::cpu.ret 1\n}\n",
        );
        assert_eq!(
            simulate_stack(&p, None),
            vec![],
            "a nonzero-arity callee must not report an underflow this pass cannot justify"
        );

        // The machine is the authority the assertion above defers to.
        let mut machine = crate::isa::machine::LanesMachine::new();
        let run = machine.run(&p, 10_000).expect("the program should run");
        assert!(matches!(
            run.stopped,
            crate::isa::machine::Stopped::Halted | crate::isa::machine::Stopped::Returned
        ));
    }

    /// Only callees that receive operands are skipped. A zero-arity callee is
    /// still simulated, so the concession above cannot quietly widen into
    /// "stop checking anything reachable by a call".
    #[test]
    fn a_zero_arity_callee_is_still_simulated() {
        let p = sst_module(
            "fn @main() {\n  cpu::cpu.call 0, helper\n  cpu::cpu.halt\n}\n\n\
             fn @helper() {\n  lanes::lanes.measure 1\n  cpu::cpu.ret 0\n}\n",
        );
        assert!(
            simulate_stack(&p, None).contains(&ValidationError::StackUnderflow { pc: 5 }),
            "a zero-arity callee's underflow is real and must still be reported: {:?}",
            simulate_stack(&p, None)
        );
    }

    /// One nonzero-arity call site is enough to skip the callee, even when
    /// another site passes nothing — the arity lives on the call, not the
    /// function, so the two disagree and the empty-stack model fits neither.
    #[test]
    fn one_operand_passing_call_site_is_enough_to_skip() {
        let p = sst_module(
            "fn @main() {\n  cpu::cpu.call 0, helper\n  \
             lanes::lanes.const_zone 0x00000000\n  cpu::cpu.call 1, helper\n  \
             cpu::cpu.halt\n}\n\n\
             fn @helper() {\n  lanes::lanes.measure 1\n  cpu::cpu.ret 0\n}\n",
        );
        assert_eq!(
            simulate_stack(&p, None),
            vec![],
            "a callee reached at arity 0 and arity 1 is skipped"
        );
    }

    #[test]
    fn reachability_follows_branch_edges() {
        // Both arms of a `cond_br` are reachable.
        let p = program(vec![
            M::Cpu(C::ConditionalBranch(2, 4)),
            M::Lanes(L::Cz),
            M::Cpu(C::Halt),
            M::Lanes(L::Cz),
            M::Cpu(C::Halt),
        ]);
        assert_eq!(
            validate_structure(&p)
                .iter()
                .filter(|e| matches!(e, ValidationError::UnreachableInstruction { .. }))
                .count(),
            0,
            "got {:?}",
            validate_structure(&p)
        );

        // And code a branch jumps over is still reported.
        let p = program(vec![M::Cpu(C::Branch(3)), M::Lanes(L::Cz), M::Cpu(C::Halt)]);
        assert!(
            validate_structure(&p).contains(&ValidationError::UnreachableInstruction { pc: 2 }),
            "the jumped-over cz should be reported: {:?}",
            validate_structure(&p)
        );
    }

    /// Every function must terminate, and one function's dead code must not
    /// hide another's missing terminator.
    #[test]
    fn terminators_are_checked_per_function() {
        use crate::isa::text::parse_text;

        // @main has unreachable code; @helper has no terminator. The old
        // either/or gate reported only the first.
        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n                   .text(root):\nfn @main() {\n  cpu::cpu.halt\n  lanes::lanes.cz\n}\n                   fn @helper() {\n  lanes::lanes.cz\n}\n.text(root).\n.section(root).\n";
        let errors = validate_structure(&parse_text(src).unwrap());
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::UnreachableInstruction { .. })),
            "got {errors:?}"
        );
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::MissingTerminator { .. })),
            "got {errors:?}"
        );

        // An empty function is not exempt: `func_end` is a no-op, so falling
        // off it runs whatever was laid out next rather than returning.
        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n                   .text(root):\nfn @main() {\n  cpu::cpu.halt\n}\n                   fn @helper() {\n}\n.text(root).\n.section(root).\n";
        assert!(
            validate_structure(&parse_text(src).unwrap())
                .iter()
                .any(|e| matches!(e, ValidationError::MissingTerminator { .. })),
            "an empty function still falls through"
        );
    }

    /// A branch is intra-procedural; leaving the function is what `call` is
    /// for. A trailing label resolves to the function's own `func_end`, and a
    /// target past it enters the next function with no frame.
    #[test]
    fn a_branch_cannot_leave_its_function() {
        use crate::isa::text::parse_text;

        let src = "sst v1\n\n.section(root):\n.header(root):\nversion 1.0\n.header(root).\n                   .text(root):\nfn @main() {\n  cpu::cpu.br @far\n  cpu::cpu.halt\n}\n                   fn @helper() {\n  cpu::cpu.label @far\n  cpu::cpu.halt\n}\n                   .text(root).\n.section(root).\n";
        assert!(
            validate_structure(&parse_text(src).unwrap())
                .iter()
                .any(|e| matches!(
                    e,
                    ValidationError::InvalidControlFlowTarget {
                        expected: "an address in the same function",
                        ..
                    }
                )),
            "a cross-function branch should be rejected"
        );
    }

    #[test]
    fn control_flow_targets_are_checked() {
        // A branch past the end of the code.
        let p = program(vec![M::Cpu(C::Branch(99)), M::Cpu(C::Halt)]);
        assert!(
            validate_structure(&p).contains(&ValidationError::InvalidControlFlowTarget {
                pc: 1,
                target: 99,
                expected: "an address in the same function",
            }),
            "got {:?}",
            validate_structure(&p)
        );

        // Both arms of a `cond_br`.
        let p = program(vec![M::Cpu(C::ConditionalBranch(99, 98)), M::Cpu(C::Halt)]);
        assert_eq!(
            validate_structure(&p)
                .iter()
                .filter(|e| matches!(e, ValidationError::InvalidControlFlowTarget { .. }))
                .count(),
            2,
            "both arms should be reported"
        );

        // A call to an address that begins no function. Address 1 is the
        // `halt` inside `@main`, not a `func_start`.
        let p = program(vec![M::Cpu(C::Call(0, 2)), M::Cpu(C::Halt)]);
        assert!(
            validate_structure(&p).contains(&ValidationError::InvalidControlFlowTarget {
                pc: 1,
                target: 2,
                expected: "a function entry",
            }),
            "got {:?}",
            validate_structure(&p)
        );

        // A branch inside the same function is fine.
        let p = program(vec![M::Cpu(C::Branch(2)), M::Cpu(C::Halt)]);
        assert!(
            !validate_structure(&p)
                .iter()
                .any(|e| matches!(e, ValidationError::InvalidControlFlowTarget { .. })),
            "got {:?}",
            validate_structure(&p)
        );
    }

    #[test]
    fn get_item_index_count_is_bounded() {
        // Arrays are at most 2-D, so three indices is structurally wrong —
        // and `u32::MAX` indices would overflow the `n + 1` the machine pops.
        for ndims in [0, 3, u32::MAX] {
            let p = program(vec![M::Lanes(L::GetItem(ndims)), M::Cpu(C::Halt)]);
            assert!(
                validate_structure(&p)
                    .contains(&ValidationError::GetItemInvalidDims { pc: 1, ndims }),
                "ndims={ndims}: got {:?}",
                validate_structure(&p)
            );
        }
        // One or two indices are the well-formed shapes.
        for ndims in [1, 2] {
            let p = program(vec![M::Lanes(L::GetItem(ndims)), M::Cpu(C::Halt)]);
            assert!(
                !validate_structure(&p)
                    .iter()
                    .any(|e| matches!(e, ValidationError::GetItemInvalidDims { .. })),
                "ndims={ndims} should be accepted"
            );
        }
    }

    /// A `store` index is a memory request, so it has to be bounded where it is
    /// read rather than where it is spent.
    ///
    /// `op_store` resizes the operand stack to reach its index and writes every
    /// new slot, so `store u64, 4294967295` touches ~68 GB from a three-
    /// instruction program. The assertion is on the *bound* — that the
    /// validator names the operand — and not on the allocator refusing, because
    /// a `resize` this size succeeds on a lazily-committing platform and the
    /// two earlier bounds of this class only failed on Linux CI.
    #[test]
    fn local_index_is_bounded() {
        let over = MAX_LOCAL_INDEX + 1;
        for (inst, mnemonic, index) in [
            (C::Store(Type::U64, u32::MAX), "store", u32::MAX),
            (C::Load(Type::U64, u32::MAX), "load", u32::MAX),
            // The first index past the bound, so the boundary is pinned from
            // both sides rather than only at an absurd operand.
            (C::Store(Type::U64, over), "store", over),
            (C::Load(Type::U64, over), "load", over),
        ] {
            let p = program(vec![M::Cpu(inst.clone()), M::Cpu(C::Halt)]);
            assert!(
                validate_structure(&p).contains(&ValidationError::LocalIndexOutOfRange {
                    // Address 0 is `@main`'s `func_start`.
                    pc: 1,
                    mnemonic,
                    index,
                }),
                "{inst:?}: got {:?}",
                validate_structure(&p)
            );
        }

        // The ceiling itself, and the indices a real frame uses, are accepted.
        for index in [0, 1, 7, MAX_LOCAL_INDEX] {
            let p = program(vec![
                M::Cpu(C::Store(Type::U64, index)),
                M::Cpu(C::Load(Type::U64, index)),
                M::Cpu(C::Halt),
            ]);
            assert!(
                !validate_structure(&p)
                    .iter()
                    .any(|e| matches!(e, ValidationError::LocalIndexOutOfRange { .. })),
                "index={index} should be accepted, got {:?}",
                validate_structure(&p)
            );
        }
    }

    /// `load`/`store` are ordinary instructions, so one still closes the window
    /// in which `initial_fill` may appear. Adding their own match arms could
    /// have dropped them out of the `_ => seen_non_constant = true` fallback.
    #[test]
    fn a_local_access_is_a_non_constant_instruction() {
        let p = program(vec![
            M::Cpu(C::Store(Type::U64, 0)),
            M::Lanes(L::InitialFill(0)),
            M::Cpu(C::Halt),
        ]);
        assert!(
            validate_structure(&p).contains(&ValidationError::InitialFillNotFirst { pc: 2 }),
            "got {:?}",
            validate_structure(&p)
        );
    }

    // ---- Display ----

    #[test]
    fn validation_error_display_strings() {
        use crate::arch::query::{LaneGroupError, LocationGroupError};

        let cases: Vec<(ValidationError, String)> = vec![
            (
                ValidationError::ControlFlowRequiresFeedForward {
                    pc: 3,
                    mnemonic: "cond_br",
                },
                "pc 3: control-flow instruction 'cond_br' requires feed_forward capability".into(),
            ),
            (
                ValidationError::MultipleMeasuresRequireFeedForward { pc: 5 },
                "pc 5: multiple measure instructions require feed_forward capability".into(),
            ),
            (
                ValidationError::FillRequiresAtomReloading { pc: 1 },
                "pc 1: fill instruction requires atom_reloading capability".into(),
            ),
            (
                ValidationError::InvalidLocation {
                    pc: 2,
                    message: "bad".into(),
                },
                "pc 2: invalid location: bad".into(),
            ),
            (
                ValidationError::InvalidLane {
                    pc: 2,
                    message: "bad".into(),
                },
                "pc 2: invalid lane: bad".into(),
            ),
            (
                ValidationError::InvalidZone {
                    pc: 2,
                    message: "bad".into(),
                },
                "pc 2: invalid zone: bad".into(),
            ),
            (
                ValidationError::NewArrayZeroDim0 { pc: 0 },
                "pc 0: new_array dim0 must be > 0".into(),
            ),
            (
                ValidationError::NewArrayInvalidTypeTag {
                    pc: 0,
                    type_tag: 99,
                },
                "pc 0: invalid new_array type tag 99".into(),
            ),
            (
                ValidationError::LocalIndexOutOfRange {
                    pc: 1,
                    mnemonic: "store",
                    index: 200_000_000,
                },
                "pc 1: store takes a local index 0..=1023, got 200000000".into(),
            ),
            (
                ValidationError::InitialFillNotFirst { pc: 4 },
                "pc 4: initial_fill must be the first non-constant instruction".into(),
            ),
            (
                ValidationError::EmptyProgram,
                "program has no instructions: missing return or halt terminator".into(),
            ),
            (
                ValidationError::MissingTerminator { pc: 7 },
                "pc 7: function must end with return or halt".into(),
            ),
            (
                ValidationError::UnreachableInstruction { pc: 8 },
                "pc 8: unreachable instruction after return or halt".into(),
            ),
            (
                ValidationError::StackUnderflow { pc: 1 },
                "pc 1: stack underflow".into(),
            ),
            (
                ValidationError::TypeMismatch {
                    pc: 1,
                    expected: tag::LOCATION,
                    got: tag::FLOAT,
                },
                "pc 1: type mismatch: expected tag 0x3, got 0x0".into(),
            ),
        ];
        for (err, expected) in cases {
            assert_eq!(err.to_string(), expected);
        }

        // The two group-validation variants prefix `pc N:` onto the wrapped
        // arch-layer error's own Display.
        let loc_err = LocationGroupError::DuplicateAddress { address: 0x10 };
        assert_eq!(
            ValidationError::LocationGroupValidation {
                pc: 2,
                error: loc_err.clone(),
            }
            .to_string(),
            format!("pc 2: {loc_err}")
        );
        let lane_err = LaneGroupError::DuplicateAddress { address: (1, 2) };
        assert_eq!(
            ValidationError::LaneGroupValidation {
                pc: 3,
                error: lane_err.clone(),
            }
            .to_string(),
            format!("pc 3: {lane_err}")
        );
    }

    // ---- stack simulation: dispatch coverage ----

    #[test]
    fn stack_sim_int_const_and_pop() {
        // `cpu::cpu.const i64` pushes an INT; `pop` discards it. Well typed.
        let p = program(vec![
            M::Cpu(C::Const(Type::I64, Value::I64(7))),
            M::Lanes(L::Pop),
        ]);
        assert!(simulate_stack(&p, None).is_empty());
    }

    #[test]
    fn stack_sim_swap_needs_two_entries() {
        let ok = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::ConstLoc(loc(0, 0, 1))),
            M::Lanes(L::Swap),
        ]);
        assert!(simulate_stack(&ok, None).is_empty());

        // Only one entry: swap underflows.
        let bad = program(vec![M::Lanes(L::ConstLoc(loc(0, 0, 0))), M::Lanes(L::Swap)]);
        assert_eq!(
            simulate_stack(&bad, None),
            vec![ValidationError::StackUnderflow { pc: 2 }]
        );
    }

    #[test]
    fn stack_sim_dup_underflow_on_empty() {
        let p = program(vec![M::Cpu(C::Dup)]);
        assert_eq!(
            simulate_stack(&p, None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
    }

    #[test]
    fn stack_sim_dup_copies_top_of_stack() {
        // `dup` on a non-empty stack pushes a copy of the top entry; popping
        // both leaves an empty, well-typed stack.
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Cpu(C::Dup),
            M::Lanes(L::Pop),
            M::Lanes(L::Pop),
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_sim_gate_ops_are_well_typed() {
        // Exercises the LocalRz / GlobalR / GlobalRz / Cz dispatch arms.
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            cpu_float(0.5),
            M::Lanes(L::LocalRz(1)),
            cpu_float(0.5),
            M::Lanes(L::GlobalRz),
            cpu_float(0.5),
            cpu_float(0.25),
            M::Lanes(L::GlobalR),
            M::Lanes(L::ConstZone(0)),
            M::Lanes(L::Cz),
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_sim_gate_underflow_on_empty_stack() {
        // GlobalRz pops one float via `pop_typed`; empty stack -> underflow.
        assert_eq!(
            simulate_stack(&program(vec![M::Lanes(L::GlobalRz)]), None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
        // Move pops a lane via `pop_addr`; empty stack -> underflow.
        assert_eq!(
            simulate_stack(&program(vec![M::Lanes(L::Move(1))]), None),
            vec![ValidationError::StackUnderflow { pc: 1 }]
        );
    }

    #[test]
    fn stack_sim_new_array_and_get_item() {
        // new_array pops `dim0` elements and pushes an ARRAY_REF; get_item pops
        // `ndims` INT indices plus the ARRAY_REF and pushes the element.
        let p = program(vec![
            M::Cpu(C::Const(Type::I64, Value::I64(1))),
            M::Cpu(C::Const(Type::I64, Value::I64(2))),
            M::Lanes(L::NewArray(tag::INT as u32, 2, 0)),
            M::Cpu(C::Const(Type::I64, Value::I64(0))),
            M::Lanes(L::GetItem(1)),
        ]);
        assert!(
            simulate_stack(&p, None).is_empty(),
            "{:?}",
            simulate_stack(&p, None)
        );
    }

    #[test]
    fn stack_sim_set_detector_and_observable() {
        // Each consumes an ARRAY_REF (produced by a 1-element new_array).
        let det = program(vec![
            M::Cpu(C::Const(Type::I64, Value::I64(0))),
            M::Lanes(L::NewArray(tag::INT as u32, 1, 0)),
            M::Lanes(L::SetDetector),
        ]);
        assert!(
            simulate_stack(&det, None).is_empty(),
            "{:?}",
            simulate_stack(&det, None)
        );

        let obs = program(vec![
            M::Cpu(C::Const(Type::I64, Value::I64(0))),
            M::Lanes(L::NewArray(tag::INT as u32, 1, 0)),
            M::Lanes(L::SetObservable),
        ]);
        assert!(
            simulate_stack(&obs, None).is_empty(),
            "{:?}",
            simulate_stack(&obs, None)
        );
    }

    #[test]
    fn stack_sim_stops_at_control_flow_rather_than_guessing() {
        // A `pop` on an empty stack is an underflow the simulator would
        // normally catch — but behind a branch it cannot know the stack state,
        // so it stops instead of reporting something it cannot justify.
        let behind_branch = program(vec![
            M::Cpu(C::Branch(2)),
            M::Lanes(L::Pop),
            M::Cpu(C::Halt),
        ]);
        assert!(
            simulate_stack(&behind_branch, None).is_empty(),
            "must not report errors derived from an unknown post-branch state"
        );

        // The same underflow ahead of the branch is still caught.
        let before_branch = program(vec![
            M::Lanes(L::Pop),
            M::Cpu(C::Branch(2)),
            M::Cpu(C::Halt),
        ]);
        assert!(
            simulate_stack(&before_branch, None)
                .iter()
                .any(|e| matches!(e, ValidationError::StackUnderflow { .. })),
            "linear prefix is still checked"
        );
    }

    #[test]
    fn stack_sim_halt_is_a_noop() {
        // `halt` neither pushes nor pops, so it cannot underflow an empty stack.
        let p = program(vec![M::Cpu(C::Halt)]);
        assert!(simulate_stack(&p, None).is_empty());
    }

    #[test]
    fn location_group_validated_against_arch() {
        // With an arch, `initial_fill` runs the arch location-group check
        // (not just the no-arch duplicate fallback). Two valid distinct
        // locations produce no group error.
        let arch = simple_arch();
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::ConstLoc(loc(0, 0, 1))),
            M::Lanes(L::InitialFill(2)),
        ]);
        let errors = simulate_stack(&p, Some(&arch));
        assert!(
            !errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn location_group_arch_errors_are_reported() {
        // With an arch, a bad fill group surfaces the arch layer's own
        // location-group error (exercising the arch branch's error path, not
        // the no-arch duplicate fallback). A location repeated within the
        // group is invalid.
        let arch = simple_arch();
        let p = program(vec![
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::ConstLoc(loc(0, 0, 0))),
            M::Lanes(L::InitialFill(2)),
        ]);
        let errors = simulate_stack(&p, Some(&arch));
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LocationGroupValidation { .. })),
            "got {errors:?}"
        );
    }

    #[test]
    fn duplicate_lanes_flagged_without_arch() {
        // The no-arch `move` fallback reports repeated lanes as duplicates.
        let lane = LaneAddr {
            direction: crate::arch::addr::Direction::Forward,
            move_type: crate::arch::addr::MoveType::SiteBus,
            zone_id: 0,
            word_id: 0,
            site_id: 0,
            bus_id: 0,
        };
        let p = program(vec![
            M::Lanes(L::ConstLane(lane.encode_u64())),
            M::Lanes(L::ConstLane(lane.encode_u64())),
            M::Lanes(L::Move(2)),
        ]);
        let errors = simulate_stack(&p, None);
        assert!(
            errors
                .iter()
                .any(|e| matches!(e, ValidationError::LaneGroupValidation { .. })),
            "got {errors:?}"
        );
    }
}
