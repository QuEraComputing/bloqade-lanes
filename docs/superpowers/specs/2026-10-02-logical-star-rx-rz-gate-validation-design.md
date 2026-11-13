# Logical STAR-X and Rz-producing gate validation

## Goal

Reject a Gemini logical kernel at definition time when it combines `StarRx`
with a SQUIN gate that can produce a logical native `Rz`. This is an intentionally
conservative compatibility check. The compiler pass lives in lanes; definition-time integration into the Gemini
frontend is a follow-up in the bloqade repository. Independently, `EliminateRz`
flushes any pending Z frame before `StarRx` so low-level rewriting preserves
the operation order.

## Rule

The validation pass rejects a program if
both of these occur anywhere in the reachable logical program:

1. A `bloqade.lanes.dialects.steane713.extensions.stmts.StarRx` statement.
2. One of `squin.rz`, `squin.z`, `squin.s` (including its adjoint), or `squin.h`.

The validator does not compare qubits or gate order. It also rejects a first-gate
`squin.rz` that later lowering would absorb into initialization; that false
positive is accepted for this initial conservative rule. `H` is included because
the current `DecomposeCliffordToNative` expands it into `S`, `SqrtX`, `S`, and
`S` lowers to native `Rz`. The other listed gates directly lower to native
`Rz`. `StarRz` is exempt because its physical STAR gadget does not create a
pending logical `Rz` frame. Other SQUIN gates remain allowed unless the
documented lowering begins generating logical native `Rz` for them.

## Placement and behavior

The dedicated `StarRxRzGateValidation` pass lives in
`bloqade.lanes.validation.star_rx_rz`. Its tests invoke it directly on kernels
built from lanes dialects, without importing Gemini.

A follow-up PR in bloqade must add the pass to the suite in
`src/bloqade/gemini/logical/group.py`, after inlining/unrolling and alongside the
other logical checks. That will make failure visible when defining
`@logical.kernel`; `verify=False` will continue to skip the validation suite.
This PR does not change the validation defaults of `LogicalPipeline`.

The pass inspects the reachable logical program, including statically resolved
helper calls when they have not been inlined. It reports an actionable error
anchored to `StarRx`, naming the conflicting SQUIN gate classes and explaining
that the restriction is conservative. Dynamic calls continue to be governed
by existing logical validation; this pass does not add a new call-resolution
system.

## Tests

Cover `StarRx` paired separately with `Rz`, `Z`, `S`, its adjoint, and `H`.
Demonstrate the deliberately strict behavior for a gate after `StarRx`, a gate
on another qubit, and a first-gate `Rz`. Cover an allowed `StarRx` program with
`CZ` or `CX`, and `StarRx` together with `StarRz`. Verify that a program without
`StarRx` keeps its existing behavior. Check both the normal default-unrolling
path and a non-inlined static helper where practical.
