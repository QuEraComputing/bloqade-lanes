# Logical STAR-X and Rz-producing gate validation

## Goal

Reject a Gemini logical kernel at definition time when it combines `StarRx`
with a SQUIN gate that can produce a logical native `Rz`. This is an intentionally
conservative compatibility check. It replaces the proposed compilation-time,
same-qubit, before-`StarRx` check; no change to `EliminateRz` is required.

## Rule

When `@logical.kernel` runs with its normal `verify=True`, reject a program if
both of these occur anywhere in the reachable logical program:

1. A `gemini.logical.extensions.StarRx` statement.
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

Implement a dedicated `ValidationPass` in the Gemini logical validation
package. Add it to the suite in `python/bloqade/gemini/logical/group.py`, after
the existing inlining/unrolling step and alongside the other logical checks.
This makes failure visible when defining `@logical.kernel`, rather than when
creating a simulator task or lowering to place/move. `verify=False` retains its
existing meaning and skips the validation suite.

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
