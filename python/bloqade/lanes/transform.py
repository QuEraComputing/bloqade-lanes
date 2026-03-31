from dataclasses import dataclass
from typing import Literal

from bloqade.rewrite.passes import aggressive_unroll as agg
from bloqade.squin.rewrite import SquinU3ToClifford
from kirin import ir, rewrite
from kirin.dialects import ilist, scf
from kirin.passes import TypeInfer

from bloqade import qubit, squin
from bloqade.lanes import layout
from bloqade.lanes.analysis import atom
from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite import move2squin
from bloqade.lanes.rewrite.move2squin import (
    LogicalNoiseModelABC as LogicalNoiseModelABC,
    NoiseModelABC as NoiseModelABC,
    SimpleLogicalNoiseModel as SimpleLogicalNoiseModel,
    SimpleNoiseModel as SimpleNoiseModel,
)


def _emit_common(
    main: ir.Method,
    arch_spec: layout.ArchSpec,
    initialize_kernel: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None]
        | None
    ),
    noise_model: NoiseModelABC | None,
    aggressive_unroll: bool,
    no_raise: bool,
    initialize_noise_kernel: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None]
        | None
    ) = None,
) -> ir.Method:
    """Shared implementation for all MoveToSquin variants."""
    main = main.similar(main.dialects.union(squin.kernel.discard(scf.lowering)))

    vqpu = atom.AtomInterpreter(main.dialects, arch_spec=arch_spec)
    run_method = vqpu.run_no_raise if no_raise else vqpu.run

    frame, _ = run_method(main)
    qubit_rule = move2squin.InsertQubits(frame)
    rewrite.Walk(qubit_rule).rewrite(main.code)
    rules = [
        move2squin.InsertGates(
            arch_spec,
            qubit_rule.physical_ssa_values,
            frame,
            initialize_kernel,
        ),
        move2squin.InsertMeasurements(qubit_rule.physical_ssa_values, frame),
    ]
    if noise_model is not None:
        rules.append(
            move2squin.InsertNoise(
                arch_spec,
                qubit_rule.physical_ssa_values,
                frame,
                noise_model,
                initialize_noise_kernel=initialize_noise_kernel,
            ),
        )

    rewrite.Walk(rewrite.Chain(*rules)).rewrite(main.code)

    # we need to fold before writing U3 to Clifford
    if aggressive_unroll:
        agg.AggressiveUnroll(main.dialects).fixpoint(main)
    else:
        agg.Fold(main.dialects)(main)

    rewrite.Walk(
        SquinU3ToClifford(),
    ).rewrite(main.code)

    rewrite.Fixpoint(
        rewrite.Walk(
            rewrite.Chain(
                move2squin.CleanUpMoveDialect(),
                rewrite.DeadCodeElimination(),
                rewrite.CommonSubexpressionElimination(),
            )
        )
    ).rewrite(main.code)

    out = main.similar(main.dialects.discard(move.dialect))

    TypeInfer(out.dialects)(out)
    out.verify()
    out.verify_type()

    return out


@dataclass
class MoveToSquinLogical:
    """Rewrite pass for **logical** compilation.

    Handles ``PhysicalInitialize`` rewrites using clean/noisy initialization
    kernels from the noise model and optionally inserts gate/move noise.

    Parameters
    ----------
    arch_spec:
        The architecture specification.
    noise_model:
        A logical noise model that provides initialization kernels.
    add_noise:
        When ``True``, insert noise gates **and** use the noisy init kernel
        (if available).  When ``False``, use the clean init kernel and skip
        noise insertion.
    aggressive_unroll:
        Use aggressive unrolling instead of simple folding.
    """

    arch_spec: layout.ArchSpec
    noise_model: LogicalNoiseModelABC
    add_noise: bool = False
    aggressive_unroll: bool = False

    def _resolve_initialize_kernel(self):
        """Resolve the clean initialization kernel for InsertGates."""
        clean, _ = self.noise_model.get_logical_initialize()
        return clean

    def _resolve_initialize_noise_kernel(self):
        """Resolve the noisy initialization kernel for InsertNoise."""
        if not self.add_noise:
            return None
        _, noisy = self.noise_model.get_logical_initialize()
        return noisy

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        init_kernel = self._resolve_initialize_kernel()
        noise = self.noise_model if self.add_noise else None
        return _emit_common(
            main,
            self.arch_spec,
            init_kernel,
            noise,
            self.aggressive_unroll,
            no_raise,
            initialize_noise_kernel=self._resolve_initialize_noise_kernel(),
        )


@dataclass
class MoveToSquinPhysical:
    """Rewrite pass for **physical** compilation.

    No initialization kernel handling — passes ``None`` to ``InsertGates`` for
    the init kernel.  Noise is inserted only when a noise model is provided.

    Parameters
    ----------
    arch_spec:
        The architecture specification.
    noise_model:
        An optional physical noise model for gate/move noise insertion.
    aggressive_unroll:
        Use aggressive unrolling instead of simple folding.
    """

    arch_spec: layout.ArchSpec
    noise_model: NoiseModelABC | None = None
    aggressive_unroll: bool = False

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        return _emit_common(
            main,
            self.arch_spec,
            None,
            self.noise_model,
            self.aggressive_unroll,
            no_raise,
        )


@dataclass
class MoveToSquin:
    """Backwards-compatible rewrite pass.

    Works exactly as the original ``MoveToSquin``: when
    ``logical_initialization`` is provided it is used directly; otherwise falls
    through to the noise model (if any) for initialization.  ``InsertNoise`` is
    added whenever ``noise_model is not None``.
    """

    arch_spec: layout.ArchSpec
    logical_initialization: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None]
        | None
    ) = None
    noise_model: NoiseModelABC | None = None
    aggressive_unroll: bool = False

    def _resolve_initialize_kernel(self):
        """Resolve the initialization kernel.

        Priority: explicit logical_initialization param > noise model > None.
        """
        if self.logical_initialization is not None:
            return self.logical_initialization
        if self.noise_model is not None:
            clean, _ = self.noise_model.get_logical_initialize()
            return clean
        return None

    def _resolve_initialize_noise_kernel(self):
        """Resolve the noisy initialization kernel from the noise model."""
        if self.noise_model is not None:
            _, noisy = self.noise_model.get_logical_initialize()
            return noisy
        return None

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        return _emit_common(
            main,
            self.arch_spec,
            self._resolve_initialize_kernel(),
            self.noise_model,
            self.aggressive_unroll,
            no_raise,
            initialize_noise_kernel=self._resolve_initialize_noise_kernel(),
        )
