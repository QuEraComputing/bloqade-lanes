from dataclasses import dataclass
from typing import Any, Literal

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

InitKernel = (
    ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Any]], None] | None
)


@dataclass
class MoveToSquinBase:
    """Base class for all MoveToSquin variants.

    Subclasses override ``_get_initialize_kernel``, ``_get_noise_model``,
    and ``_get_initialize_noise_kernel`` to control which kernels are
    passed to the rewrite rules.
    """

    arch_spec: layout.ArchSpec
    aggressive_unroll: bool = False

    def _get_initialize_kernel(self) -> InitKernel:
        """Return the initialization kernel for InsertGates, or None."""
        return None

    def _get_noise_model(self) -> NoiseModelABC | None:
        """Return the noise model for InsertNoise, or None to skip noise."""
        return None

    def _get_initialize_noise_kernel(self) -> InitKernel:
        """Return the noisy initialization kernel for InsertNoise, or None."""
        return None

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        main = main.similar(main.dialects.union(squin.kernel.discard(scf.lowering)))

        vqpu = atom.AtomInterpreter(main.dialects, arch_spec=self.arch_spec)
        run_method = vqpu.run_no_raise if no_raise else vqpu.run

        frame, _ = run_method(main)
        qubit_rule = move2squin.InsertQubits(frame)
        rewrite.Walk(qubit_rule).rewrite(main.code)

        noise_model = self._get_noise_model()
        rules = [
            move2squin.InsertGates(
                self.arch_spec,
                qubit_rule.physical_ssa_values,
                frame,
                self._get_initialize_kernel(),
            ),
            move2squin.InsertMeasurements(qubit_rule.physical_ssa_values, frame),
        ]
        if noise_model is not None:
            rules.append(
                move2squin.InsertNoise(
                    self.arch_spec,
                    qubit_rule.physical_ssa_values,
                    frame,
                    noise_model,
                    initialize_noise_kernel=self._get_initialize_noise_kernel(),
                ),
            )

        rewrite.Walk(rewrite.Chain(*rules)).rewrite(main.code)

        if self.aggressive_unroll:
            agg.AggressiveUnroll(main.dialects).fixpoint(main)
        else:
            agg.Fold(main.dialects)(main)

        rewrite.Walk(SquinU3ToClifford()).rewrite(main.code)

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
class MoveToSquinLogical(MoveToSquinBase):
    """Rewrite pass for **logical** compilation.

    Handles ``PhysicalInitialize`` rewrites using clean/noisy initialization
    kernels from the noise model and optionally inserts gate/move noise.
    """

    noise_model: LogicalNoiseModelABC = None  # type: ignore[assignment]
    add_noise: bool = False

    def _get_initialize_kernel(self) -> InitKernel:
        clean, _ = self.noise_model.get_logical_initialize()
        return clean

    def _get_noise_model(self) -> NoiseModelABC | None:
        return self.noise_model if self.add_noise else None

    def _get_initialize_noise_kernel(self) -> InitKernel:
        if not self.add_noise:
            return None
        _, noisy = self.noise_model.get_logical_initialize()
        return noisy


@dataclass
class MoveToSquinPhysical(MoveToSquinBase):
    """Rewrite pass for **physical** compilation.

    No initialization kernel handling. Noise is inserted only when a noise
    model is provided.
    """

    noise_model: NoiseModelABC | None = None

    def _get_noise_model(self) -> NoiseModelABC | None:
        return self.noise_model


@dataclass
class MoveToSquin(MoveToSquinBase):
    """Backwards-compatible rewrite pass.

    When ``logical_initialization`` is provided it is used directly;
    otherwise falls through to the noise model (if any) for initialization.
    ``InsertNoise`` is added whenever ``noise_model is not None``.
    """

    logical_initialization: (
        ir.Method[[float, float, float, ilist.IList[qubit.Qubit, Literal[7]]], None]
        | None
    ) = None
    noise_model: NoiseModelABC | None = None

    def _get_initialize_kernel(self) -> InitKernel:
        if self.logical_initialization is not None:
            return self.logical_initialization
        if self.noise_model is not None:
            clean, _ = self.noise_model.get_logical_initialize()
            return clean
        return None

    def _get_noise_model(self) -> NoiseModelABC | None:
        return self.noise_model

    def _get_initialize_noise_kernel(self) -> InitKernel:
        if self.noise_model is not None:
            _, noisy = self.noise_model.get_logical_initialize()
            return noisy
        return None
