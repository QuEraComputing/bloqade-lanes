import abc
from dataclasses import dataclass, field
from typing import Any

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
class MoveToSquinBase(abc.ABC):
    """Base class for all MoveToSquin variants.

    Subclasses must implement ``_get_initialize_kernel``,
    ``_get_noise_model``, and ``_get_initialize_noise_kernel`` to control
    which kernels are passed to the rewrite rules.
    """

    arch_spec: layout.ArchSpec
    aggressive_unroll: bool = field(default=False, kw_only=True)

    @abc.abstractmethod
    def _get_initialize_kernel(self) -> InitKernel:
        """Return the initialization kernel for InsertGates, or None."""
        ...

    @abc.abstractmethod
    def _get_noise_model(self) -> NoiseModelABC | None:
        """Return the noise model for InsertNoise, or None to skip noise."""
        ...

    @abc.abstractmethod
    def _get_initialize_noise_kernel(self) -> InitKernel:
        """Return the noisy initialization kernel for InsertNoise, or None."""
        ...

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

    When ``add_noise`` is ``False``, uses the clean initialization kernel
    from the noise model for ``InsertGates`` to rewrite
    ``PhysicalInitialize`` nodes. No noise is inserted.

    When ``add_noise`` is ``True``, the clean initialization kernel is
    **not** passed to ``InsertGates``. Instead, only ``InsertNoise``
    handles initialization using the noisy kernel, and gate/move noise
    is inserted as well. This ensures initialization is applied exactly
    once — either clean or noisy, never both.
    """

    noise_model: LogicalNoiseModelABC
    add_noise: bool = False

    def _get_initialize_kernel(self) -> InitKernel:
        if self.add_noise:
            return None

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

    noise_model: NoiseModelABC = None  # type: ignore[assignment]

    def _get_initialize_kernel(self) -> InitKernel:
        return None

    def _get_noise_model(self) -> NoiseModelABC | None:
        return self.noise_model

    def _get_initialize_noise_kernel(self) -> InitKernel:
        return None
