import abc
from dataclasses import dataclass, field

from bloqade.rewrite.passes import aggressive_unroll as agg
from bloqade.squin.rewrite import SquinU3ToClifford
from kirin import ir, rewrite
from kirin.dialects import scf
from kirin.passes import TypeInfer

from bloqade import squin
from bloqade.lanes.analysis import atom
from bloqade.lanes.arch.spec import ArchSpec
from bloqade.lanes.bytecode import Program
from bloqade.lanes.bytecode.encode import dump_program
from bloqade.lanes.dialects import move, stack_move
from bloqade.lanes.rewrite import move2squin
from bloqade.lanes.rewrite.move2squin import (
    LogicalInitKernel as LogicalInitKernel,
    LogicalNoiseModelABC as LogicalNoiseModelABC,
    NoiseModelABC as NoiseModelABC,
    SimpleLogicalNoiseModel as SimpleLogicalNoiseModel,
    SimpleNoiseModel as SimpleNoiseModel,
)
from bloqade.lanes.rewrite.move2stack_move import RewriteMoveToStackMove
from bloqade.lanes.rewrite.stackify import stackify
from bloqade.lanes.utils import statements_outside_dialect_group

InitKernel = LogicalInitKernel | None


@dataclass
class MoveToSquinBase(abc.ABC):
    """Base class for all MoveToSquin variants.

    Subclasses must implement ``_get_initialize_kernel``,
    ``_get_noise_model``, and ``_get_initialize_noise_kernel`` to control
    which kernels are passed to the rewrite rules.
    """

    arch_spec: ArchSpec
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


@dataclass
class MoveToStackMove:
    """Lower a ``move``-dialect kernel to a canonicalized ``stack_move`` kernel.

    ``emit`` runs the full move → stack_move lowering pipeline, producing an
    ``ir.Method`` that is stack-consistent and ready for bytecode emission:

    1. ``RewriteMoveToStackMove`` — in-place move → stack_move rewrite
       (strips Load/Store state threading, materialises address attributes
       as ``Const*`` SSA values, rebuilds Measure/AwaitMeasure/GetItem).
    2. DCE + CSE to a fixpoint — the cleanup the encoder pipeline expects
       before stackification (see ``stackify``'s docstring).
    3. ``stackify`` — normalise into stack-consistent form (single block,
       each SSA value used at most once, defining statements in stack order).

    ``RewriteMoveToStackMove`` only lowers the subset of ``move`` statements
    the bytecode path supports, so after dropping ``move`` from the dialect
    group ``emit`` runs ``statements_outside_dialect_group`` and raises if any
    statement is left outside the group — Kirin's ``verify()`` does not check
    dialect-group membership, so an unlowered statement would otherwise slip
    through and fail lazily inside ``dump_program``.

    ``emit_bytecode`` runs ``emit`` and encodes the result to a bytecode
    ``Program`` via ``dump_program``.
    """

    arch_spec: ArchSpec

    def emit(self, main: ir.Method, no_raise: bool = True) -> ir.Method:
        # Copy into a dialect group that includes stack_move so the rewritten
        # statements are legal members of the method's dialects.
        out = main.similar(main.dialects.union([stack_move.dialect]))

        # move → stack_move (single pass; the rule deletes the move statements).
        rewrite.Walk(RewriteMoveToStackMove(arch_spec=self.arch_spec)).rewrite(out.code)

        # DCE + CSE, matching the cleanup the real pipeline runs before stackify.
        rewrite.Fixpoint(
            rewrite.Walk(
                rewrite.Chain(
                    rewrite.DeadCodeElimination(),
                    rewrite.CommonSubexpressionElimination(),
                )
            )
        ).rewrite(out.code)

        # Drop the now-unused move dialect from the group.
        out = out.similar(out.dialects.discard(move.dialect))

        # Fail fast: RewriteMoveToStackMove passes through any move statement it
        # doesn't handle, and verify() does not police dialect-group membership,
        # so an unlowered statement would otherwise surface as a confusing
        # EncodingError deep inside dump_program.
        leftover = statements_outside_dialect_group(out)
        if leftover:
            kinds = sorted({type(stmt).__name__ for stmt in leftover})
            raise ValueError(
                "MoveToStackMove left statements outside the stack_move dialect "
                f"group: {', '.join(kinds)}; RewriteMoveToStackMove does not "
                "lower them, so the kernel cannot be emitted as stack_move IR"
            )

        # Canonicalize into stack-consistent form, ready for dump_program.
        stackify(out)

        if not no_raise:
            out.verify()

        return out

    def emit_bytecode(
        self,
        main: ir.Method,
        version: tuple[int, int] = (1, 0),
        no_raise: bool = True,
    ) -> Program:
        return dump_program(self.emit(main, no_raise=no_raise), version=version)
