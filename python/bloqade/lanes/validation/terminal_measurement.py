from dataclasses import dataclass, field
from typing import Any

import bloqade.qubit as qubit
from bloqade.analysis import address, measure_id
from bloqade.analysis.address.impls import Func as AddressFuncMethodTable
from bloqade.analysis.measure_id.lattice import MeasureIdTuple
from bloqade.squin import gate
from kirin import interp as _interp, ir
from kirin.analysis import Forward, ForwardFrame
from kirin.dialects import func
from kirin.lattice import EmptyLattice
from kirin.validation import ValidationPass

from bloqade.lanes.dialects import steane713 as operations
from bloqade.lanes.dialects.steane713 import extensions as extensions


@dataclass
class _SteaneTerminalMeasurementValidationAnalysis(Forward[EmptyLattice]):
    keys = ("lanes.validate.terminal_measurement",)

    measurement_analysis_results: ForwardFrame
    unique_qubits_allocated: int = 0
    terminal_measurement_encountered: bool = False
    lattice = EmptyLattice

    def check_gate_after_measurement(self, stmt: ir.Statement) -> None:
        """Report ``stmt`` if the terminal measurement has already been seen.

        The terminal measurement consumes every allocated qubit, so any gate
        evaluated after it acts on a measured qubit, whichever qubit it names.
        """
        if self.terminal_measurement_encountered:
            self.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    f"Gate {stmt.name} is applied after the terminal measurement; "
                    "no gates are allowed after terminal_measure in Gemini "
                    "logical programs!",
                ),
            )

    # boilerplate, not really worried about these right now
    def eval_fallback(self, frame: ForwardFrame, node: ir.Statement):
        return tuple(self.lattice.bottom() for _ in range(len(node.results)))

    def method_self(self, method: ir.Method) -> EmptyLattice:
        return self.lattice.bottom()

    def run_lattice(
        self,
        callee: EmptyLattice,
        inputs: tuple[EmptyLattice, ...],
        keys: tuple[str, ...],
        kwargs: tuple[EmptyLattice, ...],
    ) -> EmptyLattice:
        """Handle a dynamic ``func.Call``.

        Required, not optional: ``impls.Func`` subclasses the address analysis'
        ``func`` method table, so this analysis inherits a ``func.Call`` impl
        that calls ``run_lattice`` on the interpreter. Without an override, any
        kernel containing a dynamic call dies with an ``AttributeError``.

        Dynamic callees are deliberately not descended into. ``EmptyLattice``
        carries no information, so a callee's return value would add nothing to
        the frame; the cost is that a terminal-measurement violation *inside* a
        dynamically called kernel goes unreported.
        """
        return self.lattice.bottom()


@dataclass
class SteaneTerminalMeasurementValidation(ValidationPass):

    analysis_cache: dict = field(default_factory=dict)

    def name(self) -> str:
        return "Gemini Terminal Measurement Validation"

    def get_required_analyses(self) -> list[type]:
        return [measure_id.MeasurementIDAnalysis]

    def set_analysis_cache(self, cache: dict[type, Any]) -> None:
        self.analysis_cache.update(cache)

    def run(self, method: ir.Method) -> tuple[Any, list[ir.ValidationError]]:

        # get the data out of the cache and forward it to the underlying analysis
        measurement_analysis_results = self.analysis_cache.get(
            measure_id.MeasurementIDAnalysis
        )

        address_analysis = address.AddressAnalysis(dialects=method.dialects)
        address_analysis.run(method)
        unique_qubits_allocated = address_analysis.qubit_count

        assert (
            measurement_analysis_results is not None
        ), "Measurement ID analysis results not found in cache"

        analysis = _SteaneTerminalMeasurementValidationAnalysis(
            method.dialects,
            measurement_analysis_results,
            unique_qubits_allocated=unique_qubits_allocated,
        )

        frame, _ = analysis.run(method)

        errors = list(analysis.get_validation_errors())

        if (
            unique_qubits_allocated > 0
            and not analysis.terminal_measurement_encountered
        ):
            return_stmt = method.callable_region.blocks[0].last_stmt
            assert return_stmt is not None
            errors.append(
                ir.ValidationError(
                    return_stmt,
                    "Gemini logical programs must contain exactly one "
                    "logical.terminal_measure statement.",
                )
            )

        return frame, errors


@operations.dialect.register(key="lanes.validate.terminal_measurement")
class __GeminiLogicalMeasurementValidation(_interp.MethodTable):
    @_interp.impl(operations.stmts.TerminalLogicalMeasurement)
    def terminal_measure(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: operations.stmts.TerminalLogicalMeasurement,
    ):

        # should only be one terminal measurement EVER
        if not interp.terminal_measurement_encountered:
            interp.terminal_measurement_encountered = True
        else:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Multiple terminal measurements are not allowed in Gemini logical programs!",
                ),
            )
            return (interp.lattice.bottom(),)

        measurement_analysis_results = interp.measurement_analysis_results
        total_qubits_allocated = interp.unique_qubits_allocated

        # NOTE: `entries` rather than `Frame.get`, which *raises* on a missing
        # key. The analysis has no entry for this value when the statement sits
        # inside a callee that was not inlined -- it was evaluated in a nested
        # frame -- and the branch below already handles "no usable result", so a
        # miss belongs there rather than escaping as an `InterpreterError` for
        # `ValidationSuite` to re-report as "Validation pass '...' failed:" plus
        # a traceback. Reading `entries` directly also skips `get`'s parent-frame
        # fallback, which is unreachable here: this is the root frame the suite
        # cached, so its `parent` is None.
        measure_lattice_element = measurement_analysis_results.entries.get(stmt.result)

        if not isinstance(measure_lattice_element, MeasureIdTuple):
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "Measurement ID Analysis failed to produce the necessary results needed for validation.",
                ),
            )
            return (interp.lattice.bottom(),)

        if len(measure_lattice_element.data) != total_qubits_allocated:
            interp.add_validation_error(
                stmt,
                ir.ValidationError(
                    stmt,
                    "The number of qubits in the terminal measurement does not match the number of total qubits allocated! "
                    + f"{total_qubits_allocated} qubits were allocated but only {len(measure_lattice_element.data)} were measured.",
                ),
            )
            return (interp.lattice.bottom(),)

        return (interp.lattice.bottom(),)


@extensions.dialect.register(key="lanes.validate.terminal_measurement")
class __GeminiLogicalExtensionsMeasurementValidation(_interp.MethodTable):
    @_interp.impl(extensions.stmts.StarRz)
    def star_rz(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: extensions.stmts.StarRz,
    ):
        interp.check_gate_after_measurement(stmt)
        return ()


@qubit.dialect.register(key="lanes.validate.terminal_measurement")
class _QubitGeminiMeasurementValidation(_interp.MethodTable):
    # This is a non-logical measurement, can safely flag as invalid
    @_interp.impl(qubit.stmts.Measure)
    def measure(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: qubit.stmts.Measure,
    ):

        interp.add_validation_error(
            stmt,
            ir.ValidationError(
                stmt,
                "Non-terminal measurements are not allowed in Gemini programs!",
            ),
        )

        return (interp.lattice.bottom(),)


@gate.dialect.register(key="lanes.validate.terminal_measurement")
class _GateGeminiMeasurementValidation(_interp.MethodTable):
    # NOTE: every concrete statement of the gate dialect must be listed here;
    # a gate missing from this list falls through to `eval_fallback` and is
    # silently accepted after the terminal measurement. The test suite checks
    # this list against `gate.dialect.stmts`.
    @_interp.impl(gate.stmts.X)
    @_interp.impl(gate.stmts.Y)
    @_interp.impl(gate.stmts.Z)
    @_interp.impl(gate.stmts.H)
    @_interp.impl(gate.stmts.T)
    @_interp.impl(gate.stmts.S)
    @_interp.impl(gate.stmts.SqrtX)
    @_interp.impl(gate.stmts.SqrtY)
    @_interp.impl(gate.stmts.Rx)
    @_interp.impl(gate.stmts.Ry)
    @_interp.impl(gate.stmts.Rz)
    @_interp.impl(gate.stmts.U3)
    @_interp.impl(gate.stmts.PhasedXZ)
    @_interp.impl(gate.stmts.CX)
    @_interp.impl(gate.stmts.CY)
    @_interp.impl(gate.stmts.CZ)
    @_interp.impl(gate.stmts.CCZ)
    @_interp.impl(gate.stmts.Swap)
    def gate_after_measurement(
        self,
        interp: _SteaneTerminalMeasurementValidationAnalysis,
        frame: ForwardFrame,
        stmt: gate.stmts.Gate,
    ):
        interp.check_gate_after_measurement(stmt)
        return ()


@func.dialect.register(key="lanes.validate.terminal_measurement")
class Func(AddressFuncMethodTable):
    pass
