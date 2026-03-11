import io
from functools import cache
from typing import TypeVar

from bloqade.analysis.validation.simple_nocloning import FlatKernelNoCloningValidation
from bloqade.decoders.dialects.annotate.stmts import SetDetector, SetObservable
from bloqade.gemini.analysis.logical_validation import GeminiLogicalValidation
from bloqade.gemini.analysis.measurement_validation import (
    GeminiTerminalMeasurementValidation,
)
from bloqade.gemini.logical.dialects.operations.stmts import (
    TerminalLogicalMeasurement,
)
from bloqade.stim.emit.stim_str import EmitStimMain
from bloqade.stim.upstream.from_squin import squin_to_stim
from kirin import ir, rewrite
from kirin.dialects import func, ilist, py
from kirin.validation import ValidationSuite

from bloqade import qubit
from bloqade.lanes import visualize
from bloqade.lanes.analysis.layout import LayoutHeuristicABC
from bloqade.lanes.arch.gemini import logical
from bloqade.lanes.arch.gemini.impls import generate_arch_hypercube
from bloqade.lanes.heuristics import fixed
from bloqade.lanes.heuristics.logical_placement import LogicalPlacementStrategyNoHome
from bloqade.lanes.noise_model import generate_simple_noise_model
from bloqade.lanes.rewrite import transversal
from bloqade.lanes.rewrite.move2squin.noise import NoiseModelABC
from bloqade.lanes.rewrite.squin2stim import RemoveReturn
from bloqade.lanes.transform import MoveToSquin
from bloqade.lanes.upstream import squin_to_move

__all__ = [
    "run_squin_kernel_validation",
    "compile_squin_to_move",
    "compile_squin_to_move_and_visualize",
    "compile_to_physical_squin_noise_model",
    "compile_to_physical_stim_program",
    "transversal_rewrites",
    "append_measurements_and_annotations",
]


def run_squin_kernel_validation(mt: ir.Method):
    """
    Run validation checks on a Squin kernel method.

    Args:
        mt (ir.Method): The Squin kernel method to validate.

    Returns:
        ValidationResult: A validation result object containing the
            validation errors, if they exist

    Note: To trigger an error run `run_squin_kernel_validation(mt).raise_if_invalid()`.

    """
    validator = ValidationSuite(
        [
            GeminiLogicalValidation,
            GeminiTerminalMeasurementValidation,
            FlatKernelNoCloningValidation,
        ]
    )
    return validator.validate(mt)


def transversal_rewrites(mt: ir.Method):
    """Apply transversal rewrite rules to a squin method.

    Args:
        mt (ir.Method): rewrite the method in place.

    Returns:
        ir.Method: The rewritten method.

    """

    rewrite.Walk(
        rewrite.Chain(
            transversal.RewriteLocations(logical.steane7_transversal_map),
            transversal.RewriteLogicalInitialize(logical.steane7_transversal_map),
            transversal.RewriteMoves(logical.steane7_transversal_map),
            transversal.RewriteGetItem(logical.steane7_transversal_map),
            transversal.RewriteLogicalToPhysicalConversion(),
        )
    ).rewrite(mt.code)

    return mt


def compile_squin_to_move(
    mt: ir.Method,
    transversal_rewrite: bool = False,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
    insert_return_moves: bool = True,
):
    """
    Compile a squin kernel to move dialect.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.

    Returns:
        ir.Method: The compiled move dialect method.
    """
    if layout_heuristic is None:
        layout_heuristic = fixed.LogicalLayoutHeuristic()

    mt = squin_to_move(
        mt,
        layout_heuristic=layout_heuristic,
        placement_strategy=LogicalPlacementStrategyNoHome(),
        insert_return_moves=insert_return_moves,
        no_raise=no_raise,
    )
    if transversal_rewrite:
        mt = transversal_rewrites(mt)

    return mt


def compile_squin_to_move_and_visualize(
    mt: ir.Method,
    interactive: bool = True,
    transversal_rewrite: bool = False,
    animated: bool = False,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
):
    """
    Compile a squin kernel to moves and visualize the program.

    Args:
        mt (ir.Method): The Squin kernel to compile.
        interactive (bool, optional): Whether to display the visualization interactively. Defaults to True.
        transversal_rewrite (bool, optional): Whether to apply transversal rewrite rules. Defaults to False.
        animated (bool, optional): Whether to use animated visualization for displaying moves. Defaults to False.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.
    """
    # Compile to move dialect
    mt = compile_squin_to_move(
        mt,
        transversal_rewrite,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
    )
    if transversal_rewrite:
        arch_spec = generate_arch_hypercube(4)
        marker = "o"
    else:
        arch_spec = logical.get_arch_spec()
        marker = "s"

    if animated:
        visualize.animated_debugger(
            mt, arch_spec, interactive=interactive, atom_marker=marker
        )
    else:
        visualize.debugger(mt, arch_spec, interactive=interactive, atom_marker=marker)


def compile_to_physical_squin_noise_model(
    mt: ir.Method,
    noise_model: NoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
) -> ir.Method:
    """
    Compiles a logical squin kernel to a physical squin kernel with noise channels inserted.

    Args:
        mt (ir.Method): The logical squin method to compile.
        noise_model (NoiseModelABC, optional): The noise model to insert during compilation. Defaults to None.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.

    Returns:
        ir.Method: The compiled physical squin method.
    """
    if noise_model is None:
        noise_model = generate_simple_noise_model()

    move_mt = compile_squin_to_move(
        mt,
        transversal_rewrite=True,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
    )
    transformer = MoveToSquin(
        arch_spec=generate_arch_hypercube(4),
        logical_initialization=logical.steane7_initialize,
        noise_model=noise_model,
        aggressive_unroll=False,
    )

    return transformer.emit(move_mt, no_raise=no_raise)


def compile_to_physical_stim_program(
    mt: ir.Method,
    noise_model: NoiseModelABC | None = None,
    no_raise: bool = True,
    layout_heuristic: LayoutHeuristicABC | None = None,
) -> str:
    """
    Compiles a logical squin kernel to a physical stim kernel with noise channels inserted.

    Args:
        mt (ir.Method): The logical squin method to compile.
        noise_model (NoiseModelABC, optional): The noise model to insert during compilation. Defaults to None.
        no_raise (bool, optional): Whether to suppress exceptions during compilation. Defaults to True.

    Returns:
        str: The compiled physical stim program as a string.
    """
    noise_kernel = compile_to_physical_squin_noise_model(
        mt,
        noise_model,
        no_raise=no_raise,
        layout_heuristic=layout_heuristic,
    )
    RemoveReturn().rewrite(noise_kernel.code)
    noise_kernel = squin_to_stim(noise_kernel)
    buf = io.StringIO()
    emit = EmitStimMain(dialects=noise_kernel.dialects, io=buf)
    emit.initialize()
    emit.run(node=noise_kernel)

    return buf.getvalue().strip()


_S = TypeVar("_S", bound=ir.Statement)


def _find_qubit_ssas(mt: ir.Method) -> list[ir.SSAValue]:
    """Walk the squin IR and collect SSA values for all qubit allocations."""
    return [
        stmt.result
        for stmt in mt.callable_region.walk()
        if isinstance(stmt, qubit.stmts.New)
    ]


def _find_return_stmt(mt: ir.Method) -> func.Return:
    """Find the func.Return statement at the end of the function body."""
    block = mt.callable_region.blocks[0]
    last = block.last_stmt
    assert isinstance(last, func.Return), f"Expected func.Return, got {type(last)}"
    return last


def _insert_before(stmt: _S, anchor: ir.Statement) -> _S:
    """Insert stmt before anchor and return stmt for chaining."""
    stmt.insert_before(anchor)
    return stmt


def append_measurements_and_annotations(
    mt: ir.Method,
    m2dets: list[list[int]] | None,
    m2obs: list[list[int]] | None,
) -> None:
    """Append terminal measurement, detector, and observable IR statements to a squin kernel.

    The method is mutated in-place.

    Args:
        mt: A squin ``ir.Method`` whose body returns ``None``.
        m2dets: Binary matrix of shape ``(num_total_meas, num_detectors)``.
            Each column defines a detector by its non-zero row indices.
        m2obs: Binary matrix of shape ``(num_total_meas, num_observables)``.
            Each column defines an observable by its non-zero row indices.
    """

    if m2dets is None and m2obs is None:
        raise ValueError("At least one of m2dets or m2obs must be provided")

    qubit_ssas = _find_qubit_ssas(mt)
    num_qubits = len(qubit_ssas)
    if num_qubits == 0:
        raise ValueError("No qubit allocations found in the kernel")

    m2 = m2dets if m2dets is not None else m2obs
    assert m2 is not None
    num_total_meas = len(m2)
    meas_per_qubit = num_total_meas // num_qubits
    assert (
        meas_per_qubit * num_qubits == num_total_meas
    ), "Incompatible shape of m2dets or m2obs"

    return_stmt = _find_return_stmt(mt)

    # insert TerminalLogicalMeasurement if not present
    terminal_measurement = next(
        (
            s
            for s in mt.callable_region.walk()
            if isinstance(s, TerminalLogicalMeasurement)
        ),
        None,
    )
    if terminal_measurement is not None:
        term_meas = terminal_measurement
    else:
        qlist_stmt = _insert_before(ilist.New(qubit_ssas), return_stmt)
        term_meas = _insert_before(
            TerminalLogicalMeasurement(qlist_stmt.result), return_stmt
        )

    @cache
    def _get_logical_measurement(q_idx: int) -> ir.SSAValue:
        (idx := py.Constant(q_idx)).insert_before(return_stmt)
        (getitem := py.GetItem(term_meas.result, idx.result)).insert_before(return_stmt)
        return getitem.result

    @cache
    def _get_physical_measurement(q_idx: int, m_idx: int) -> ir.SSAValue:
        (idx := py.Constant(m_idx)).insert_before(return_stmt)
        (
            getitem := py.GetItem(_get_logical_measurement(q_idx), idx.result)
        ).insert_before(return_stmt)
        return getitem.result

    # insert detectors
    if m2dets is not None:
        for j in range(len(m2dets[0])):
            indices = [i for i, row in enumerate(m2dets) if row[j]]
            meas_ssas = [
                _get_physical_measurement(*divmod(idx, meas_per_qubit))
                for idx in indices
            ]
            meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)

            coord_0 = _insert_before(py.Constant(0.0), return_stmt)
            coord_1 = _insert_before(py.Constant(float(j)), return_stmt)
            coords = _insert_before(
                ilist.New([coord_0.result, coord_1.result]), return_stmt
            )

            _insert_before(SetDetector(meas_list.result, coords.result), return_stmt)

    # insert observables
    if m2obs is not None:
        for j in range(len(m2obs[0])):
            indices = [i for i, row in enumerate(m2obs) if row[j]]
            meas_ssas = [
                _get_physical_measurement(*divmod(idx, meas_per_qubit))
                for idx in indices
            ]
            meas_list = _insert_before(ilist.New(meas_ssas), return_stmt)
            _insert_before(SetObservable(meas_list.result), return_stmt)
