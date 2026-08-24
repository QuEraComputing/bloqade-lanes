from __future__ import annotations

from dataclasses import dataclass, field
from importlib.metadata import version
from typing import TYPE_CHECKING

from bloqade.core.device import Device
from bloqade.core.device.task import (
    KernelBatchTask,
    ParameterScanTask,
    SingleKernelTask,
)
from kirin import ir
from kirin.validation import ValidationSuite

if TYPE_CHECKING:
    from .future import GeminiLogicalFuture


def _default_validation_suite() -> ValidationSuite:
    """Build validations only when a logical device is constructed.

    Importing a device must not initialize Lanes analyses: Lanes dialect imports
    can themselves enter the Gemini package while those analyses are loading.
    """
    from bloqade.analysis.validation.simple_nocloning import (
        FlatKernelNoCloningValidation,
    )

    from ...common.validation.duplicate_address import DuplicateAddressValidation
    from ...logical.validation.clifford.analysis import GeminiLogicalValidation
    from ...logical.validation.measurement.analysis import (
        GeminiTerminalMeasurementValidation,
    )

    return ValidationSuite(
        [
            GeminiLogicalValidation,
            GeminiTerminalMeasurementValidation,
            FlatKernelNoCloningValidation,
            DuplicateAddressValidation,
        ]
    )


def _logical_dialect_group() -> ir.DialectGroup:
    from ...logical.group import kernel

    return kernel


def _logical_future_cls() -> type[GeminiLogicalFuture]:
    from .future import GeminiLogicalFuture

    return GeminiLogicalFuture


def _single_kernel_task_cls() -> type[SingleKernelTask[GeminiLogicalFuture]]:
    from .task import GeminiSingleKernelTask

    return GeminiSingleKernelTask


def _parameter_scan_task_cls() -> type[ParameterScanTask[GeminiLogicalFuture]]:
    from .task import GeminiParameterScanTask

    return GeminiParameterScanTask


def _kernel_batch_task_cls() -> type[KernelBatchTask[GeminiLogicalFuture]]:
    from .task import GeminiKernelBatchTask

    return GeminiKernelBatchTask


_bloqade_version = version("bloqade-circuit")


@dataclass(kw_only=True)
class GeminiLogicalDevice(Device["GeminiLogicalFuture"]):
    """Device that builds tasks for the Gemini logical backend.

    Wires the Gemini-specific task and future classes into the generic
    `Device` factory and defaults `context_name` to "gemini-logical".
    """

    program_language: str = "squin"
    language_version: str = "0.1.0"
    validation_suite: ValidationSuite | None = field(
        default_factory=_default_validation_suite
    )
    dialect_group: ir.DialectGroup = field(default_factory=_logical_dialect_group)

    future_cls: type[GeminiLogicalFuture] = field(default_factory=_logical_future_cls)
    single_kernel_task_cls: type[SingleKernelTask[GeminiLogicalFuture]] = field(
        default_factory=_single_kernel_task_cls,
        init=False,
    )
    parameter_scan_task_cls: type[ParameterScanTask[GeminiLogicalFuture]] = field(
        default_factory=_parameter_scan_task_cls,
        init=False,
    )
    kernel_batch_task_cls: type[KernelBatchTask[GeminiLogicalFuture]] = field(
        default_factory=_kernel_batch_task_cls,
        init=False,
    )

    context_name: str = "gemini-logical"
