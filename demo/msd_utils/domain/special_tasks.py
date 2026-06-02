"""Compatibility re-exports for Gemini special-task helpers."""

from bloqade.gemini.decoding.special_tasks import (
    _apply_prefix_prepare_to_task,
    _attach_special_circuit_kernel,
    _build_compiled_unitary_prefix_circuit,
    _build_physical_prefix_source_tsim_circuit,
    _build_task,
    _clear_task_tsim_artifacts,
    _first_nonunitary_instruction_index,
    _override_task_tsim_circuit,
    _prepend_inverse_tsim_circuit,
    _set_task_override,
    apply_special_tsim_circuit_strategy,
    build_task_map,
)

__all__ = [
    "_apply_prefix_prepare_to_task",
    "_attach_special_circuit_kernel",
    "_build_compiled_unitary_prefix_circuit",
    "_build_physical_prefix_source_tsim_circuit",
    "_build_task",
    "_clear_task_tsim_artifacts",
    "_first_nonunitary_instruction_index",
    "_override_task_tsim_circuit",
    "_prepend_inverse_tsim_circuit",
    "_set_task_override",
    "apply_special_tsim_circuit_strategy",
    "build_task_map",
]
