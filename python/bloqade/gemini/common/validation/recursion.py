"""Deprecated compatibility imports; use ``bloqade.lanes.validation.recursion``."""

from bloqade.lanes.validation.recursion import (
    CallGraph as CallGraph,
    Cycle as Cycle,
    NoOpaqueCallValidation as NoOpaqueCallValidation,
    NoRecursionValidation as NoRecursionValidation,
    _callable_operands as _callable_operands,
    _constant_method as _constant_method,
    _name as _name,
    _parameter_values as _parameter_values,
    _self_value as _self_value,
    _sort_key as _sort_key,
    check_call_graph as check_call_graph,
    format_cycle as format_cycle,
)
