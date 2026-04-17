from __future__ import annotations

import pytest
from benchmarks.utils.visualize_kernel_moves import _build_parser, _load_case_kernel
from kirin import ir


def test_build_parser_uses_random_stabilizers_defaults():
    parser = _build_parser()
    args = parser.parse_args(["--case-id", "multi_qubit_rb_5_8_ZZXZZ"])

    assert args.bucket == "random_stabilizers"
    assert args.architecture == "logical"
    assert args.strategy == "astar"
    assert args.animated is False


def test_load_case_kernel_returns_single_method():
    kernel = _load_case_kernel(
        bucket="random_stabilizers",
        case_id="multi_qubit_rb_5_8_ZZXZZ",
    )
    assert isinstance(kernel, ir.Method)


def test_load_case_kernel_rejects_unknown_module():
    with pytest.raises(ValueError, match="Cannot import case module"):
        _load_case_kernel(
            bucket="random_stabilizers",
            case_id="definitely_not_a_case",
        )
