"""Tests that the upstream ``scaling_factor`` on ``GeminiNoiseModelABC`` is
honored when lowering to squin kernels.

Regression for QuEraComputing/bloqade-lanes#785: ``generate_simple_noise_model``
used to read the raw ``*_px`` fields, which ignore ``scaling_factor`` entirely,
so every value of ``scaling_factor`` produced identical noise. The fix reads the
scaled getter APIs (``mover_pauli_rates``, ``two_qubit_pauli``, ...).

Loss probabilities are intentionally *not* scaled: upstream ``scaling_factor``
only touches the Pauli rates (per-category loss scaling is tracked upstream in
QuEraComputing/bloqade-circuit#836).
"""

import math

import pytest
from bloqade.cirq_utils.noise.model import GeminiOneZoneNoiseModel
from kirin.dialects.py import Constant

from bloqade.lanes.noise_model import (
    PAIRED_KEYS,
    generate_logical_noise_model,
    generate_simple_noise_model,
)


def _float_constants(kernel) -> set[float]:
    """Collect scalar float constants baked into a squin kernel closure.

    ``bool`` is a subclass of ``int`` (not ``float``) so kernel flags such as the
    ``loss`` toggle are naturally excluded.
    """
    out: set[float] = set()
    for stmt in kernel.callable_region.walk():
        if isinstance(stmt, Constant):
            data = stmt.value.unwrap()
            if isinstance(data, float):
                out.add(data)
    return out


def _list_constant(kernel) -> list[float]:
    """Return the first list/IList constant baked into a kernel, as floats."""
    for stmt in kernel.callable_region.walk():
        if not isinstance(stmt, Constant):
            continue
        data = stmt.value.unwrap()
        if isinstance(data, (bool, str, float, int)):
            continue
        try:
            return [float(x) for x in data]
        except TypeError:
            continue
    raise AssertionError("no list constant found in kernel")


def _has(consts, target: float) -> bool:
    return any(math.isclose(c, target, rel_tol=1e-9, abs_tol=1e-18) for c in consts)


# (kernel attribute, scaled getter property the kernel should bake). Expectations
# are derived from the getter APIs themselves (the source of truth), not by
# re-deriving raw * scale, so the tests stay valid if upstream changes how
# scaling is applied.
SINGLE_QUBIT_RATE_KERNELS = [
    ("lane_noise", "mover_pauli_rates"),
    ("idle_noise", "sitter_pauli_rates"),
    ("cz_unpaired_noise", "cz_unpaired_pauli_rates"),
    ("local_r_noise", "local_pauli_rates"),
    ("local_rz_noise", "local_pauli_rates"),
    ("global_r_noise", "global_pauli_rates"),
    ("global_rz_noise", "global_pauli_rates"),
]


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
@pytest.mark.parametrize("kernel_name,getter", SINGLE_QUBIT_RATE_KERNELS)
def test_single_qubit_pauli_rates_honor_scaling_factor(scale, kernel_name, getter):
    model = GeminiOneZoneNoiseModel(scaling_factor=scale)
    nm = generate_simple_noise_model(model)
    consts = _float_constants(getattr(nm, kernel_name))
    for expected in getattr(model, getter):
        assert _has(
            consts, expected
        ), f"{kernel_name}: expected scaled rate {expected}, got {sorted(consts)}"


@pytest.mark.parametrize("scale", [0.0, 0.5, 2.0])
def test_cz_paired_probabilities_honor_scaling_factor(scale):
    model = GeminiOneZoneNoiseModel(scaling_factor=scale)
    nm = generate_simple_noise_model(model)
    baked = _list_constant(nm.cz_paired_noise)
    assert len(baked) == len(PAIRED_KEYS)
    # Expected values come from the scaled getter (the source of truth) rather
    # than re-deriving raw[key] * scale.
    expected = model.two_qubit_pauli.error_probabilities
    for key, got in zip(PAIRED_KEYS, baked):
        assert math.isclose(
            got, expected[key], rel_tol=1e-9, abs_tol=1e-18
        ), f"paired {key}: expected {expected[key]}, got {got}"


def test_scaling_changes_baked_constants():
    """Regression: distinct scaling_factor values must produce distinct noise.

    Before the fix, all scaling_factor values produced identical baked constants.
    """
    base = _float_constants(
        generate_simple_noise_model(GeminiOneZoneNoiseModel()).lane_noise
    )
    doubled = _float_constants(
        generate_simple_noise_model(
            GeminiOneZoneNoiseModel(scaling_factor=2.0)
        ).lane_noise
    )
    assert base != doubled


def test_scaling_factor_zero_zeroes_rates_without_keyerror():
    """scaling_factor=0 must produce all-zero rates and not raise (paired dict
    lookup is robust to keys dropped when a scaled rate hits zero)."""
    nm = generate_simple_noise_model(GeminiOneZoneNoiseModel(scaling_factor=0.0))
    for name in ("lane_noise", "idle_noise", "cz_unpaired_noise", "local_r_noise"):
        assert _float_constants(getattr(nm, name)) <= {0.0}
    assert all(v == 0.0 for v in _list_constant(nm.cz_paired_noise))


def test_loss_probabilities_are_not_scaled():
    """Loss probabilities must stay raw even with a non-unit scaling_factor."""
    model = GeminiOneZoneNoiseModel(
        scaling_factor=2.0,
        move_loss_prob=0.1,
        sit_loss_prob=0.11,
        local_loss_prob=0.12,
        global_loss_prob=0.13,
    )
    nm = generate_simple_noise_model(model)

    cases = [
        ("lane_noise", 0.1),
        ("idle_noise", 0.11),
        ("local_r_noise", 0.12),
        ("global_r_noise", 0.13),
    ]
    for kernel_name, raw_loss in cases:
        consts = _float_constants(getattr(nm, kernel_name))
        assert _has(consts, raw_loss), f"{kernel_name}: raw loss {raw_loss} missing"
        assert not _has(
            consts, raw_loss * 2.0
        ), f"{kernel_name}: loss appears scaled ({raw_loss * 2.0} present)"


def test_no_loss_kernels_omit_loss_constants():
    """With loss=False the loss probabilities are not baked at all."""
    model = GeminiOneZoneNoiseModel(scaling_factor=2.0, move_loss_prob=0.1)
    nm = generate_simple_noise_model(model, loss=False)
    assert not _has(_float_constants(nm.lane_noise), 0.1)


def test_logical_init_kernels_honor_scaling_factor():
    """generate_logical_noise_model must scale the rates passed into the Steane
    initialization kernels."""
    scale = 3.0
    model = GeminiOneZoneNoiseModel(scaling_factor=scale)
    nm = generate_logical_noise_model(model)
    _, noisy = nm.get_logical_initialize()
    assert noisy is not None
    consts = _float_constants(noisy)
    # The px component of each getter should be baked into the init kernel.
    for getter in ("local_pauli_rates", "mover_pauli_rates", "sitter_pauli_rates"):
        expected = getattr(model, getter)[0]
        assert _has(
            consts, expected
        ), f"logical init: expected scaled {getter}[0]={expected}, got {sorted(consts)}"


def test_logical_init_cz_unpaired_rates_honor_scaling_factor():
    """The Steane initializer must use the scaled unpaired-CZ getter."""
    raw_rates = (0.011, 0.013, 0.017)
    model = GeminiOneZoneNoiseModel(
        scaling_factor=2.0,
        cz_unpaired_gate_px=raw_rates[0],
        cz_unpaired_gate_py=raw_rates[1],
        cz_unpaired_gate_pz=raw_rates[2],
    )

    nm = generate_logical_noise_model(model)
    _, noisy = nm.get_logical_initialize()
    assert noisy is not None
    consts = _float_constants(noisy)

    for raw, expected in zip(raw_rates, model.cz_unpaired_pauli_rates):
        assert _has(
            consts, expected
        ), f"logical init: expected scaled unpaired-CZ rate {expected}"
        assert not _has(
            consts, raw
        ), f"logical init: unexpectedly baked raw unpaired-CZ rate {raw}"
