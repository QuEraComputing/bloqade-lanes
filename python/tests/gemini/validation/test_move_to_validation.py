"""Tests for MoveToValidation — eager per-statement validation (failures 1-5)."""

from kirin import ir, types
from kirin.analysis import const
from kirin.dialects import func, ilist
from kirin.prelude import structural_no_opt

from bloqade import types as bloqade_types
from bloqade.gemini.common.dialects.arrange import dialect as arrange_dialect
from bloqade.gemini.common.dialects.arrange.stmts import MoveTo
from bloqade.gemini.common.validation.move_to import MoveToValidation
from bloqade.lanes.arch.gemini.logical import get_arch_spec
from bloqade.lanes.bytecode.encoding import LocationAddress

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIALECTS = structural_no_opt.union([arrange_dialect])

_LOC_A = LocationAddress(zone_id=0, word_id=0, site_id=0)
_LOC_B = LocationAddress(zone_id=0, word_id=1, site_id=0)


def _build_test_method(stmts: list[ir.Statement]) -> ir.Method:
    """Build a minimal ir.Method containing the given statements."""
    block = ir.Block(list(stmts) + [func.Return()])
    block.args.append_from(types.MethodType, name="self")
    region = ir.Region([block])
    func_stmt = func.Function(
        sym_name="test_fn",
        signature=func.Signature(inputs=(), output=types.NoneType),
        body=region,
    )
    return ir.Method(
        dialects=_DIALECTS,
        sym_name="test_fn",
        arg_names=["self"],
        code=func_stmt,
    )


def _validator() -> MoveToValidation:
    return MoveToValidation(arch_spec=get_arch_spec())


# ---------------------------------------------------------------------------
# Failure 1: length mismatch
# ---------------------------------------------------------------------------


def test_length_mismatch_one_qubit_two_locs():
    """1 qubit, 2 locations — failure 1 (length mismatch)."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0,))

    locs_new = ilist.New(values=(ir.TestValue(), ir.TestValue()))
    locs_new.result.hints["const"] = const.Value((_LOC_A, _LOC_B))

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert len(errors) >= 1
    assert any("len(qubits)" in str(e) for e in errors)


def test_length_mismatch_two_qubits_one_loc():
    """2 qubits, 1 location — failure 1 (length mismatch)."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    q1 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0, q1))

    locs_new = ilist.New(values=(ir.TestValue(),))
    locs_new.result.hints["const"] = const.Value((_LOC_A,))

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert len(errors) >= 1
    assert any("len(qubits)" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Failure 2: non-const locations
# ---------------------------------------------------------------------------


def test_non_const_locations():
    """Locations without a const hint — failure 2."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0,))
    locs_new = ilist.New(values=(ir.TestValue(),))
    # No const hint set on locs_new.result

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert len(errors) >= 1
    assert any("compile-time constant" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Failure 3: out-of-range LocationAddress
# ---------------------------------------------------------------------------


def test_out_of_range_location():
    """zone_id=99 does not exist — failure 3 (out-of-range)."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0,))

    bad_loc = LocationAddress(zone_id=99, word_id=0, site_id=0)
    locs_new = ilist.New(values=(ir.TestValue(),))
    locs_new.result.hints["const"] = const.Value((bad_loc,))

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert len(errors) >= 1
    assert any("Invalid location address" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Failure 4: duplicate destination addresses
# ---------------------------------------------------------------------------


def test_duplicate_destinations():
    """Two identical LocationAddress in one call — failure 4."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    q1 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0, q1))

    locs_new = ilist.New(values=(ir.TestValue(), ir.TestValue()))
    locs_new.result.hints["const"] = const.Value((_LOC_A, _LOC_A))  # duplicate

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert any("duplicate destination" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Failure 5: duplicate qubit SSA values
# ---------------------------------------------------------------------------


def test_duplicate_qubit_ssa_values():
    """Same Qubit SSA value used twice — failure 5."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0, q0))  # same SSA value twice

    locs_new = ilist.New(values=(ir.TestValue(), ir.TestValue()))
    locs_new.result.hints["const"] = const.Value((_LOC_A, _LOC_B))

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert any("same Qubit SSA value" in str(e) for e in errors)


# ---------------------------------------------------------------------------
# Valid case — no errors
# ---------------------------------------------------------------------------


def test_valid_single_qubit_no_errors():
    """Valid single-qubit move with a valid location — no errors."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0,))

    locs_new = ilist.New(values=(ir.TestValue(),))
    locs_new.result.hints["const"] = const.Value((_LOC_A,))

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert errors == []


def test_valid_two_qubit_no_errors():
    """Valid two-qubit move with distinct valid locations — no errors."""
    q0 = ir.TestValue(type=bloqade_types.QubitType)
    q1 = ir.TestValue(type=bloqade_types.QubitType)
    qubits_new = ilist.New(values=(q0, q1))

    locs_new = ilist.New(values=(ir.TestValue(), ir.TestValue()))
    locs_new.result.hints["const"] = const.Value((_LOC_A, _LOC_B))

    stmt = MoveTo(qubits_new.result, locs_new.result)
    method = _build_test_method([qubits_new, locs_new, stmt])

    _, errors = _validator().run(method)
    assert errors == []


# ---------------------------------------------------------------------------
# Non-ilist qubits argument
# ---------------------------------------------------------------------------


def test_non_ilist_qubits():
    """qubits argument that is not an ilist.New — reports an error."""
    qubits_val = ir.TestValue()  # raw SSA value, not ilist.New
    locs_new = ilist.New(values=(ir.TestValue(),))
    locs_new.result.hints["const"] = const.Value((_LOC_A,))

    stmt = MoveTo(qubits_val, locs_new.result)
    method = _build_test_method([locs_new, stmt])

    _, errors = _validator().run(method)
    assert len(errors) >= 1
    assert any("literal list" in str(e) for e in errors)
