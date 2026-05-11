from bloqade.lanes.dialects import stack_move


def test_dialect_exists():
    assert stack_move.dialect.name == "lanes.stack_move"


def test_no_typeinfer_table_registered():
    # Return and Halt lowered to kirin.basic's func dialect, so
    # stack_move no longer needs its own typeinfer MethodTable — every
    # remaining stack_move statement's result type is fully determined
    # by its declaration, and func.Return / func.ConstantNone carry
    # their own type-inference methods.
    assert stack_move.dialect.interps.get("typeinfer") is None
