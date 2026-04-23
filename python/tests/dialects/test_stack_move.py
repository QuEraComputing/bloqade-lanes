from bloqade.lanes.dialects import stack_move


def test_dialect_exists():
    assert stack_move.dialect.name == "lanes.stack_move"


def test_typeinfer_registered():
    # Verify the dialect has a typeinfer MethodTable registered with
    # impls for Return and Halt — needed so type-inference analyses
    # running on stack_move IR (before the rewrite to move / func) can
    # propagate return-value types through the terminators.
    table = stack_move.dialect.interps.get("typeinfer")
    assert table is not None
    assert isinstance(table, stack_move.TypeInfer)

    # BoundedDef.signature is a tuple of Signature records whose `head`
    # is the statement class the impl is registered against. Access via
    # getattr keeps pyright from trying to resolve the dynamic impls.
    return_heads = {sig.head for sig in getattr(table, "return_").signature}
    halt_heads = {sig.head for sig in getattr(table, "halt").signature}
    assert stack_move.Return in return_heads
    assert stack_move.Halt in halt_heads
