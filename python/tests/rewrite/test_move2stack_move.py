from kirin import ir
from kirin.dialects import func
from kirin.rewrite import Walk

from bloqade.lanes.dialects import move
from bloqade.lanes.rewrite.move2stack_move import RewriteMoveToStackMove


def test_load_and_store_are_removed():
    load = move.Load()
    store = move.Store(current_state=load.result)
    none_stmt = func.ConstantNone()
    ret = func.Return(none_stmt.result)
    block = ir.Block()
    for s in [load, store, none_stmt, ret]:
        block.stmts.append(s)

    Walk(RewriteMoveToStackMove()).rewrite(block)

    assert not any(isinstance(s, move.Load) for s in block.stmts)
    assert not any(isinstance(s, move.Store) for s in block.stmts)
