from bloqade.lanes.dialects import stack_move


def test_dialect_exists():
    assert stack_move.dialect.name == "lanes.stack_move"
