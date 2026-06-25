"""Const-prop registration for the movement dialect."""

from kirin import interp

from bloqade.gemini.common.dialects.movement import dialect as movement_dialect


def test_movement_dialect_registers_constprop_method_table():
    # ``Dialect.interps`` is the dict of registered method tables keyed by
    # interpreter key. The const-prop analysis looks under "constprop".
    table = movement_dialect.interps.get("constprop")
    assert table is not None, "movement dialect has no 'constprop' registration"
    assert isinstance(table, interp.MethodTable)
    assert type(table).__name__ == "CzPartnerConstProp"
