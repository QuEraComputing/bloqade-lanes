"""Const-prop registration for the arch dialect (cz_partner)."""

from kirin import interp

from bloqade.lanes.dialects.arch import dialect as arch_dialect
from bloqade.lanes.dialects.arch.constprop import CzPartnerConstProp


def test_arch_dialect_registers_constprop_method_table():
    # ``Dialect.interps`` is the dict of registered method tables keyed by
    # interpreter key. The const-prop analysis looks under "constprop".
    table = arch_dialect.interps.get("constprop")
    assert table is not None, "arch dialect has no 'constprop' registration"
    assert isinstance(table, interp.MethodTable)
    assert isinstance(table, CzPartnerConstProp)
