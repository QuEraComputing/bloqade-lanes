from bloqade.analysis.address.analysis import AddressAnalysis
from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin import interp
from kirin.analysis import ForwardFrame

from ._dialect import dialect
from .stmts import NewAt


@dialect.register(key="qubit.address")
class _GeminiOperationsAddressMethods(interp.MethodTable):
    @interp.impl(NewAt)
    def new_at(
        self,
        interp_: AddressAnalysis,
        frame: ForwardFrame[Address],
        stmt: NewAt,
    ):
        addr = AddressQubit(interp_.next_address)
        interp_.next_address += 1
        return (addr,)
