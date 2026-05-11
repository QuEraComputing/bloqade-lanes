from bloqade.analysis.address.analysis import AddressAnalysis
from bloqade.analysis.address.lattice import Address, AddressQubit
from kirin import interp
from kirin.analysis import ForwardFrame

from bloqade.gemini.common.dialects import qubit


@qubit.dialect.register(key="qubit.address")
class _GeminiCommonAddressMethods(interp.MethodTable):
    @interp.impl(qubit.stmts.NewAt)
    def new_at(
        self,
        interp_: AddressAnalysis,
        frame: ForwardFrame[Address],
        stmt: qubit.stmts.NewAt,
    ):
        addr = AddressQubit(interp_.next_address)
        interp_.next_address += 1
        return (addr,)
