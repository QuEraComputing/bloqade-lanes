from kirin import interp

from ._dialect import dialect
from .stmts import Register


@dialect.register
class ConcreteMethods(interp.MethodTable):
    """Registration is compile-time metadata; executing it does nothing."""

    @interp.impl(Register)
    def register(
        self, _interp: interp.Interpreter, frame: interp.Frame, stmt: Register
    ):
        return ()
