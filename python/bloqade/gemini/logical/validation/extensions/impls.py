from bloqade.analysis.address.impls import Func as AddressFuncMethodTable
from kirin.dialects import func


@func.dialect.register(key="gemini.validate.no_extensions")
class Func(AddressFuncMethodTable):
    """Give the analysis the address analysis' ``func`` impls.

    This is what makes the pass a call-graph search rather than a walk of one
    body: without a ``func.Invoke`` impl the interpreter would send the call to
    ``eval_fallback``, which returns bottom without descending, and a use inside
    an un-inlined callee would go unreported.

    Reusing the address analysis' table rather than writing one is the same
    choice ``gemini.validate.terminal_measurement`` makes next door.
    """
