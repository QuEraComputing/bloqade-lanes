from kirin import ir, rewrite

from .rewrite import RewriteMoves
from .stmts import dialect


class SpecializeGemini:

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        rewrite.Walk(RewriteMoves()).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
