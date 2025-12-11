from kirin import ir, rewrite

from .rewrite import RewriteFill, RewriteInitialize, RewriteMoves
from .stmts import dialect


class SpecializeGemini:

    def emit(self, mt: ir.Method, no_raise=True) -> ir.Method:
        out = mt.similar(dialects=mt.dialects.add(dialect))

        rewrite.Walk(
            rewrite.Chain(RewriteMoves(), RewriteFill(), RewriteInitialize())
        ).rewrite(out.code)

        if not no_raise:
            out.verify()

        return out
