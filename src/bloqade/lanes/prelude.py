from kirin import ir
from kirin.ir.method import Method
from kirin.passes.default import Default
from kirin.prelude import basic
from typing_extensions import Annotated, Doc

from .dialects import move


@ir.dialect_group(basic.add(move))
def kernel(self):
    def run_pass(
        mt: Annotated[Method, Doc("The method to run pass on.")],
        *,
        verify: Annotated[
            bool, Doc("run `verify` before running passes, default is `True`")
        ] = True,
        typeinfer: Annotated[
            bool,
            Doc(
                "run type inference and apply the inferred type to IR, default `False`"
            ),
        ] = False,
        fold: Annotated[bool, Doc("run folding passes")] = True,
        aggressive: Annotated[
            bool, Doc("run aggressive folding passes if `fold=True`")
        ] = False,
        no_raise: Annotated[bool, Doc("do not raise exception during analysis")] = True,
    ) -> None:
        default_pass = Default(
            self,
            verify=verify,
            fold=fold,
            aggressive=aggressive,
            typeinfer=typeinfer,
            no_raise=no_raise,
        )
        default_pass.fixpoint(mt)

    return run_pass
