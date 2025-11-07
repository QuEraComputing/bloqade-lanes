import abc

from bloqade.squin import qubit
from bloqade.squin.op import stmts as op_stmts, types as op_types
from kirin import ir, types
from kirin.dialects import func, ilist
from kirin.rewrite.abc import RewriteResult, RewriteRule

from bloqade import native


class SplitMult(RewriteRule):
    """Split apply(a * b) into apply(a); apply(b)"""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, (qubit.Apply, qubit.Broadcast)) or not isinstance(
            op_stmt := node.operator.owner, op_stmts.Mult
        ):
            return RewriteResult()

        node_type = type(node)
        node_type(operator=op_stmt.lhs, qubits=node.qubits).insert_before(node)
        node_type(operator=op_stmt.rhs, qubits=node.qubits).replace_by(node)
        return RewriteResult(has_done_something=True)


class SplitKron(RewriteRule):
    """Split apply(a ⊗ b) into apply(a) on first n qubits, apply(b) on last m qubits
    depending on the number of qubits each operator acts on."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if (
            not isinstance(node, (qubit.Apply, qubit.Broadcast))
            or not isinstance(op_stmt := node.operator.owner, op_stmts.Kron)
            or not isinstance(lhs_stmt := op_stmt.lhs.owner, ir.Statement)
            or not isinstance(rhs_stmt := op_stmt.rhs.owner, ir.Statement)
        ):
            return RewriteResult()

        if (trait := lhs_stmt.get_trait(op_stmts.HasSites)) is not None:
            lhs_sites = trait.get_sites(lhs_stmt)
            lhs_qubits = node.qubits[:lhs_sites]
            rhs_qubits = node.qubits[lhs_sites:]
        elif (trait := rhs_stmt.get_trait(op_stmts.HasSites)) is not None:
            rhs_sites = trait.get_sites(rhs_stmt)
            lhs_qubits = node.qubits[:-rhs_sites]
            rhs_qubits = node.qubits[-rhs_sites:]
        else:
            return RewriteResult()

        node_type = type(node)
        node_type(operator=op_stmt.lhs, qubits=lhs_qubits).insert_before(node)
        node_type(operator=op_stmt.rhs, qubits=rhs_qubits).replace_by(node)
        return RewriteResult(has_done_something=True)


class RemoveId(RewriteRule):
    """Remove apply(I) statements."""

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, (qubit.Apply, qubit.Broadcast)) or not isinstance(
            node.operator.owner, op_stmts.Identity
        ):
            return RewriteResult()

        node.delete()
        return RewriteResult(has_done_something=True)


class GateInfoABC:

    @classmethod
    @abc.abstractmethod
    def stdlib_func(cls) -> ir.Method: ...

    @abc.abstractmethod
    def operator_matches(self, type: types.TypeAttribute) -> bool: ...

    @abc.abstractmethod
    def get_params(self, op_stmt: ir.Statement) -> tuple[ir.SSAValue, ...] | None: ...


class XInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.x

    def get_params(self, op_stmt: ir.Statement):
        return ()

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.XOpType)


class YInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.y

    def get_params(self, op_stmt: ir.Statement):
        return ()

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.YOpType)


class ZInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.z

    def get_params(self, op_stmt: ir.Statement):
        return ()

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ZOpType)


class CXInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.cx

    def get_params(self, op_stmt: ir.Statement):
        return ()

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ControlOpType[op_types.XOpType])


class CYInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.cy

    def get_params(self, op_stmt: ir.Statement):
        return ()

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ControlOpType[op_types.YOpType])


class CZInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.cz

    def get_params(self, op_stmt: ir.Statement):
        return ()

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ControlOpType[op_types.ZOpType])


class RxInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.rx

    def get_params(self, op_stmt: ir.Statement):
        if not isinstance(op_stmt, op_stmts.Rot):
            return None

        return (op_stmt.angle,)

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ROpType[op_types.XOpType])


class RyInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.ry

    def get_params(self, op_stmt: ir.Statement):
        if not isinstance(op_stmt, op_stmts.Rot):
            return None

        return (op_stmt.angle,)

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ROpType[op_types.YOpType])


class RzInfo(GateInfoABC):

    @classmethod
    def stdlib_func(cls) -> ir.Method:
        return native.rz

    def get_params(self, op_stmt: ir.Statement):
        if not isinstance(op_stmt, op_stmts.Rot):
            return None

        return (op_stmt.angle,)

    def operator_matches(self, type: types.TypeAttribute) -> bool:
        return type.is_subseteq(op_types.ROpType[op_types.ZOpType])


class RewriteApply(RewriteRule):

    GATES = (
        RxInfo(),
        RyInfo(),
        RzInfo(),
        CXInfo(),
        CYInfo(),
        CZInfo(),
        XInfo(),
        YInfo(),
        ZInfo(),
    )

    def get_qubit_args(
        self, node: qubit.Apply | qubit.Broadcast
    ) -> tuple[ir.SSAValue, ...]:
        if isinstance(node, qubit.Apply):
            qubit_args: list[ir.SSAValue] = []
            for arg in node.args:
                (new_qubit_list := ilist.New((arg,))).insert_before(node)
                qubit_args.append(new_qubit_list.result)

            return tuple(qubit_args)
        else:
            return node.qubits

    def rewrite_Statement(self, node: ir.Statement) -> RewriteResult:
        if not isinstance(node, (qubit.Apply, qubit.Broadcast)) or not isinstance(
            op_stmt := node.operator.owner, ir.Statement
        ):
            return RewriteResult(has_done_something=False)

        for gate in self.GATES:
            if not gate.operator_matches(node.operator.type):
                continue

            params = gate.get_params(op_stmt)

            if params is None:
                continue

            inputs = params + self.get_qubit_args(node)
            node.replace_by(func.Invoke(inputs, callee=gate.stdlib_func(), kwargs=()))
            return RewriteResult(has_done_something=True)

        return RewriteResult()
