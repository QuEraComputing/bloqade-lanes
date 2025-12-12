from dataclasses import dataclass, field

from kirin import ir
from kirin.dialects import func, ilist
from kirin.rewrite import abc as rewrite_abc

from bloqade import qubit
from bloqade.lanes.dialects import move
from bloqade.lanes.layout.encoding import LocationAddress


@dataclass
class InsertInitialize(rewrite_abc.RewriteRule):
    """Note that this rewrite assumes that the Initialize statements
    have location addresses that map to logical qubits only not physical.

    """

    logical_reg_maps: dict[LocationAddress, ir.SSAValue]
    initialize_circuit: ir.Method

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, move.Initialize):
            return rewrite_abc.RewriteResult()

        for loc_addr, theta, phi, lam in zip(
            node.location_addresses,
            node.thetas,
            node.phis,
            node.lams,
        ):
            if loc_addr not in self.logical_reg_maps:
                continue

            logical_reg = self.logical_reg_maps[loc_addr]
            node.replace_by(
                func.Invoke(
                    (theta, phi, lam, logical_reg), callee=self.initialize_circuit
                )
            )

        return rewrite_abc.RewriteResult(has_done_something=True)


@dataclass
class InsertQubits(rewrite_abc.RewriteRule):
    logical_to_physical: dict[LocationAddress, tuple[LocationAddress, ...]]
    physical_reg_maps: dict[LocationAddress, ir.SSAValue] = field(
        default_factory=dict, init=False
    )
    logical_reg_maps: dict[LocationAddress, ir.SSAValue] = field(
        default_factory=dict, init=False
    )

    def __post_init__(self):
        self.physical_to_logical = {
            phys_addr: log_addr
            for log_addr, phys_addrs in self.logical_to_physical.items()
            for phys_addr in phys_addrs
        }

    def rewrite_Statement(self, node: ir.Statement) -> rewrite_abc.RewriteResult:
        if not isinstance(node, move.Fill):
            return rewrite_abc.RewriteResult()

        for loc_addr in node.location_addresses:
            assert (
                loc_addr not in self.physical_reg_maps
            ), f"Physical address {loc_addr} already has a mapping."

            (new_qubit := qubit.stmts.New()).insert_before(node)
            self.physical_reg_maps[loc_addr] = new_qubit.result

        for logical_addr, physical_addrs in self.logical_to_physical.items():
            physical_ssas = tuple(
                self.physical_reg_maps[phys_addr] for phys_addr in physical_addrs
            )
            (logical_reg := ilist.New(physical_ssas)).insert_before(node)
            self.logical_reg_maps[logical_addr] = logical_reg.result

        return rewrite_abc.RewriteResult(has_done_something=True)
