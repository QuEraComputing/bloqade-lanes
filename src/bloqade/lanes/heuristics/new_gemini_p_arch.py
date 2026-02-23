from functools import cached_property
from itertools import chain, combinations, product, starmap
from typing import cast, List, Sequence
from bloqade.lanes.arch.gemini import impls
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes import layout
from bloqade.lanes.utils import no_none_elements_tuple
from numpy import random, around
from bloqade.lanes.analysis.placement import ConcreteState
from bloqade.cirq_utils.lineprog import LPProblem, Variable, Expression
from dataclasses import dataclass, field
import rustworkx as rx
import numpy as np


@dataclass(frozen=True)
class MoveOp:
    atom: int
    src: layout.LocationAddress
    dst: layout.LocationAddress


@dataclass(frozen=True)
class PathSegment:
    lane: layout.LaneAddress
    src: layout.LocationAddress
    dst: layout.LocationAddress
    var: Variable = field(init=False, default_factory=Variable)




@dataclass(frozen=True)
class Path:
    segments: tuple[PathSegment, ...]

    @classmethod
    def from_path_finder(
        cls,
        locations: Sequence[layout.LocationAddress],
        lanes: Sequence[layout.LaneAddress],
    ):
        return cls(
            tuple(
                PathSegment(lane, src, dst)
                for lane, src, dst in zip(lanes, locations, locations[1:])
            )
        )

    @cached_property
    def src(self):
        return self.segments[0].src

    @cached_property
    def dst(self):
        return self.segments[-1].dst

    @cached_property
    def intermediate_locations(self):
        return tuple(seg.src for seg in self.segments[1:])

    def __iter__(self):
        return iter(self.segments)


arch_spec = impls.generate_arch_linear(num_words=2)

path_finder = PathFinder(arch_spec)

random.seed(0)
pair_indices = random.choice(
    len(path_finder.physical_addresses), size=(2, 2), replace=False
)

pairs = [
    (path_finder.physical_addresses[i], path_finder.physical_addresses[j])
    for i, j in pair_indices
]

# pairs = [
#     (layout.LocationAddress(i, j), layout.LocationAddress(i+8, k+5))
#     for i in range(7)
#     for (j,k) in [(0, 0), (1, 4), (2, 1)]
# ]

# pairs = [
#     (layout.LocationAddress(0, 3), layout.LocationAddress(4, 3)),
#     (layout.LocationAddress(0, 4), layout.LocationAddress(5, 3)),
#     (layout.LocationAddress(0, 1), layout.LocationAddress(2, 3)),
#     (layout.LocationAddress(0, 0), layout.LocationAddress(1, 3)),
#     (layout.LocationAddress(0, 2), layout.LocationAddress(3, 3)),
# ]

srcs = set(src for src, _ in pairs)
print("moves:", *("\n  " + str(s) + " -> " + str(d) for s, d in pairs))
dsts = set(dst for _, dst in pairs)


assert srcs.isdisjoint(dsts), "cyclic moves found"


def get_path(
    src: layout.LocationAddress,
    dst: layout.LocationAddress,
    occupied: frozenset[layout.LocationAddress] = frozenset(),
    path_heuristic=lambda _, __: 0.0,
) -> Path | None:
    path_results = path_finder.find_path(
        src, dst, occupied=occupied, path_heuristic=path_heuristic
    )

    if path_results is None:
        return None

    locations, lanes = path_results
    return Path(tuple(map(PathSegment, lanes, locations, locations[1:])))


shortest_paths = tuple(starmap(get_path, pairs))

if not no_none_elements_tuple(shortest_paths):
    exit()

shortest_paths = list(sorted(shortest_paths, key=lambda p: len(p.segments)))

occupied = srcs

all_paths: list[Path] = []


def path_heuristic(
    path: tuple[layout.LocationAddress, ...], lanes: tuple[layout.LaneAddress, ...]
) -> float:

    def compat_lanes(lane1: layout.LaneAddress, lane2: layout.LaneAddress):
        return int(
            lane1.bus_id == lane2.bus_id
            and lane1.direction == lane2.direction
            and lane1.move_type == lane2.move_type
            and lane1.src_site() != lane2.src_site()
        )

    diff = 0.0
    for other_path in all_paths:
        other_lanes = tuple(seg.lane for seg in other_path)

        diff += sum(map(compat_lanes, lanes, other_lanes))

    return diff


while shortest_paths:
    path = shortest_paths.pop(0)

    occupied.remove(path.src)
    path = get_path(path.src, path.dst, occupied=frozenset(occupied))

    if path is None:
        raise ValueError("No feasible path found")

    occupied.add(path.dst)
    all_paths.append(path)


all_paths.sort(key=lambda p: len(p.segments))

lp_problem = LPProblem()

constraints = set()
for path in all_paths:
    for seg1, seg2 in zip(path.segments, path.segments[1:]):
        lp_problem.add_gez(seg2.var - seg1.var - 1)
        constraints.add(frozenset((seg1, seg2)))


path_seg_list = list(
    (path, seg, i) for path in all_paths for i, seg in enumerate(path.segments)
)


for (path_i, seg_i, i), (path_j, seg_j, j) in combinations(path_seg_list, 2):
    if seg_i.dst == seg_j.dst:
        if j + 1 < len(path_j.segments):
            lp_problem.add_gez(seg_i.var - path_j.segments[j + 1].var - 1)
        elif i - 1 >= 0:
            lp_problem.add_gez(path_i.segments[i - 1].var - seg_j.var - 1)

    if (
        seg_i.lane.bus_id == seg_j.lane.bus_id
        and seg_i.lane.direction == seg_j.lane.direction
        and seg_i.lane.move_type == seg_j.lane.move_type
    ):  # try to group compatible lanes together
        lp_problem.add_abs(seg_i.var - seg_j.var)


cost = 0
for seg in chain.from_iterable(all_paths):
    cost = cost + 0.1 * seg.var
    lp_problem.add_gez(seg.var - 0)


lp_problem.add_linear(cast(Expression, cost))
sol = lp_problem.solve()
groups = {}
for seg in chain.from_iterable(all_paths):
    groups.setdefault(int(around(sol[seg.var], 4)), []).append(seg)


for group in sorted(groups.keys()):
    segs = groups[group]
    print(f"Group {group}:")
    for seg in segs:
        print(f" {seg.lane}: {seg.src} -> {seg.dst}")


@dataclass
class MoveSolver:
    arch_spec: layout.ArchSpec
    concrete_state: ConcreteState
    path_finder: PathFinder = field(init=False)

    def __post_init__(self):
        self.path_finder = PathFinder(self.arch_spec)

    def dist(self, move_op: MoveOp) -> float:
        src_pos = np.array(self.arch_spec.get_position(move_op.src))
        dst_pos = np.array(self.arch_spec.get_position(move_op.dst))
        return float(np.linalg.norm(dst_pos - src_pos))


    @property
    def unoccupied_locations(self) -> frozenset[layout.LocationAddress]:
        unoccupied = set(self.arch_spec.yield_all_locations())
        unoccupied.difference_update(self.concrete_state.layout)
        unoccupied.difference_update(self.concrete_state.occupied)
        return frozenset(unoccupied)
    
    @property
    def occupied_locations(self) -> frozenset[layout.LocationAddress]:
        occupied = set(self.concrete_state.occupied)
        occupied.update(self.concrete_state.layout)
        return frozenset(occupied)

    def unemcumbered_path_len(self, move_op: MoveOp) -> float:
        result = self.path_finder.find_path(
            move_op.src,
            move_op.dst,
        )
        assert (
            result is not None
        ), "Every site must be reachable in an unoccupied architecture"

        _, lanes = result
        return len(lanes)

    def generate_paths(self, new_layout: tuple[layout.LocationAddress, ...]) -> dict[int, Path]:


        # NOTE: one of the problems we run into here is that even if we remove swap moves,
        # we can still end up in deadlock situations because the temporary locations we
        # need to move to could block another atom's path. To try to mitigate this problem,
        # we first try to find a path for all atoms and skip the ones that fail. 
        # 
        # After that we can try to resolve deadlocks by building a dependency graph of the
        # remaining moves and finding cycles. We should only break one cycle at a time to 
        # prevent further deadlocks. 
        
        current_layout = tuple(self.concrete_state.layout)
        next_layout = list(current_layout)
        new_layout = tuple(new_layout)
        
        
        move_ops = [
            MoveOp(
                atom=i,
                src=current_layout[i],
                dst=new_layout[i],
            )
            for i in range(len(new_layout))
            if current_layout[i] != new_layout[i]
        ]

        # start with longest paths first
        move_ops = sorted(move_ops, key=self.unemcumbered_path_len, reverse=True)
        occupied = set(self.concrete_state.occupied.union(current_layout))
        
        paths: dict[int, Path] = {}
        while move_ops:
            new_paths: dict[int, Path] = {}
            for move_index, move_op in enumerate(move_ops):
                occupied.remove(move_op.src)
                path = self.path_finder.find_path(
                    move_op.src, move_op.dst, occupied=frozenset(occupied)
                )

                if path is None:
                    # move next atom and try again later
                    occupied.add(move_op.src)
                else:
                    occupied.add(move_op.dst)
                    new_paths[move_index] = Path.from_path_finder(*path)


            for move_index in new_paths.keys():
                move_op = move_ops.pop(move_index)
                paths[move_op.atom] = new_paths[move_index]
                next_layout[move_op.atom] = move_op.dst

            if len(new_paths) == 0 and len(move_ops) > 0:
                break

        # find a path to another location to break and deadlocks.
        move_graph = rx.PyDiGraph()
        moveop_map = {move_op: move_graph.add_node(move_op) for move_op in move_ops}


        edge_maps = {}
        for (move_op1, node1), (move_op2, node2) in product(moveop_map.items(), repeat=2):
            if node1 == node2:
                continue
            
            path1_with_src2 = self.path_finder.find_path(
                move_op1.src,move_op2.src, occupied=frozenset(occupied)
            )
            path1_without_src2 = self.path_finder.find_path(
                move_op1.src,move_op2.src, occupied=frozenset(occupied | {move_op2.src})
            )
            
            if path1_with_src2 is None and path1_without_src2 is not None:
                # this indicates that move_op2.src is blocking move_op1's path
                move_graph.add_edge(node1, node2, None)
                
                
        # find the first cycle and break it
        cycles = rx.digraph_find_cycle(move_graph)
        
        if cycles is None:
            raise ValueError("No feasible paths found for all moves")
        
        # find node with highest predecessor count
        cycle_nodes = list(set(node for pair in cycles for node in pair))
        cycle_nodes.sort(key=move_graph.predecessors, reverse=True)

        # break cycle by attempting to find a temporary location for one of the nodes in the cycle
        # starting with the node that is blocking the most other nodes
        cycle_broken = False
        for cycle_node in cycle_nodes:
            move_op = move_ops[cycle_node]
            
            # find all possible temporary locations
            # exclude any locations used in the preducessors' paths
            
            predecessors: list[MoveOp] = move_graph.predecessors(cycle_node)
            
            blocked_locations: set[layout.LocationAddress] = set()
            tmp_occupied = frozenset(occupied | {move_op.src})
            for pred in predecessors:
                path = self.path_finder.find_path(
                    pred.src, pred.dst, occupied=tmp_occupied
                )
                assert path is not None, "predecessor path must exist if we remove cycle node"
                locs, _ = path
                blocked_locations.update(locs)
                
            tmp_occupied = frozenset(tmp_occupied.union(blocked_locations))
            tmp_locs =  [loc for loc in self.path_finder.physical_addresses if loc not in blocked_locations]
            tmp_nodes = list(map(self.path_finder.physical_address_map.__getitem__, tmp_locs))
            src_node = tmp_locs.index(move_op.src)
            tmp_subgraph = self.path_finder.site_graph.subgraph(tmp_nodes)
            
            # find all neighbors reachable from move_op.src
            neighbors: list[layout.LocationAddress] = tmp_subgraph.successors(src_node)
            
            if len(neighbors) > 0:            
                # pick the closest neighbor as temporary location
                def dist(loc: layout.LocationAddress) -> float:
                    return self.dist(MoveOp(move_op.atom, move_op.src, loc))
            
                tmp_loc = min(neighbors, key=dist)
                
                tmp_move_op = MoveOp(
                    atom=move_op.atom,
                    src=move_op.src,
                    dst=tmp_loc,
                )
                
                move_ops[moveop_map[move_op]] = tmp_move_op
                cycle_broken = True
                break
            
        if not cycle_broken:
            raise ValueError("Could not break cycle in move graph")
        
        current_layout = tuple(next_layout)
        
