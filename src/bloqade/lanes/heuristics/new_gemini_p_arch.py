from functools import cached_property
from itertools import chain, combinations, starmap
from typing import cast
from bloqade.lanes.arch.gemini import impls
from bloqade.lanes.layout.path import PathFinder
from bloqade.lanes import layout
from bloqade.lanes.utils import no_none_elements_tuple
from numpy import random, around
from bloqade.cirq_utils.lineprog import LPProblem, Variable, Expression
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PathSegment:
    lane: layout.LaneAddress
    src: layout.LocationAddress
    dst: layout.LocationAddress
    var: Variable = field(init=False, default_factory=Variable)


@dataclass(frozen=True)
class Path:
    segments: tuple[PathSegment, ...]

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
    ): # try to group compatible lanes together
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
