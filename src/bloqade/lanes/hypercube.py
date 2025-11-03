from itertools import chain
from .types import ArchSpec, Block, InterLane, SiteType, IntraLane, Grid
from typing import TypeVar



def intra_hypercube(dim: int) -> tuple[IntraLane, ...]:
    num_sites = 2 ** dim
    intra_lanes = []
    for i in range(num_sites):
        for d in range(dim):
            if ((i >> d) & 1) == 0:
                neighbor = i | (1 << d)
                lane = IntraLane(site_1=i, site_2=neighbor)
                intra_lanes.append(lane)
    return tuple(intra_lanes)


def partition_grid(sites: Grid, num_x: int, num_y: int):
    if sites.shape[0] % num_x != 0:
        raise ValueError("Number of x sites must be divisible by num_x")
    if sites.shape[1] % num_y != 0:
        raise ValueError("Number of y sites must be divisible by num_y")

    block_size_x = sites.shape[0] // num_x
    block_size_y = sites.shape[1] // num_y

    blocks: list[list[Block[tuple[float, float]]]] = []
    for by in range(num_y):
        row_blocks = []
        for bx in range(num_x):
            x_start = bx * block_size_x
            x_end = x_start + block_size_x
            y_start = by * block_size_y
            y_end = y_start + block_size_y
            row_blocks.append(Block(sites=sites[x_start:x_end, y_start:y_end].positions))
        blocks.append(row_blocks)

    return blocks


Nx = TypeVar("Nx")
Ny = TypeVar("Ny")

def holobyte_geometry(compute_sites: Grid[Nx, Ny], cache_sites: Grid[Nx, Ny], memory_sites: Grid[Nx, Ny]):
    num_compute_bytes_x = compute_sites.shape[0] // 16
    num_compute_bytes_y = compute_sites.shape[1] 

    num_mem_bytes_x = memory_sites.shape[0] // 16
    num_mem_bytes_y = memory_sites.shape[1]

    if compute_sites.shape != cache_sites.shape:
        raise ValueError("Compute sites and cache sites must have the same shape")
    
    if num_compute_bytes_x != num_mem_bytes_x:
        raise ValueError("Compute sites and memory sites must have the same number of holobytes in x dimension")
    
    if num_mem_bytes_y % num_compute_bytes_y != 0:
        raise ValueError("Memory sites must have an integer multiple of holobytes in y dimension compared to compute sites")


    mem_to_cache_ratio = num_mem_bytes_y // num_compute_bytes_y

    compute_sites = compute_sites[:num_compute_bytes_x*16,:]
    cache_sites = cache_sites[:num_compute_bytes_x*16,:]
    memory_sites = memory_sites[:num_mem_bytes_x*16,:]

    # intra-holobyte lanes for compute_blocks only
    intra_lanes = intra_hypercube(4)

    compute_blocks = partition_grid(compute_sites, 16, 1)
    cache_blocks = partition_grid(cache_sites, 16, 1)
    memory_blocks = partition_grid(memory_sites, 16, 1)

    blocks = tuple(chain.from_iterable(compute_blocks + cache_blocks + memory_blocks))
    has_intra_blocks = frozenset(chain.from_iterable(compute_blocks))

    inter_lanes = []
    for mem_y_id, mem_row in enumerate(memory_blocks):
        cache_y_id = mem_y_id % num_compute_bytes_y # linearly map memory rows to cache rows
        for mem_x_id, mem_block in enumerate(mem_row):
            cache_block = cache_blocks[cache_y_id][mem_x_id]
            for site_id in range(16):
                inter_lane = InterLane(
                    block_1=mem_block,
                    block_2=cache_block,
                    site=site_id,
                )
                inter_lanes.append(inter_lane)

    return ArchSpec(
        blocks=tuple(blocks),
        intra_lanes=intra_lanes,
        allowed_inter_lanes=tuple(inter_lanes),
        has_intra_lanes=has_intra_blocks,
    )

    












                
