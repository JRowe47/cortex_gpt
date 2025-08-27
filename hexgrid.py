from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class Hex:
    q: int; r: int

HEX_DIRS = [Hex(1,0), Hex(1,-1), Hex(0,-1), Hex(-1,0), Hex(-1,1), Hex(0,1)]
def _add(a: Hex, b: Hex) -> Hex: return Hex(a.q+b.q, a.r+b.r)

def neighbors(h: Hex) -> List[Hex]:
    return [_add(h, d) for d in HEX_DIRS]

def make_grid(width: int, height: int) -> List[Hex]:
    return [Hex(q, r) for q in range(width) for r in range(height)]

def build_adjacency(hexes: List[Hex], long_range_per_node: int = 2) -> Dict[int, List[int]]:
    idx = {h: i for i, h in enumerate(hexes)}
    nbrs: Dict[int, List[int]] = {}
    for i, h in enumerate(hexes):
        local = [idx[n] for n in neighbors(h) if n in idx]
        nbrs[i] = local
    # simple ring "rich-club"
    N = len(hexes)
    for i in range(N):
        jumps = [ (i + (j+1) * (N // (long_range_per_node+1))) % N for j in range(long_range_per_node) ]
        nbrs[i].extend(jumps)
    return nbrs
