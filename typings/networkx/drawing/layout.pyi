"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete

def random_layout(G, center: Incomplete | None = ..., dim: int = ..., seed: Incomplete | None = ...):
    ...

def circular_layout(G, scale: float = ..., center: Incomplete | None = ..., dim: int = ...):
    ...

def shell_layout(G, nlist: Incomplete | None = ..., rotate: Incomplete | None = ..., scale: float = ..., center: Incomplete | None = ..., dim: int = ...):
    ...

def bipartite_layout(G, nodes, align: str = ..., scale: float = ..., center: Incomplete | None = ..., aspect_ratio: float = ...):
    ...

def spring_layout(G, k: Incomplete | None = ..., pos: Incomplete | None = ..., fixed: Incomplete | None = ..., iterations: int = ..., threshold: float = ..., weight: str = ..., scale: float = ..., center: Incomplete | None = ..., dim: int = ..., seed: Incomplete | None = ...):
    ...

fruchterman_reingold_layout = ...
def kamada_kawai_layout(G, dist: Incomplete | None = ..., pos: Incomplete | None = ..., weight: str = ..., scale: float = ..., center: Incomplete | None = ..., dim: int = ...):
    ...

def spectral_layout(G, weight: str = ..., scale: float = ..., center: Incomplete | None = ..., dim: int = ...):
    ...

def planar_layout(G, scale: float = ..., center: Incomplete | None = ..., dim: int = ...):
    ...

def spiral_layout(G, scale: float = ..., center: Incomplete | None = ..., dim: int = ..., resolution: float = ..., equidistant: bool = ...):
    ...

def multipartite_layout(G, subset_key: str = ..., align: str = ..., scale: float = ..., center: Incomplete | None = ...):
    ...

def arf_layout(G, pos: Incomplete | None = ..., scaling: float = ..., a: float = ..., etol: float = ..., dt: float = ..., max_iter: int = ...):
    ...

def rescale_layout(pos, scale: float = ...):
    ...

def rescale_layout_dict(pos, scale: float = ...):
    ...

