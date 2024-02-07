"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete

def draw(G, pos: Incomplete | None = ..., ax: Incomplete | None = ..., **kwds) -> None:
    ...

def draw_networkx(G, pos: Incomplete | None = ..., arrows: Incomplete | None = ..., with_labels: bool = ..., **kwds) -> None:
    ...

def draw_networkx_nodes(G, pos, nodelist: Incomplete | None = ..., node_size: Incomplete | int = ..., node_color: str = ..., node_shape: str = ..., alpha: Incomplete | None = ..., cmap: Incomplete | None = ..., vmin: Incomplete | None = ..., vmax: Incomplete | None = ..., ax: Incomplete | None = ..., linewidths: Incomplete | None = ..., edgecolors: Incomplete | None = ..., label: Incomplete | None = ..., margins: Incomplete | None = ...):
    ...

def draw_networkx_edges(G, pos, edgelist: Incomplete | None = ..., width: float = ..., edge_color: str = ..., style: str = ..., alpha: Incomplete | None = ..., arrowstyle: Incomplete | None = ..., arrowsize: int = ..., edge_cmap: Incomplete | None = ..., edge_vmin: Incomplete | None = ..., edge_vmax: Incomplete | None = ..., ax: Incomplete | None = ..., arrows: Incomplete | None = ..., label: Incomplete | None = ..., node_size: Incomplete | int = ..., nodelist: Incomplete | None = ..., node_shape: str = ..., connectionstyle: str = ..., min_source_margin: int = ..., min_target_margin: int = ...):
    ...

def draw_networkx_labels(G, pos, labels: Incomplete | None = ..., font_size: int = ..., font_color: str = ..., font_family: str = ..., font_weight: str = ..., alpha: Incomplete | None = ..., bbox: Incomplete | None = ..., horizontalalignment: str = ..., verticalalignment: str = ..., ax: Incomplete | None = ..., clip_on: bool = ...):
    ...

def draw_networkx_edge_labels(G, pos, edge_labels: Incomplete | None = ..., label_pos: float = ..., font_size: int = ..., font_color: str = ..., font_family: str = ..., font_weight: str = ..., alpha: Incomplete | None = ..., bbox: Incomplete | None = ..., horizontalalignment: str = ..., verticalalignment: str = ..., ax: Incomplete | None = ..., rotate: bool = ..., clip_on: bool = ...):
    ...

def draw_circular(G, **kwargs) -> None:
    ...

def draw_kamada_kawai(G, **kwargs) -> None:
    ...

def draw_random(G, **kwargs) -> None:
    ...

def draw_spectral(G, **kwargs) -> None:
    ...

def draw_spring(G, **kwargs) -> None:
    ...

def draw_shell(G, nlist: Incomplete | None = ..., **kwargs) -> None:
    ...

def draw_planar(G, **kwargs) -> None:
    ...

