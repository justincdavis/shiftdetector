"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete
from collections.abc import Hashable, Mapping
from typing import Literal, TypeVar, overload
from networkx.classes.digraph import DiGraph
from networkx.classes.graph import Graph
from networkx.classes.multidigraph import MultiDiGraph
from networkx.classes.multigraph import MultiGraph

_X = TypeVar("_X", bound=Hashable)
_Y = TypeVar("_Y", bound=Hashable)
@overload
def relabel_nodes(G: MultiDiGraph[_X], mapping: Mapping[_X, _Y], copy: bool = ...) -> MultiDiGraph[_X | _Y]:
    ...

@overload
def relabel_nodes(G: DiGraph[_X], mapping: Mapping[_X, _Y], copy: bool = ...) -> DiGraph[_X | _Y]:
    ...

@overload
def relabel_nodes(G: MultiGraph[_X], mapping: Mapping[_X, _Y], copy: bool = ...) -> MultiGraph[_X | _Y]:
    ...

@overload
def relabel_nodes(G: Graph[_X], mapping: Mapping[_X, _Y], copy: bool = ...) -> Graph[_X | _Y]:
    ...

def convert_node_labels_to_integers(G: Graph[Hashable], first_label: int = ..., ordering: Literal["default", "sorted", "increasing degree", "decreasing degree"] = ..., label_attribute: Incomplete | None = ...) -> Graph[int]:
    ...
