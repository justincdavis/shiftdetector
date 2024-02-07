"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete
from typing_extensions import TypeAlias
from networkx.classes.graph import Graph, _Node
from networkx.classes.multidigraph import MultiDiGraph

_MultiEdge: TypeAlias = ...
class MultiGraph(Graph[_Node]):
    def __init__(self, incoming_graph_data: Incomplete | None = ..., multigraph_input: bool | None = ..., **attr) -> None:
        ...
    
    def new_edge_key(self, u: _Node, v: _Node) -> int:
        ...
    
    def add_edge(self, u_for_edge, v_for_edge, key: Incomplete | None = ..., **attr):
        ...
    
    def remove_edge(self, u, v, key: Incomplete | None = ...):
        ...
    
    def has_edge(self, u, v, key: Incomplete | None = ...):
        ...
    
    def get_edge_data(self, u, v, key: Incomplete | None = ..., default: Incomplete | None = ...):
        ...
    
    def copy(self, as_view: bool = ...) -> MultiGraph[_Node]:
        ...
    
    def to_directed(self, as_view: bool = ...) -> MultiDiGraph[_Node]:
        ...
    
    def to_undirected(self, as_view: bool = ...) -> MultiGraph[_Node]:
        ...
    
    def number_of_edges(self, u: _Node | None = ..., v: _Node | None = ...) -> int:
        ...
    


