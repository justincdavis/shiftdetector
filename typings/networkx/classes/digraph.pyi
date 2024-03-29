"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete
from collections.abc import Iterator
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import Graph, _Node
from networkx.classes.reportviews import InDegreeView, InEdgeView, InMultiDegreeView, OutDegreeView, OutEdgeView, OutMultiDegreeView

class DiGraph(Graph[_Node]):
    succ: AdjacencyView[_Node, _Node, dict[str, Incomplete]]
    pred: AdjacencyView[_Node, _Node, dict[str, Incomplete]]
    def has_successor(self, u: _Node, v: _Node) -> bool:
        ...
    
    def has_predecessor(self, u: _Node, v: _Node) -> bool:
        ...
    
    def successors(self, n: _Node) -> Iterator[_Node]:
        ...
    
    def predecessors(self, n: _Node) -> Iterator[_Node]:
        ...
    
    in_edges: InEdgeView[_Node]
    in_degree: InDegreeView[_Node] | InMultiDegreeView[_Node]
    out_edges: OutEdgeView[_Node]
    out_degree: OutDegreeView[_Node] | OutMultiDegreeView[_Node]
    def to_undirected(self, reciprocal: bool = ..., as_view: bool = ...):
        ...
    
    def reverse(self, copy: bool = ...) -> DiGraph[_Node]:
        ...
    
    def copy(self, as_view: bool = ...) -> DiGraph[_Node]:
        ...
    


