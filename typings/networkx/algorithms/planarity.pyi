"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete
from collections.abc import Generator, Mapping, MutableSet, Reversible
from networkx.classes.digraph import DiGraph
from networkx.classes.graph import _Node

__all__ = ["check_planarity", "is_planar", "PlanarEmbedding"]
def is_planar(G) -> bool:
    ...

def check_planarity(G, counterexample: bool = ...):
    ...

class Interval:
    low: Incomplete
    high: Incomplete
    def __init__(self, low: Incomplete | None = ..., high: Incomplete | None = ...) -> None:
        ...
    
    def empty(self):
        ...
    
    def copy(self):
        ...
    
    def conflicting(self, b, planarity_state):
        ...
    


class ConflictPair:
    left: Incomplete
    right: Incomplete
    def __init__(self, left: Incomplete = ..., right: Incomplete = ...) -> None:
        ...
    
    def swap(self) -> None:
        ...
    
    def lowest(self, planarity_state):
        ...
    


class LRPlanarity:
    G: Incomplete
    roots: Incomplete
    height: Incomplete
    lowpt: Incomplete
    lowpt2: Incomplete
    nesting_depth: Incomplete
    parent_edge: Incomplete
    DG: Incomplete
    adjs: Incomplete
    ordered_adjs: Incomplete
    ref: Incomplete
    side: Incomplete
    S: Incomplete
    stack_bottom: Incomplete
    lowpt_edge: Incomplete
    left_ref: Incomplete
    right_ref: Incomplete
    embedding: Incomplete
    def __init__(self, G) -> None:
        ...
    
    def lr_planarity(self):
        ...
    
    def lr_planarity_recursive(self):
        ...
    
    def dfs_orientation(self, v):
        ...
    
    def dfs_orientation_recursive(self, v) -> None:
        ...
    
    def dfs_testing(self, v):
        ...
    
    def dfs_testing_recursive(self, v):
        ...
    
    def add_constraints(self, ei, e):
        ...
    
    def remove_back_edges(self, e) -> None:
        ...
    
    def dfs_embedding(self, v):
        ...
    
    def dfs_embedding_recursive(self, v) -> None:
        ...
    
    def sign(self, e):
        ...
    
    def sign_recursive(self, e):
        ...
    


class PlanarEmbedding(DiGraph[_Node]):
    def get_data(self) -> dict[_Node, list[_Node]]:
        ...
    
    def set_data(self, data: Mapping[_Node, Reversible[_Node]]) -> None:
        ...
    
    def neighbors_cw_order(self, v: _Node) -> Generator[_Node, None, None]:
        ...
    
    def check_structure(self) -> None:
        ...
    
    def add_half_edge_ccw(self, start_node: _Node, end_node: _Node, reference_neighbor: _Node) -> None:
        ...
    
    def add_half_edge_cw(self, start_node: _Node, end_node: _Node, reference_neighbor: _Node) -> None:
        ...
    
    def connect_components(self, v: _Node, w: _Node) -> None:
        ...
    
    def add_half_edge_first(self, start_node: _Node, end_node: _Node) -> None:
        ...
    
    def next_face_half_edge(self, v: _Node, w: _Node) -> tuple[_Node, _Node]:
        ...
    
    def traverse_face(self, v: _Node, w: _Node, mark_half_edges: MutableSet[tuple[_Node, _Node]] | None = ...) -> list[_Node]:
        ...
    


