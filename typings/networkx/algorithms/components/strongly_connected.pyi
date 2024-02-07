"""
This type stub file was generated by pyright.
"""

from collections.abc import Generator, Hashable, Iterable
from networkx.classes.digraph import DiGraph
from networkx.classes.graph import Graph, _Node

def strongly_connected_components(G: Graph[_Node]) -> Generator[set[_Node], None, None]:
    ...

def kosaraju_strongly_connected_components(G: Graph[_Node], source: _Node | None = ...) -> Generator[set[_Node], None, None]:
    ...

def strongly_connected_components_recursive(G: Graph[_Node]) -> Generator[set[_Node], None, None]:
    ...

def number_strongly_connected_components(G: Graph[Hashable]) -> int:
    ...

def is_strongly_connected(G: Graph[Hashable]) -> bool:
    ...

def condensation(G: DiGraph[_Node], scc: Iterable[Iterable[_Node]] | None = ...) -> DiGraph[int]:
    ...

