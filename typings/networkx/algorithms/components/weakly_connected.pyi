"""
This type stub file was generated by pyright.
"""

from collections.abc import Generator, Hashable
from networkx.classes.graph import Graph, _Node

def weakly_connected_components(G: Graph[_Node]) -> Generator[set[_Node], None, None]:
    ...

def number_weakly_connected_components(G: Graph[Hashable]) -> int:
    ...

def is_weakly_connected(G: Graph[Hashable]) -> bool:
    ...
