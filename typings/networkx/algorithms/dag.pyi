"""
This type stub file was generated by pyright.
"""

from _typeshed import SupportsRichComparison
from collections.abc import Callable, Generator, Iterable, Reversible
from networkx.classes.graph import Graph, _Node

def descendants(G: Graph[_Node], source: _Node) -> set[_Node]:
    ...

def ancestors(G: Graph[_Node], source: _Node) -> set[_Node]:
    ...

def is_directed_acyclic_graph(G: Graph[_Node]) -> bool:
    ...

def topological_sort(G: Graph[_Node]) -> Generator[_Node, None, None]:
    ...

def lexicographical_topological_sort(G: Graph[_Node], key: Callable[[_Node], SupportsRichComparison] | None = ...) -> Generator[_Node, None, None]:
    ...

def all_topological_sorts(G: Graph[_Node]) -> Generator[list[_Node], None, None]:
    ...

def is_aperiodic(G: Graph[_Node]) -> bool:
    ...

def transitive_closure(G: Graph[_Node], reflexive: bool = ...) -> Graph[_Node]:
    ...

def transitive_reduction(G: Graph[_Node]) -> Graph[_Node]:
    ...

def antichains(G: Graph[_Node], topo_order: Reversible[_Node] | None = ...) -> Generator[list[_Node], None, None]:
    ...

def dag_longest_path(G: Graph[_Node], weight: str = ..., default_weight: int = ..., topo_order: Iterable[_Node] | None = ...) -> list[_Node]:
    ...

def dag_longest_path_length(G: Graph[_Node], weight: str = ..., default_weight: int = ...) -> int:
    ...

def dag_to_branching(G: Graph[_Node]) -> Graph[_Node]:
    ...

