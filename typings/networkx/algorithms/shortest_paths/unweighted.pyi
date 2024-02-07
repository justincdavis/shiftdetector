"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete
from collections.abc import Generator

def single_source_shortest_path_length(G, source, cutoff: Incomplete | None = ...):
    ...

def single_target_shortest_path_length(G, target, cutoff: Incomplete | None = ...):
    ...

def all_pairs_shortest_path_length(G, cutoff: Incomplete | None = ...) -> Generator[Incomplete, None, None]:
    ...

def bidirectional_shortest_path(G, source, target):
    ...

def single_source_shortest_path(G, source, cutoff: Incomplete | None = ...):
    ...

def single_target_shortest_path(G, target, cutoff: Incomplete | None = ...):
    ...

def all_pairs_shortest_path(G, cutoff: Incomplete | None = ...) -> Generator[Incomplete, None, None]:
    ...

def predecessor(G, source, target: Incomplete | None = ..., cutoff: Incomplete | None = ..., return_seen: Incomplete | None = ...):
    ...

