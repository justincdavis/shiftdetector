"""
This type stub file was generated by pyright.
"""

from _typeshed import Incomplete
from typing import NamedTuple

def not_implemented_for(*graph_types):
    ...

def open_file(path_arg, mode: str = ...):
    ...

def nodes_or_number(which_args):
    ...

def np_random_state(random_state_argument):
    ...

def py_random_state(random_state_argument):
    ...

class argmap:
    def __init__(self, func, *args, try_finally: bool = ...) -> None:
        ...
    
    def __call__(self, f):
        ...
    
    def compile(self, f):
        ...
    
    def assemble(self, f):
        ...
    
    @classmethod
    def signature(cls, f):
        ...
    
    class Signature(NamedTuple):
        name: Incomplete
        signature: Incomplete
        def_sig: Incomplete
        call_sig: Incomplete
        names: Incomplete
        n_positional: Incomplete
        args: Incomplete
        kwargs: Incomplete
        ...
    
    


