"""
This type stub file was generated by pyright.
"""

import types
from _typeshed import Incomplete

__all__ = ["attach", "_lazy_import"]
def attach(module_name, submodules: Incomplete | None = ..., submod_attrs: Incomplete | None = ...):
    ...

class DelayedImportErrorModule(types.ModuleType):
    def __init__(self, frame_data, *args, **kwargs) -> None:
        ...
    
    def __getattr__(self, x) -> None:
        ...
    

