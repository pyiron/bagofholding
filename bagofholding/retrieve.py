from importlib import import_module
from typing import Any


def import_from_string(library_path: str) -> Any:
    split_path = library_path.split(".", 1)
    if len(split_path) == 1:
        module_name, path = split_path[0], ""
    else:
        module_name, path = split_path
    obj = import_module(module_name)
    for k in path.split("."):
        obj = getattr(obj, k)
    return obj
