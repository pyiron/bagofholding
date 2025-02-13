from __future__ import annotations

from collections.abc import ItemsView
from dataclasses import asdict, dataclass
from importlib import import_module
from sys import version_info
from typing import Any


@dataclass
class Metadata:
    module: str | None = None
    version: str | None = None
    meta: str | None = None

    def field_items(self) -> ItemsView[str, str | None]:
        return asdict(self).items()


def get_metadata(obj: Any) -> Metadata | None:
    module = get_module(obj)
    if module == "builtins":
        return None
    else:
        return Metadata(
            module=module,
            version=get_version(module),
            meta=obj.__metadata__ if hasattr(obj, "__metadata__") else None,
        )


def get_module(obj: Any) -> str:
    return obj.__module__ if isinstance(obj, type) else type(obj).__module__


def get_version(module_name: str) -> str | None:
    if module_name == "builtins":
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    module = import_module(module_name)
    for version in ["__version__", "version", "_version"]:
        if hasattr(module, version):
            return str(getattr(module, version))

    return None
