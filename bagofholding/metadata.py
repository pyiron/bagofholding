from __future__ import annotations

from collections.abc import Callable, ItemsView
from dataclasses import asdict, dataclass
from importlib import import_module
from sys import version_info
from typing import Any


@dataclass
class Metadata:
    qualname: str | None = None
    module: str | None = None
    version: str | None = None
    meta: str | None = None

    def field_items(self) -> ItemsView[str, str | None]:
        return asdict(self).items()


def get_metadata(
    obj: Any,
    version_scraping: dict[str, Callable[[str], str | None]] | None = None,
) -> Metadata | None:
    """

    Args:
        obj (Any): The object who's module to extract metadata from.
        version_scraping (dict[str, Callable[[str], str]] | None): An optional
            dictionary mapping module names to a callable that takes this name and
            returns a version (or None). The default callable imports the module
            string and looks for a `__version__` attribute.

    Returns:
        (Metadata|None): The metadata extracted from the object, or `None` if the
            object is builtin.
    """
    module = _get_module(obj)
    if module == "builtins":
        return None
    else:
        return Metadata(
            qualname=obj.__class__.__qualname__,
            module=module,
            version=_get_version(
                module, {} if version_scraping is None else version_scraping
            ),
            meta=str(obj.__metadata__) if hasattr(obj, "__metadata__") else None,
        )


def _get_module(obj: Any) -> str:
    return obj.__module__ if isinstance(obj, type) else type(obj).__module__


def _get_version(
    module_name: str, version_scraping: dict[str, Callable[[str], str | None]]
) -> str | None:
    if module_name == "builtins":
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    scraper = version_scraping.get(module_name, _scrape_version_attribute)
    return scraper(module_name)


def _scrape_version_attribute(module_name: str) -> str | None:
    module = import_module(module_name.split(".")[0])
    try:
        return str(module.__version__)
    except AttributeError:
        return None


@dataclass(frozen=True)
class BagInfo:
    qualname: str
    module: str
    version: str

    def field_items(self) -> ItemsView[str, str | None]:
        return asdict(self).items()
