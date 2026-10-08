"""
Tools for extracting and logging information about python objects.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, ItemsView
from importlib import import_module
from sys import version_info
from typing import Any, Literal, TypeAlias

from packaging import version as packaging_version

from bagofholding.exceptions import EnvironmentMismatchError


@dataclasses.dataclass(frozen=True)
class HasFieldIterator:
    """A simple helper mixin for dataclasses"""

    def field_items(self) -> ItemsView[str, str | None]:
        return dataclasses.asdict(self).items()


@dataclasses.dataclass(frozen=True)
class HasContentType:
    content_type: str


@dataclasses.dataclass(frozen=True)
class HasVersionInfo:
    qualname: str | None = None
    module: str | None = None
    version: str | None = None


@dataclasses.dataclass(frozen=True)
class Metadata(HasVersionInfo, HasContentType, HasFieldIterator):
    meta: str | None = None


def get_module(obj: Any) -> str:
    return obj.__module__ if isinstance(obj, type) else type(obj).__module__


def get_qualname(obj: Any) -> str:
    return obj.__qualname__ if isinstance(obj, type) else type(obj).__qualname__


VersionScraperType: TypeAlias = Callable[[str], str | None]
VersionScrapingMap: TypeAlias = dict[str, VersionScraperType]


def get_version(
    module_name: str,
    version_scraping: VersionScrapingMap | None = None,
) -> str | None:
    """
    Given a module name, get its associated version (if any). By default, this simply
    looks for the :attr:`__version__` attribute on the imported module.

    For :mod:`builtins` this is just the python interpreter version.

    Args:
        module_name (str): The module to examine.
        version_scraping (VersionScrapingMap | None): Since some modules may store
            their version in other ways, this provides an optional map between module
            names and callables to leverage for extracting that module's version.

    Returns:
        (str | None): The module's version as a string, if any can be found.
    """
    if module_name == "builtins":
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    module_base = module_name.split(".")[0]
    scraper_map: VersionScrapingMap = (
        {} if version_scraping is None else version_scraping
    )

    scraper = (
        scraper_map[module_base]  # noqa: SIM401
        if module_base in scraper_map
        else _scrape_version_attribute
    )
    # mypy struggles with .get even when the fallback is specified,
    # so break it apart and tell Ruff to not worry that we avoid .get
    return scraper(module_base)


def _scrape_version_attribute(module_name: str) -> str | None:
    module = import_module(module_name)
    try:
        return str(module.__version__)
    except AttributeError:
        return None


VersionValidatorType: TypeAlias = (
    Literal["exact", "semantic-patch", "semantic-minor", "semantic-major", "none"]
    | Callable[[str, str], bool]
)

_SEMANTIC_DEPTHS: dict[str, int] = {
    "semantic-patch": 3,
    "semantic-minor": 2,
    "semantic-major": 1,
}


def versions_match(current: str, stored: str, validator: VersionValidatorType) -> bool:
    """
    Compare a current version string against a stored reference.

    Args:
        current (str): The version in the current environment.
        stored (str): The version recorded at save time.
        validator (VersionValidatorType): "exact" (literal string equality),
            "semantic-patch"/"semantic-minor"/"semantic-major" (PEP 440 versions
            match in their major.minor.micro / major.minor / major release
            components, ignoring pre-, post-, dev- and local segments; versions
            that cannot be parsed must match exactly), "none" (always matches), or
            a callable taking `(current, stored)` and returning a bool.

    Returns:
        (bool): Whether the versions match under the validator.

    Raises:
        ValueError: If the validator is an unrecognized keyword.
    """
    if validator == "none":
        return True
    if validator == "exact":
        return current == stored
    if isinstance(validator, str):
        if validator not in _SEMANTIC_DEPTHS:
            raise ValueError(
                f"Unrecognized validator keyword {validator} -- please supply "
                f"{VersionValidatorType}"
            )
        return _semantic_match(current, stored, _SEMANTIC_DEPTHS[validator])
    return validator(current, stored)


def _semantic_match(current: str, stored: str, depth: int) -> bool:
    try:
        current_release = _release(packaging_version.Version(current))
        stored_release = _release(packaging_version.Version(stored))
    except packaging_version.InvalidVersion:
        return current == stored
    return current_release[:depth] == stored_release[:depth]


def _release(version: packaging_version.Version) -> tuple[int, int, int]:
    return version.major, version.minor, version.micro


def validate_version(
    metadata: Metadata,
    validator: VersionValidatorType = "exact",
    version_scraping: VersionScrapingMap | None = None,
) -> None:
    """
    Check whether versioning information in a piece of metadata matches the current
    environment.

    Args:
        metadata (Metadata): The metadata to validate.
        validator (VersionValidatorType): A recognized keyword or a callable;
            see :func:`versions_match` for semantics.
        version_scraping (dict[str, Callable[[str], str]] | None): An optional
            dictionary mapping module names to a callable that takes this name and
            returns a version (or None). The default callable imports the module
            string and looks for a `__version__` attribute.

    Raises:
        EnvironmentMismatch: If the module in the metadata cannot be found, or if the
            current and metadata versions do not pass validation.
    """
    if (
        metadata.version is not None
        and metadata.version != ""
        and isinstance(metadata.module, str)
    ):
        try:
            current_version = str(get_version(metadata.module, version_scraping))
        except ModuleNotFoundError as e:
            raise EnvironmentMismatchError(
                f"When unpacking an object, encountered a module {metadata.module}  "
                f"in the metadata that could not be found in the current environment."
            ) from e

        if versions_match(current_version, metadata.version, validator):
            return
        raise EnvironmentMismatchError(
            f"{metadata.module} is stored with version {metadata.version}, "
            f"but the current environment has {current_version}. This does not pass "
            f"validation criterion: {validator}"
        )
