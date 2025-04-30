"""
Tools for extracting and logging information about python objects.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, ItemsView
from importlib import import_module
from sys import version_info
from typing import Any, Literal, TypeAlias

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
    Literal["exact", "semantic-minor", "semantic-major", "none"]
    | Callable[[str, str], bool]
)


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
        validator ("exact" | Callable[[str, str], bool]): A recognized keyword or a
            callable that takes the current and metadata versions as strings and
            returns a boolean to indicate whether the current version matches the
            metadata reference. Keywords are "exact" (versions must be identical),
            "semantic-minor" (semantic versions (X.Y.Z where all are integers) match
            in the first two digits; all non-semantic versions must match exactly),
            "semantic-major" (semantic versions match in the first digit), and "none"
            (don't compare the versions at all).
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

        version_validator: VersionValidatorType
        if validator == "exact":
            version_validator = _versions_are_equal
        elif validator == "semantic-minor":
            version_validator = _versions_match_semantic_minor
        elif validator == "semantic-major":
            version_validator = _versions_match_semantic_major
        else:
            version_validator = validator

        if isinstance(version_validator, str):
            if version_validator == "none":
                return
            else:
                raise ValueError(
                    f"Unrecognized validator keyword {version_validator} -- please supply {VersionValidatorType}"
                )
        elif version_validator(current_version, metadata.version):
            return
        raise EnvironmentMismatchError(
            f"{metadata.module} is stored with version {metadata.version}, "
            f"but the current environment has {current_version}. This does not pass "
            f"validation criterion: {version_validator}"
        )


def _versions_are_equal(version: str, reference: str) -> bool:
    return version == reference


def _decompose_semver(version: str) -> tuple[int, int, int] | None:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match:
        major, minor, patch = match.groups()
        return int(major), int(minor), int(patch)
    return None


def _versions_match_semantic_minor(version: str, reference: str) -> bool:
    v_parts = _decompose_semver(version)
    r_parts = _decompose_semver(reference)
    if v_parts and r_parts:
        return v_parts[:2] == r_parts[:2]
    return version == reference


def _versions_match_semantic_major(version: str, reference: str) -> bool:
    v_parts = _decompose_semver(version)
    r_parts = _decompose_semver(reference)
    if v_parts and r_parts:
        return v_parts[0] == r_parts[0]
    return version == reference
