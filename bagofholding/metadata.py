from __future__ import annotations

from collections.abc import Callable, ItemsView
from dataclasses import asdict, dataclass
from importlib import import_module
from sys import version_info
from typing import Any, Literal

from jedi.inference.gradual.typing import TypeAlias

from bagofholding.exception import BagOfHoldingError


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
            version=get_version(
                module, {} if version_scraping is None else version_scraping
            ),
            meta=str(obj.__metadata__) if hasattr(obj, "__metadata__") else None,
        )


def _get_module(obj: Any) -> str:
    return obj.__module__ if isinstance(obj, type) else type(obj).__module__


def get_version(
    module_name: str,
    version_scraping: dict[str, Callable[[str], str | None]] | None = None,
) -> str | None:
    if module_name == "builtins":
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"

    module_base = module_name.split(".")[0]
    scraper_map = {} if version_scraping is None else version_scraping
    scraper = scraper_map.get(module_base, _scrape_version_attribute)
    return scraper(module_base)


def _scrape_version_attribute(module_name: str) -> str | None:
    module = import_module(module_name)
    try:
        return str(module.__version__)
    except AttributeError:
        return None


class EnvironmentMismatch(BagOfHoldingError, ModuleNotFoundError):
    pass


VersionValidatorType: TypeAlias = (
    Literal["exact", "semantic-minor", "semantic-major"] | Callable[[str, str], bool]
)


def validate_version(
    metadata: Metadata,
    validator: VersionValidatorType = "exact",
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
            current_version = str(get_version(metadata.module))
        except ModuleNotFoundError as e:
            raise EnvironmentMismatch(
                f"When unpacking an object, encountered a module {metadata.module}  "
                f"in the metadata that could not be found in the current environment."
            ) from e

        if validator == "exact":
            version_validator = versions_are_equal
        elif validator == "semantic-minor":
            raise NotImplementedError("semantic-minor not implemented")
        elif validator == "semantic-major":
            raise NotImplementedError("semantic-major not implemented")
        else:
            version_validator = validator

        if (
            isinstance(version_validator, str) and version_validator == "none"
        ) or version_validator(current_version, metadata.version):
            return
        raise EnvironmentMismatch(
            f"{metadata.module} is stored with version {metadata.version}, "
            f"but the current environment has {current_version}. This does not pass "
            f"validation criterion: {version_validator}"
        )


def versions_are_equal(version: str, reference: str) -> bool:
    return version == reference
