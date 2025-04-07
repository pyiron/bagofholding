from __future__ import annotations

import abc
import dataclasses
import pathlib
import pickle
from collections.abc import ItemsView, Iterator, Mapping
from typing import Any, ClassVar, Generic, SupportsIndex, TypeVar

from bagofholding.exception import BagOfHoldingError
from bagofholding.metadata import (
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_version,
)


class BagMismatchError(BagOfHoldingError, ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class BagInfo:
    qualname: str
    module: str
    version: str

    def field_items(self) -> ItemsView[str, str | None]:
        return dataclasses.asdict(self).items()


InfoType = TypeVar("InfoType", bound=BagInfo)


class Bag(Mapping[str, Metadata | None], Generic[InfoType], abc.ABC):
    """
    Bags are the user-facing object.
    """

    bag_info: InfoType
    storage_root: ClassVar[str] = "object"
    filepath: pathlib.Path

    @classmethod
    def save(
        cls,
        obj: Any,
        filepath: str | pathlib.Path,
        version_scraping: VersionScrapingMap | None = None,
        _pickle_protocol: SupportsIndex = pickle.DEFAULT_PROTOCOL,
    ) -> None:
        """
        Save a python object to file.

        Args:
            obj (Any): The (pickleble) python object to be saved.
            filepath (str|pathlib.Path): The path to save the object to.
            version_scraping (dict[str, Callable[[str], str]] | None): An optional
                dictionary mapping module names to a callable that takes this name and
                returns a version (or None). The default callable imports the module
                string and looks for a `__version__` attribute.
        """
        # pass _pickle_protocol to invocations of __reduce_ex__
        cls._write_bag_info(filepath, cls.get_bag_info())
        cls._save(obj, filepath, version_scraping, _pickle_protocol)

    @classmethod
    @abc.abstractmethod
    def _write_bag_info(
        cls,
        filepath: str | pathlib.Path,
        bag_info: InfoType,
    ) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def get_bag_info(cls) -> InfoType:
        pass

    @classmethod
    @abc.abstractmethod
    def _save(
        cls,
        obj: Any,
        filepath: str | pathlib.Path,
        version_scraping: VersionScrapingMap | None,
        _pickle_protocol: SupportsIndex,
    ) -> None:
        pass

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)
        self.bag_info = self.read_bag_info(self.filepath)
        if not self.validate_bag_info(self.bag_info, self.get_bag_info()):
            raise BagMismatchError(
                f"The bag class {self.__class__} does not match the bag saved at "
                f"{filepath}; class info is {self.get_bag_info()}, but the info saved "
                f"is {self.bag_info}"
            )

    @abc.abstractmethod
    def read_bag_info(self, filepath: pathlib.Path) -> InfoType:
        pass

    @staticmethod
    def validate_bag_info(bag_info: InfoType, reference: InfoType) -> bool:
        return bag_info == reference

    @abc.abstractmethod
    def load(
        self,
        path: str = storage_root,
        version_validator: VersionValidatorType = "exact",
        version_scraping: VersionScrapingMap | None = None,
    ) -> Any:
        pass

    @abc.abstractmethod
    def __getitem__(self, path: str) -> Metadata | None:
        pass

    @abc.abstractmethod
    def get_enriched_metadata(
        self, path: str
    ) -> tuple[str, Metadata | None, tuple[str, ...] | None]:
        """
        Enriched browsing information, e.g. to support a browsing widget.
        Still doesn't actually load the object, but exploits more available information.

        Args:
            path (str): Where in the h5 file to look

        Returns:
            (str): The content type class string (module and qualname).
            (Metadata | None): The metadata, if any.
            (tuple[str, ...] | None): The sub-entry name(s), if any.
        """

    @abc.abstractmethod
    def list_paths(self) -> list[str]:
        """A list of all available content paths."""

    def __len__(self) -> int:
        return len(self.list_paths())

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_paths())

    @classmethod
    def get_version(cls) -> str:
        return str(get_version(cls.__module__, {}))
