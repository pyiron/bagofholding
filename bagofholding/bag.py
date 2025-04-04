from __future__ import annotations

import abc
import dataclasses
import pathlib
from collections.abc import Callable, ItemsView, Iterator, Mapping
from typing import Any, ClassVar, Generic, TypeVar

from bagofholding import __version__
from bagofholding.metadata import Metadata


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
        version_scraping: dict[str, Callable[[str], str | None]] | None = None,
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
        cls._write_bag_info(filepath, cls.get_bag_info())
        cls._save(obj, filepath, version_scraping)

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
        version_scraping: dict[str, Callable[[str], str | None]] | None = None,
    ) -> None:
        pass

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)
        self.bag_info = self.read_bag_info(self.filepath)

    @abc.abstractmethod
    def read_bag_info(self, filepath: pathlib.Path) -> InfoType:
        pass

    @abc.abstractmethod
    def load(self, path: str = storage_root) -> Any:
        pass

    @abc.abstractmethod
    def __getitem__(self, path: str) -> Metadata | None:
        pass

    @abc.abstractmethod
    def list_paths(self) -> list[str]:
        """A list of all available content paths."""

    def __len__(self) -> int:
        return len(self.list_paths())

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_paths())

    @classmethod
    def get_version(self) -> str:
        return str(__version__)
