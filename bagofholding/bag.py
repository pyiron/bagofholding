from __future__ import annotations

import abc
import pathlib
from collections.abc import Callable, Iterator, Mapping
from typing import Any, ClassVar

from bagofholding import __version__
from bagofholding.metadata import BagInfo, Metadata


class Bag(Mapping[str, Metadata | None], abc.ABC):
    """
    Bags are the user-facing object.
    """

    bag_info: BagInfo
    storage_root: ClassVar[str] = "object"
    filepath: pathlib.Path

    @classmethod
    @abc.abstractmethod
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
        pass

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)
        self.bag_info = self.read_bag_info(self.filepath)

    @abc.abstractmethod
    def read_bag_info(self, filepath: pathlib.Path) -> BagInfo:
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

    @classmethod
    def get_bag_info(cls) -> BagInfo:
        return BagInfo(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=cls.get_version(),
        )
