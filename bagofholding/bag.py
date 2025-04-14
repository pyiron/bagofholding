from __future__ import annotations

import abc
import dataclasses
import os.path
import pathlib
import pickle
from collections.abc import ItemsView, Iterator, Mapping
from typing import Any, ClassVar, Generic, Protocol, SupportsIndex, TypeVar

from bagofholding.exception import BagOfHoldingError
from bagofholding.metadata import (
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_version,
)

PATH_DELIMITER = "/"


class BagMismatchError(BagOfHoldingError, ValueError):
    pass


class FilepathError(BagOfHoldingError, FileExistsError):
    pass


@dataclasses.dataclass(frozen=True)
class BagInfo:
    qualname: str
    module: str
    version: str

    def field_items(self) -> ItemsView[str, str | None]:
        return dataclasses.asdict(self).items()


InfoType = TypeVar("InfoType", bound=BagInfo)


class GroupLike(Protocol):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]:
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        pass


class Bag(Mapping[str, Metadata | None], Generic[InfoType], abc.ABC):
    """
    Bags are the user-facing object.
    """

    bag_info: InfoType
    storage_root: ClassVar[str] = "object"
    filepath: pathlib.Path

    @classmethod
    @abc.abstractmethod
    def get_bag_info(cls) -> InfoType:
        pass

    @classmethod
    def save(
        cls,
        obj: Any,
        filepath: str | pathlib.Path,
        require_versions: bool = False,
        forbidden_modules: list[str] | tuple[str, ...] = (),
        version_scraping: VersionScrapingMap | None = None,
        _pickle_protocol: SupportsIndex = pickle.DEFAULT_PROTOCOL,
        overwrite_existing: bool = True,
    ) -> None:
        """
        Save a python object to file.

        Args:
            obj (Any): The (pickleble) python object to be saved.
            filepath (str|pathlib.Path): The path to save the object to.
            require_versions (bool): Whether to require a metadata for reduced
                and complex objects to contain a non-None version. (Default is False,
                objects can be stored from non-versioned packages/modules.)
            forbidden_modules (list[str] | tuple[str, ...] | None): Do not allow saving
                objects whose root-most modules are listed here. (Default is an empty
                tuple, i.e. don't disallow anything.) This is particularly useful to
                disallow  `"__main__"` to improve the odds that objects will actually
                be loadable in the future.
            version_scraping (dict[str, Callable[[str], str]] | None): An optional
                dictionary mapping module names to a callable that takes this name and
                returns a version (or None). The default callable imports the module
                string and looks for a `__version__` attribute.
        """
        if os.path.exists(filepath):
            if overwrite_existing and os.path.isfile(filepath):
                os.remove(filepath)
            else:
                raise FileExistsError(f"{filepath} already exists or is not a file.")
        bag = cls(filepath)
        bag._write_bag_info(cls.get_bag_info())
        bag._save(
            obj,
            require_versions,
            forbidden_modules,
            version_scraping,
            _pickle_protocol,
        )

    @classmethod
    def get_version(cls) -> str:
        return str(get_version(cls.__module__, {}))

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)
        if os.path.isfile(self.filepath):
            self.bag_info = self.read_bag_info(self.filepath)
            if not self.validate_bag_info(self.bag_info, self.get_bag_info()):
                raise BagMismatchError(
                    f"The bag class {self.__class__} does not match the bag saved at "
                    f"{filepath}; class info is {self.get_bag_info()}, but the info saved "
                    f"is {self.bag_info}"
                )

    @abc.abstractmethod
    def _write_bag_info(
        self,
        bag_info: InfoType,
    ) -> None:
        pass

    @abc.abstractmethod
    def _save(
        self,
        obj: Any,
        require_versions: bool,
        forbidden_modules: list[str] | tuple[str, ...],
        version_scraping: VersionScrapingMap | None,
        _pickle_protocol: SupportsIndex,
    ) -> None:
        # pass _pickle_protocol to invocations of __reduce_ex__
        pass

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

    def join(self, *paths: str) -> str:
        return PATH_DELIMITER.join(paths)

    @staticmethod
    def pickle_check(
        obj: Any, raise_exceptions: bool = True, print_message: bool = False
    ) -> str | None:
        """
        A simple helper to check if an object can be pickled and unpickled.
        Useful if you run into trouble saving or loading and want to see whether the
        underlying object is compliant with pickle-ability requirements to begin with.

        Args:
            obj: The object to test for pickling support.
            raise_exceptions: If True, re-raise any exception encountered.
            print_message: If True, print the exception message on failure.

        Returns:
            None if pickling is successful; otherwise, returns the exception message as a string.
        """

        try:
            pickle.loads(pickle.dumps(obj))
        except Exception as e:
            if print_message:
                print(e)
            if raise_exceptions:
                raise e
            return str(e)
        return None

    @abc.abstractmethod
    def open_group(self, path: str) -> GroupLike:
        pass
