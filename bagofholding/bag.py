from __future__ import annotations

import abc
import dataclasses
import os.path
import pathlib
import pickle
from collections.abc import ItemsView, Iterator, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Protocol,
    SupportsIndex,
    TypeVar,
)

from bagofholding.exceptions import BagMismatchError
from bagofholding.metadata import (
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_version,
)
from bagofholding.widget import BagTree

if TYPE_CHECKING:
    from bagofholding.content import BespokeItem


PATH_DELIMITER = "/"


@dataclasses.dataclass(frozen=True)
class BagInfo:
    qualname: str
    module: str
    version: str

    def field_items(self) -> ItemsView[str, str | None]:
        return dataclasses.asdict(self).items()


InfoType = TypeVar("InfoType", bound=BagInfo)


class HasContents(Protocol):
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
        bag._pack_bag_info(cls.get_bag_info())
        bag._pack(
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
            self.bag_info = self.unpack_bag_info(self.filepath)
            if not self.validate_bag_info(self.bag_info, self.get_bag_info()):
                raise BagMismatchError(
                    f"The bag class {self.__class__} does not match the bag saved at "
                    f"{filepath}; class info is {self.get_bag_info()}, but the info saved "
                    f"is {self.bag_info}"
                )

    @abc.abstractmethod
    def _pack_bag_info(
        self,
        bag_info: InfoType,
    ) -> None:
        pass

    @abc.abstractmethod
    def _pack(
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
    def unpack_bag_info(self, filepath: pathlib.Path) -> InfoType:
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
    def __getitem__(self, path: str) -> Metadata:
        pass

    @abc.abstractmethod
    def list_paths(self) -> list[str]:
        """A list of all available content paths."""

    def browse(self) -> BagTree | list[str]:
        try:
            return BagTree(self)  # type: ignore
            # BagTree is wrapped by pyiron_snippets.import_alarm.ImportAlarm.__call__
            # and this is not correctly passing on the hint
        except ImportError:
            return self.list_paths()

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
    def pack_metadata(self, metadata: Metadata, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_metadata(self, path: str) -> Metadata:
        pass

    @abc.abstractmethod
    def pack_empty(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def pack_string(self, obj: str, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_string(self, path: str) -> str:
        pass

    @abc.abstractmethod
    def pack_bool(self, obj: bool, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_bool(self, path: str) -> bool:
        pass

    @abc.abstractmethod
    def pack_long(self, obj: int, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_long(self, path: str) -> int:
        pass

    @abc.abstractmethod
    def pack_float(self, obj: float, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_float(self, path: str) -> float:
        pass

    @abc.abstractmethod
    def pack_complex(self, obj: complex, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_complex(self, path: str) -> complex:
        pass

    @abc.abstractmethod
    def pack_bytes(self, obj: bytes, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_bytes(self, path: str) -> bytes:
        pass

    @abc.abstractmethod
    def pack_bytearray(self, obj: bytearray, path: str) -> None:
        pass

    @abc.abstractmethod
    def unpack_bytearray(self, path: str) -> bytearray:
        pass

    @abc.abstractmethod
    def create_group(self, path: str) -> None:
        pass

    @abc.abstractmethod
    def open_group(self, path: str) -> HasContents:
        pass

    @staticmethod
    def get_bespoke_content_class(obj: object) -> type[BespokeItem[Any]] | None:
        return None
