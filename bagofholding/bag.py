"""
The core user-facing object.

Full implementations of bags should guarantee the key features promised by the package:
- Storage and retrieval of arbitrary pickleable python objects
- Metadata preservation
- Versioning verification
- Browsing without loading
- Partial reloading
"""

from __future__ import annotations

import abc
import dataclasses
import os.path
import pathlib
import pickle
from collections.abc import Iterator, Mapping
from typing import (
    Any,
    ClassVar,
    Self,
    SupportsIndex,
)

import bidict
from pyiron_snippets import import_alarm

from bagofholding.content import BespokeItem, Packer, pack, unpack
from bagofholding.exceptions import BagMismatchError, InvalidMetadataError
from bagofholding.metadata import (
    HasFieldIterator,
    HasVersionInfo,
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_version,
)

try:
    from bagofholding.widget import BagTree

    alarm = import_alarm.ImportAlarm()
except (ImportError, ModuleNotFoundError):
    alarm = import_alarm.ImportAlarm(
        "The browsing widget relies on ipytree and traitlets, but this was "
        "unavailable. You can get a text-representation of all available paths with "
        ":meth:`bagofholding.bag.Bag.list_paths`.",
        _fail_on_warning=True,
    )

PATH_DELIMITER = "/"


@dataclasses.dataclass(frozen=True)
class BagInfo(HasVersionInfo, HasFieldIterator):
    pass


class Bag(Packer, Mapping[str, Metadata | None], abc.ABC):
    """
    Bags are the user-facing object.
    """

    bag_info: BagInfo
    storage_root: ClassVar[str] = "object"
    filepath: pathlib.Path

    @classmethod
    def get_bag_info(cls) -> BagInfo:
        return BagInfo(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=cls.get_version(),
        )

    @classmethod
    def _bag_info_class(cls) -> type[BagInfo]:
        return BagInfo

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
        bag._pack_bag_info()
        pack(
            obj,
            bag,
            bag.storage_root,
            bidict.bidict(),
            [],
            require_versions,
            forbidden_modules,
            version_scraping,
            _pickle_protocol=_pickle_protocol,
        )
        bag._write()

    @classmethod
    def get_version(cls) -> str:
        return str(get_version(cls.__module__, {}))

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)
        if os.path.isfile(self.filepath):
            self.bag_info = self._unpack_bag_info()
            if not self.validate_bag_info(self.bag_info, self.get_bag_info()):
                raise BagMismatchError(
                    f"The bag class {self.__class__} does not match the bag saved at "
                    f"{filepath}; class info is {self.get_bag_info()}, but the info saved "
                    f"is {self.bag_info}"
                )

    @abc.abstractmethod
    def _pack_field(self, path: str, key: str, value: str) -> None: ...

    @abc.abstractmethod
    def _unpack_field(self, path: str, key: str) -> str | None: ...

    @staticmethod
    def validate_bag_info(bag_info: BagInfo, reference: BagInfo) -> bool:
        return bag_info == reference

    def load(
        self,
        path: str = storage_root,
        version_validator: VersionValidatorType = "exact",
        version_scraping: VersionScrapingMap | None = None,
    ) -> Any:
        return unpack(
            self,
            path,
            {},
            version_validator=version_validator,
            version_scraping=version_scraping,
        )

    def __getitem__(self, path: str) -> Metadata:
        return self.unpack_metadata(path)

    @abc.abstractmethod
    def list_paths(self) -> list[str]:
        """A list of all available content paths."""

    @alarm
    def widget(self):  # type: ignore[no-untyped-def]
        return BagTree(self)

    def browse(self):  # type: ignore[no-untyped-def]
        try:
            return self.widget()
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

    def _pack_fields(self, dataclass: HasFieldIterator, path: str) -> None:
        for k, v in dataclass.field_items():
            if v is not None:
                self._pack_field(path, k, v)

    def _unpack_fields(
        self, dataclass_type: type[HasFieldIterator], path: str
    ) -> dict[str, str | None]:
        field_values: dict[str, str | None] = {}
        for k in dataclass_type.__dataclass_fields__:
            field_values[k] = self._unpack_field(path, k)
        return field_values

    def _pack_bag_info(self) -> None:
        self._pack_fields(self.get_bag_info(), PATH_DELIMITER)

    def _unpack_bag_info(self) -> BagInfo:
        return self._bag_info_class()(
            **self._unpack_fields(self._bag_info_class(), PATH_DELIMITER)
        )

    def _write(self) -> None:
        return

    def pack_metadata(self, metadata: Metadata, path: str) -> None:
        self._pack_fields(metadata, path)
        return None

    def unpack_metadata(self, path: str) -> Metadata:
        metadata = self._unpack_fields(Metadata, path)
        content_type = metadata.pop("content_type", None)
        if content_type is None:
            raise InvalidMetadataError(f"Metadata at {path} is missing a content type")
        return Metadata(content_type, **metadata)

    def get_bespoke_content_class(
        self, obj: object
    ) -> type[BespokeItem[Any, Self]] | None:
        return None
