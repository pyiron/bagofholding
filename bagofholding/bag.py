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
import pathlib
import pickle
import warnings
from collections.abc import Iterator, Mapping
from typing import (
    Any,
    ClassVar,
    Self,
    SupportsIndex,
)

import bidict
from packaging import version as packaging_version
from pyiron_snippets import import_alarm

from bagofholding.content import MAX_PICKLE_PROTOCOL, BespokeItem, Packer, pack, unpack
from bagofholding.exceptions import BagMismatchError, InvalidMetadataError
from bagofholding.metadata import (
    HasFieldIterator,
    HasVersionInfo,
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_version,
    versions_match,
)

try:
    from bagofholding.widget import BagTree

    alarm = import_alarm.ImportAlarm()
except (ImportError, ModuleNotFoundError):
    alarm = import_alarm.ImportAlarm(
        "The browsing widget relies on ipytree and traitlets, but this was "
        "unavailable. You can get a text-representation of all available paths with "
        ":meth:`bagofholding.bag.Bag.list_paths`.",
        raise_exception=True,
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
    min_compatible_version: ClassVar[str | None] = None
    """The oldest bagofholding version whose saved bags this class can read."""
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
        _pickle_protocol: SupportsIndex = MAX_PICKLE_PROTOCOL,
        overwrite_existing: bool = True,
        require_importable: bool | None = None,
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
            overwrite_existing (bool): Whether to overwrite an existing bag at the
                target location. (Default is True.)
            require_importable (bool | None): DEPRECATED, and will be removed in a
                future version, after which the check will always run, as in `pickle`.
                Whether to fail at save time if any stored global (class, function,
                etc.) cannot be re-imported from its module and qualified name as the
                very same object. This catches, e.g., classes made by factories
                (`collections.namedtuple`, `dataclasses.make_dataclass`, `type`),
                defined in modules that were never registered in `sys.modules`, or
                stale after their definition was re-run (e.g. re-executing a notebook
                cell). (Default is None, which behaves as True. Set it False to
                deliberately store objects that can be browsed but not (yet) loaded,
                e.g. if you will re-execute code to make them importable before
                loading.) Objects in `__main__` are importable at save time but not in
                a fresh interpreter; use `forbidden_modules` to guard against that.
        """
        if require_importable is None:
            require_importable = True
        else:
            warnings.warn(
                "`require_importable` is deprecated and will be removed in a future "
                "version. To be `pickle`-compliant, saving will then always require "
                "stored globals to be re-importable as the same object.",
                DeprecationWarning,
                stacklevel=2,
            )
        bag = cls._new_for_save(filepath, overwrite_existing)
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
            require_importable=require_importable,
            _pickle_protocol=_pickle_protocol,
        )
        bag._write()

    @classmethod
    @abc.abstractmethod
    def _new_for_save(
        cls, filepath: str | pathlib.Path, overwrite_existing: bool
    ) -> Self:
        """Hook: build a bag instance ready to be packed into.

        Implementations are responsible for clearing or validating the target
        at ``filepath`` (honoring ``overwrite_existing``) and returning an
        instance whose backing store is prepared for a fresh write.
        """

    @classmethod
    def get_version(cls) -> str:
        return str(get_version(cls.__module__, {}))

    def __init__(
        self,
        filepath: str | pathlib.Path,
        *args: object,
        bag_version_validator: VersionValidatorType = "semantic-minor",
        _skip_load: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Open a bag at a path.

        Args:
            filepath (str | pathlib.Path): Where the bag lives (or will live).
            bag_version_validator (VersionValidatorType): How strictly the
                bagofholding version saved in an existing bag must match the
                current one; see :func:`bagofholding.metadata.versions_match`. All
                other bag info (class, module, and implementation-specific fields)
                must always match exactly. (Default is "semantic-minor".)
        """
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)
        if _skip_load:
            return
        info = self._load_existing_bag_info()
        if info is not None:
            self.bag_info = info
            if bag_version_validator != "none" and not self._meets_version_floor(info):
                raise BagMismatchError(
                    f"The bag saved at {filepath} has bagofholding version "
                    f"{info.version}, but {self.__class__.__name__} can only read bags "
                    f"saved with version {self.min_compatible_version} or later. Use "
                    f'bag_version_validator="none" to attempt loading anyway.'
                )
            if not self.validate_bag_info(
                info, self.get_bag_info(), bag_version_validator
            ):
                raise BagMismatchError(
                    f"The bag class {self.__class__} does not match the bag saved at "
                    f"{filepath} under bag version validator {bag_version_validator}; "
                    f"class info is {self.get_bag_info()}, but the info saved is "
                    f"{self.bag_info}"
                )

    @abc.abstractmethod
    def _load_existing_bag_info(self) -> BagInfo | None:
        """Return the saved :class:`BagInfo` at the target, or ``None`` if absent.

        Implementations should recognize the backing store's notion of a
        target (e.g., a file, or a group inside an HDF5 file) and fold the
        existence check and the unpack into a single read.
        """

    @abc.abstractmethod
    def _pack_field(self, path: str, key: str, value: str) -> None: ...

    @abc.abstractmethod
    def _unpack_field(self, path: str, key: str) -> str | None: ...

    @classmethod
    def _meets_version_floor(cls, bag_info: BagInfo) -> bool:
        """
        Whether saved bag info is at least :attr:`min_compatible_version`.

        The floor is a version of the module declaring it, so it is only applied to
        bags saved by that module; subclasses elsewhere record their own versions.
        """
        floor_owner = next(
            c for c in cls.__mro__ if "min_compatible_version" in c.__dict__
        )
        if cls.min_compatible_version is None or (
            bag_info.module != floor_owner.__module__
        ):
            return True
        if bag_info.version is None:
            return False
        try:
            return packaging_version.Version(
                bag_info.version
            ) >= packaging_version.Version(cls.min_compatible_version)
        except packaging_version.InvalidVersion:
            return False

    @staticmethod
    def validate_bag_info(
        bag_info: BagInfo,
        reference: BagInfo,
        version_validator: VersionValidatorType = "exact",
    ) -> bool:
        if dataclasses.replace(bag_info, version=None) != dataclasses.replace(
            reference, version=None
        ):
            return False
        if bag_info.version is None or reference.version is None:
            return bag_info.version == reference.version
        return versions_match(reference.version, bag_info.version, version_validator)

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
