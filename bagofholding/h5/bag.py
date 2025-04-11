from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, SupportsIndex

import bidict
import h5py

from bagofholding.bag import Bag, BagInfo
from bagofholding.exception import BagOfHoldingError
from bagofholding.h5.content import pack, unpack
from bagofholding.h5.widget import BagTree
from bagofholding.metadata import Metadata, VersionScrapingMap, VersionValidatorType


class FileAlreadyOpenError(BagOfHoldingError):
    pass


class FileNotOpenError(BagOfHoldingError):
    pass


@dataclasses.dataclass(frozen=True)
class H5Info(BagInfo):
    libver_str: str


class H5Bag(Bag[H5Info]):
    libver_str: ClassVar[str] = "latest"
    _file: h5py.File | None
    _context_depth: int

    @classmethod
    def get_bag_info(cls) -> H5Info:
        return H5Info(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=cls.get_version(),
            libver_str=cls.libver_str,
        )

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        self._file = None
        self._context_depth = 0
        super().__init__(filepath)

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise FileNotOpenError(f"{self.filepath} is not open; use `open` or `with`")
        return self._file

    @file.setter
    def file(self, new_file: h5py.File | None) -> None:
        self._file = new_file

    def _write_bag_info(
        self,
        bag_info: H5Info,
    ) -> None:
        try:
            self.open("w")
            for k, v in bag_info.field_items():
                self.file.attrs[k] = v
        finally:
            self.close()

    def _save(
        self,
        obj: Any,
        require_versions: bool,
        forbidden_modules: list[str] | tuple[str, ...],
        version_scraping: VersionScrapingMap | None,
        _pickle_protocol: SupportsIndex,
    ) -> None:
        try:
            self.open("a")
            pack(
                obj,
                self,
                self.storage_root,
                bidict.bidict(),
                [],
                require_versions,
                forbidden_modules,
                version_scraping,
                _pickle_protocol=_pickle_protocol,
            )
        finally:
            self.close()

    def read_bag_info(self, filepath: pathlib.Path) -> H5Info:
        with self:
            info = H5Info(
                **{k: self.file.attrs[k] for k in H5Info.__dataclass_fields__}
            )
        return info

    def load(
        self,
        path: str = Bag.storage_root,
        version_validator: VersionValidatorType = "exact",
        version_scraping: VersionScrapingMap | None = None,
    ) -> Any:
        with self:
            unpacked = unpack(
                self,
                path,
                {},
                version_validator=version_validator,
                version_scraping=version_scraping,
            )
        return unpacked

    def __getitem__(self, path: str) -> Metadata | None:
        with self:
            return self.unpack_metadata(path)

    def get_enriched_metadata(
        self, path: str
    ) -> tuple[str, Metadata | None, tuple[str, ...] | None]:
        """
        Enriched browsing information, e.g. to support a browsing widget.
        Still doesn't actually load the object, but exploits more available information.

        Args:
            path (str): Where in the h5 file to look

        Returns:
            (str): The content type class string.
            (Metadata | None): The metadata, if any.
            (tuple[str, ...] | None): The sub-entry name(s), if any.
        """
        with self:
            entry = self.file[path]
            return (
                self.unpack_meta(path, "content_type"),
                self.unpack_metadata(path),
                tuple(entry.keys()) if isinstance(entry, h5py.Group) else None,
            )

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        paths: list[str] = []
        with self:
            self.file.visit(paths.append)
        return paths

    def __len__(self) -> int:
        return len(self.list_paths())

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_paths())

    def browse(self) -> BagTree | list[str]:
        try:
            return BagTree(self)  # type: ignore
            # BagTree is wrapped by pyiron_snippets.import_alarm.ImportAlarm.__call__
            # and this is not correctly passing on the hint
        except ImportError:
            return self.list_paths()

    def __enter__(self) -> Self:
        self._context_depth += 1
        if self._file is None:
            self.open("r")
        return self

    def open(self, mode: Literal["r", "r+", "w", "w-", "x", "a"]) -> h5py.File:
        if self._file is None:
            self.file = h5py.File(self.filepath, mode, libver=self.libver_str)
            return self.file
        else:
            raise FileAlreadyOpenError(f"The bag at {self.filepath} is already open")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._context_depth -= 1
        if self._context_depth == 0:
            self.close()

    def close(self) -> None:
        if self._file is not None:
            self.file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    def pack_meta(self, path: str, key: str, value: str) -> None:
        self.file[path].attrs[key] = value

    def unpack_meta(self, path: str, key: str) -> str:
        return self.maybe_decode(self.file[path].attrs[key])

    def pack_metadata(self, path: str, metadata: Metadata | None) -> None:
        if metadata is not None:
            for k, v in metadata.field_items():
                if v is not None:
                    self.pack_meta(path, k, v)

    def unpack_metadata(self, path: str) -> Metadata | None:
        metadata = {}
        has_metadata = False
        for meta_key in Metadata.__dataclass_fields__:
            try:
                metadata[meta_key] = self.unpack_meta(path, meta_key)
                has_metadata = True
            except KeyError:
                metadata[meta_key] = None
        return Metadata(**metadata) if has_metadata else None

    @staticmethod
    def maybe_decode(attr: str | bytes) -> str:
        return attr if isinstance(attr, str) else attr.decode("utf-8")