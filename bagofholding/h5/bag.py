from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator
from typing import Any, ClassVar, SupportsIndex

import bidict
import h5py

from bagofholding.bag import Bag, BagInfo
from bagofholding.h5.content import maybe_decode, pack, read_metadata, unpack
from bagofholding.h5.widget import BagTree
from bagofholding.metadata import Metadata, VersionScrapingMap, VersionValidatorType


@dataclasses.dataclass(frozen=True)
class H5Info(BagInfo):
    libver_str: str


class H5Bag(Bag[H5Info]):
    libver_str: ClassVar[str] = "latest"

    @classmethod
    def _write_bag_info(
        cls,
        filepath: str | pathlib.Path,
        bag_info: H5Info,
    ) -> None:
        with h5py.File(filepath, "w", libver=cls.libver_str) as f:
            for k, v in cls.get_bag_info().field_items():
                f.attrs[k] = v

    @classmethod
    def get_bag_info(cls) -> H5Info:
        return H5Info(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=cls.get_version(),
            libver_str=cls.libver_str,
        )

    @classmethod
    def _save(
        cls,
        obj: Any,
        filepath: str | pathlib.Path,
        require_versions: bool,
        forbidden_modules: list[str] | tuple[str, ...],
        version_scraping: VersionScrapingMap | None,
        _pickle_protocol: SupportsIndex,
    ) -> None:

        with h5py.File(filepath, "a", libver=cls.libver_str) as f:
            pack(
                obj,
                f,
                cls.storage_root,
                bidict.bidict(),
                [],
                require_versions,
                forbidden_modules,
                version_scraping,
                _pickle_protocol=_pickle_protocol,
            )

    def read_bag_info(self, filepath: pathlib.Path) -> H5Info:
        with h5py.File(filepath, "r", libver=self.libver_str) as f:
            info = H5Info(**{k: f.attrs[k] for k in H5Info.__dataclass_fields__})
        return info

    def load(
        self,
        path: str = Bag.storage_root,
        version_validator: VersionValidatorType = "exact",
        version_scraping: VersionScrapingMap | None = None,
    ) -> Any:
        with h5py.File(self.filepath, "r", libver=self.libver_str) as f:
            unpacked = unpack(
                f,
                path,
                {},
                version_validator=version_validator,
                version_scraping=version_scraping,
            )
        return unpacked

    def __getitem__(self, path: str) -> Metadata | None:
        with h5py.File(self.filepath, "r", libver=self.libver_str) as f:
            return read_metadata(f[path])

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
        with h5py.File(self.filepath, "r", libver=self.libver_str) as f:
            entry = f[path]
            return (
                maybe_decode(entry.attrs["content_type"]),
                read_metadata(entry),
                tuple(entry.keys()) if isinstance(entry, h5py.Group) else None,
            )

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        paths: list[str] = []
        with h5py.File(self.filepath, "r", libver=self.libver_str) as f:
            f.visit(paths.append)
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
