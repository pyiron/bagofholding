from __future__ import annotations

import contextlib
import dataclasses
import pathlib
from collections.abc import Iterator
from typing import Any, ClassVar

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
    filepath: pathlib.Path
    file: h5py.File
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
        version_scraping: VersionScrapingMap | None = None,
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
        with h5py.File(filepath, "a", libver=cls.libver_str) as f:
            pack(
                obj,
                f,
                cls.storage_root,
                bidict.bidict(),
                [],
                version_scraping=version_scraping,
            )

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(filepath, *args, **kwargs)
        self.file = h5py.File(filepath, mode="r", libver=self.libver_str)

    def read_bag_info(self, filepath: pathlib.Path) -> H5Info:
        with h5py.File(filepath, "r", libver=self.libver_str) as f:
            info = H5Info(**{k: f.attrs[k] for k in H5Info.__dataclass_fields__})
        return info

    def _close(self) -> None:
        with contextlib.suppress(AttributeError):
            self.file.close()

    def __del__(self) -> None:
        self._close()

    def load(
        self,
        path: str = Bag.storage_root,
        version_validator: VersionValidatorType = "exact",
    ) -> Any:
        return unpack(self.file, path, {}, version_validator=version_validator)

    def __getitem__(self, path: str) -> Metadata | None:
        return read_metadata(self.file[path])

    def _get_enriched_metadata(
        self, path: str
    ) -> tuple[str, Metadata | None, tuple[str, ...] | None]:
        """
        Enriched browsing information to support a browsing widget.
        Still doesn't actually load the object, but exploits more available information.

        Args:
            path (str): Where in the h5 file to look

        Returns:
            (str): The content type class string.
            (Metadata | None): The metadata, if any.
            (tuple[str, ...] | None): The sub-entry name(s), if any.
        """
        entry = self.file[path]
        return (
            maybe_decode(entry.attrs["content_type"]),
            read_metadata(entry),
            tuple(entry.keys()) if isinstance(entry, h5py.Group) else None,
        )

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        paths: list[str] = []
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
