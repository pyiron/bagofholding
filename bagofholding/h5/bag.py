from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Iterator
from typing import Any, ClassVar

import bidict
import h5py

from bagofholding.bag import Bag
from bagofholding.h5.content import maybe_decode, pack, read_metadata, unpack
from bagofholding.h5.widget import BagTree
from bagofholding.metadata import BagInfo, Metadata


class H5Bag(Bag):
    filepath: pathlib.Path
    file: h5py.File
    libver: ClassVar[str | tuple[str, str] | None] = "latest"

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(filepath, *args, **kwargs)
        self.file = h5py.File(filepath, mode="r", libver=self.libver)

    def read_bag_info(self, filepath: pathlib.Path) -> BagInfo:
        with h5py.File(filepath, "r", libver=self.libver) as f:
            info = BagInfo(
                **{k: f.attrs[k] for k in BagInfo.__dataclass_fields__}
            )
        return info

    def _close(self) -> None:
        with contextlib.suppress(AttributeError):
            self.file.close()

    def __del__(self) -> None:
        self._close()

    @classmethod
    def save(cls, obj: Any, filepath: str | pathlib.Path) -> None:
        with h5py.File(filepath, "w", libver=cls.libver) as f:
            for k, v in cls.get_bag_info().field_items():
                f.attrs[k] = v
            pack(obj, f, cls.storage_root, bidict.bidict(), [])

    def load(self, path: str = Bag.storage_root) -> Any:
        return unpack(self.file, path, {})

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
