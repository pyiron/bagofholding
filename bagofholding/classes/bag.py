from __future__ import annotations

import contextlib
import pathlib
from collections.abc import Iterator
from typing import Any, ClassVar

import bidict
import h5py

from bagofholding.bag import Bag
from bagofholding.classes.content import pack, read_metadata, unpack
from bagofholding.metadata import Metadata


class ClassH5Bag(Bag):
    filepath: pathlib.Path
    file: h5py.File
    libver: ClassVar[str | tuple[str, str] | None] = "latest"

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(filepath, *args, **kwargs)
        self.file = h5py.File(filepath, mode="r", libver=self.libver)

    def _close(self) -> None:
        with contextlib.suppress(AttributeError):
            self.file.close()

    def __del__(self) -> None:
        self._close()

    @classmethod
    def save(cls, obj: Any, filepath: str | pathlib.Path) -> None:
        with h5py.File(filepath, "w", libver=cls.libver) as f:
            pack(obj, f, cls.storage_root, bidict.bidict(), [])

    def load(self, path: str = Bag.storage_root) -> Any:
        return unpack(self.file, path, {})

    def __getitem__(self, path: str) -> Metadata | None:
        return read_metadata(self.file[path])

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        paths: list[str] = []
        self.file.visit(paths.append)
        return paths

    def __len__(self) -> int:
        return len(self.list_paths())

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_paths())
