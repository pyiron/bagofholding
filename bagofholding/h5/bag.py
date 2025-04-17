from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Iterator
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, cast

import h5py
import numpy as np

from bagofholding.bag import Bag, BagInfo
from bagofholding.exceptions import (
    FileAlreadyOpenError,
    FileNotOpenError,
    NotAGroupError,
)
from bagofholding.h5.content import Array, ArrayPacker, ArrayType
from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST
from bagofholding.metadata import Metadata, VersionScrapingMap, VersionValidatorType


@dataclasses.dataclass(frozen=True)
class H5Info(BagInfo):
    libver_str: str = "latest"


class H5Bag(Bag, ArrayPacker):
    _info_class: ClassVar[type[H5Info]] = H5Info
    libver_str: ClassVar[str] = "latest"
    _content_key: ClassVar[str] = "content_type"
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

    def _pack_bag_info(self) -> None:
        self.open("w")
        super()._pack_bag_info()

    def _write(self) -> None:
        self.close()

    def _unpack_bag_info(self) -> BagInfo:
        with self:
            info = super()._unpack_bag_info()
        return info

    def load(
        self,
        path: str = Bag.storage_root,
        version_validator: VersionValidatorType = "exact",
        version_scraping: VersionScrapingMap | None = None,
    ) -> Any:
        with self:
            unpacked = super().load(
                path=path,
                version_validator=version_validator,
                version_scraping=version_scraping,
            )
        return unpacked

    def __getitem__(self, path: str) -> Metadata:
        with self:
            return super().__getitem__(path)

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

    def _pack_field(self, path: str, key: str, value: str) -> None:
        self.file[path].attrs[key] = value

    def _unpack_field(self, path: str, key: str) -> str | None:
        try:
            return self.maybe_decode(self.file[path].attrs[key])
        except KeyError:
            return None

    @staticmethod
    def maybe_decode(attr: str | bytes) -> str:
        return attr if isinstance(attr, str) else attr.decode("utf-8")

    def pack_empty(self, path: str) -> None:
        self.file.create_dataset(path, data=h5py.Empty(dtype="f"))

    def pack_string(self, obj: str, path: str) -> None:
        self.file.create_dataset(
            path, data=obj, dtype=h5py.string_dtype(encoding="utf-8")
        )

    def unpack_string(self, path: str) -> str:
        return cast(str, self._unpack_raw(path).decode("utf-8"))

    def _pack_raw(self, obj: bytearray | bool | int | float, path: str) -> None:
        self.file.create_dataset(path, data=obj)

    def _unpack_raw(self, path: str) -> Any:
        return self.file[path][()]

    def pack_bool(self, obj: bool, path: str) -> None:
        return self._pack_raw(obj, path)

    def unpack_bool(self, path: str) -> bool:
        return bool(self._unpack_raw(path))

    def pack_long(self, obj: int, path: str) -> None:
        return self._pack_raw(obj, path)

    def unpack_long(self, path: str) -> int:
        return int(self._unpack_raw(path))

    def pack_float(self, obj: float, path: str) -> None:
        return self._pack_raw(obj, path)

    def unpack_float(self, path: str) -> float:
        return float(self._unpack_raw(path))

    def pack_complex(self, obj: complex, path: str) -> None:
        self.file.create_dataset(path, data=np.array([obj.real, obj.imag]))

    def unpack_complex(self, path: str) -> complex:
        data = self._unpack_raw(path)
        return complex(data[0], data[1])

    def pack_bytes(self, obj: bytes, path: str) -> None:
        self.file.create_dataset(path, data=np.void(obj))

    def unpack_bytes(self, path: str) -> bytes:
        return bytes(self._unpack_raw(path))

    def pack_bytearray(self, obj: bytearray, path: str) -> None:
        return self._pack_raw(obj, path)

    def unpack_bytearray(self, path: str) -> bytearray:
        return bytearray(self._unpack_raw(path))

    def create_group(self, path: str) -> None:
        self.file.create_group(path)

    def open_group(self, path: str) -> list[str]:
        with self:
            group = self.file[path]
            if not isinstance(group, h5py.Group):
                raise NotAGroupError(f"Asked a group at {path}, got {type(group)}")
            subcontent_names = list(group)
        return subcontent_names

    # def get_bespoke_content_class(self, obj: object) -> type[BespokeItem[Any, Self]] | None:
    def get_bespoke_content_class(self, obj: object) -> type[Array] | None:
        if type(obj) is np.ndarray and obj.dtype in H5PY_DTYPE_WHITELIST:
            return Array
        return None

    def pack_array(self, obj: ArrayType, path: str) -> None:
        self.file.create_dataset(path, data=obj)

    def unpack_array(self, path: str) -> ArrayType:
        return cast(ArrayType, self.file[path][()])
