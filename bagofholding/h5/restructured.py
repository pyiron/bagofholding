from __future__ import annotations

import pathlib
from types import TracebackType
from typing import Any, ClassVar, Literal, Self, TypeAlias, TypeVar, cast

import bidict
import h5py
import numpy as np
import pygtrie

from bagofholding.bag import PATH_DELIMITER, Bag
from bagofholding.content import BespokeItem
from bagofholding.exceptions import (
    FileAlreadyOpenError,
    FileNotOpenError,
    NotAGroupError,
)
from bagofholding.h5.bag import H5Info
from bagofholding.h5.content import Array, ArrayPacker, ArrayType
from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST, IntTypesAlias
from bagofholding.metadata import Metadata, VersionScrapingMap, VersionValidatorType

PackedThingType = TypeVar("PackedThingType", str, bool, int, float, bytes, bytearray)

StringArrayType: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.str_]]
IntArrayType: TypeAlias = np.ndarray[tuple[int, ...], IntTypesAlias]


class RestructuredH5Bag(Bag[H5Info], ArrayPacker):
    libver_str: ClassVar[str] = "latest"
    _content_key: ClassVar[str] = "content_type"

    _paths_key: ClassVar[str] = "paths"
    _type_index_key: ClassVar[str] = "type_index"
    _position_index_key: ClassVar[str] = "position_index"
    _index_map: ClassVar[bidict.bidict[str, int]] = bidict.bidict(
        {
            "str": 0,
            "bool": 1,
            "long": 2,
            "float": 3,
            "complex_real": 4,
            "complex_imag": 5,
            "bytes": 6,
            "bytearray": 7,
            "array": 8,
            "empty": 9,
            "group": 10,
        }
    )
    _field_delimiter: ClassVar[str] = "::"
    _child_delimiter: ClassVar[str] = ";"

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

    @classmethod
    def _bag_info_class(cls) -> type[H5Info]:
        return H5Info

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        self._file = None
        self._context_depth = 0
        self._unpacked_paths: StringArrayType | None = None
        self._unpacked_type_index: IntArrayType | None = None
        self._unpacked_position_index: IntArrayType | None = None
        self._unpacked_nonmetadata_paths: StringArrayType | None = None
        self._path_to_index: dict[str, int] | None = None
        self._trie: pygtrie.CharTrie | None = None
        super().__init__(filepath)
        self._packed_paths: list[str] = []
        self._packed_type_index: list[int] = []
        self._packed_position_index: list[int] = []
        self._packed: tuple[
            list[str],
            list[bool],
            list[int],
            list[float],
            list[float],
            list[float],
            list[bytes],
            list[bytearray],
            list[ArrayType],
        ] = ([], [], [], [], [], [], [], [], [])

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise FileNotOpenError(f"{self.filepath} is not open; use `open` or `with`")
        return self._file

    @file.setter
    def file(self, new_file: h5py.File | None) -> None:
        self._file = new_file

    @property
    def unpacked_paths(self) -> StringArrayType:
        if self._unpacked_paths is None:
            with self:
                self._unpacked_paths = self.file[self._paths_key][:].astype("str")
        return self._unpacked_paths

    @property
    def unpacked_type_index(self) -> IntArrayType:
        if self._unpacked_type_index is None:
            with self:
                self._unpacked_type_index = self.file[self._type_index_key][:]
        return self._unpacked_type_index

    @property
    def unpacked_position_index(self) -> IntArrayType:
        if self._unpacked_position_index is None:
            with self:
                self._unpacked_position_index = self.file[self._position_index_key][:]
        return self._unpacked_position_index

    @property
    def unpacked_nonmetadata_paths(self) -> StringArrayType:
        if self._unpacked_nonmetadata_paths is None:
            self._unpacked_nonmetadata_paths = self.unpacked_paths[
                ~np.char.find(self.unpacked_paths, self._field_delimiter) >= 0
            ].tolist()
        return self._unpacked_nonmetadata_paths

    @property
    def path_to_index(self) -> dict[str, int]:
        if self._path_to_index is None:
            self._path_to_index = {p: i for i, p in enumerate(self.unpacked_paths)}
        return self._path_to_index

    @property
    def trie(self) -> pygtrie.CharTrie:
        if self._trie is None:
            self._trie = pygtrie.CharTrie()
            for path in self.list_paths():
                self._trie[path] = True
        return self._trie

    def _write(self) -> None:
        str_type = h5py.string_dtype(encoding="utf-8")

        self.open("w")
        self.file.create_dataset(
            self._paths_key, data=np.array(self._packed_paths, dtype=str_type)
        )
        self.file.create_dataset(
            self._type_index_key, data=np.array(self._packed_type_index, dtype=int)
        )
        self.file.create_dataset(
            self._position_index_key,
            data=np.array(self._packed_position_index, dtype=int),
        )

        self.file.create_dataset("str", data=np.array(self._packed[0], dtype=str_type))
        self.file.create_dataset("bool", data=np.array(self._packed[1], dtype=bool))
        self.file.create_dataset("long", data=np.array(self._packed[2], dtype=int))
        self.file.create_dataset("float", data=np.array(self._packed[3], dtype=float))
        self.file.create_dataset(
            "complex_real", data=np.array(self._packed[4], dtype=float)
        )
        self.file.create_dataset(
            "complex_imag", data=np.array(self._packed[5], dtype=float)
        )
        self.file.create_dataset("bytes", data=np.array(self._packed[6]))
        self.file.create_dataset(
            "bytearray", data=np.array(self._packed[7])
        )  # dtype=bytearray
        array_group = self.file.create_group("ndarrays")
        for i, ra in enumerate(self._packed[8]):
            array_group.create_dataset(f"i{i}", data=ra)
        # Empty doesn't need to be packed -- it's always None so the meta info is enough

        self.close()

    def _unpack_bag_info(self) -> H5Info:
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
                path=self._sanitize_path(path),
                version_validator=version_validator,
                version_scraping=version_scraping,
            )
        return unpacked

    def __getitem__(self, path: str) -> Metadata:
        with self:
            return super().__getitem__(self._sanitize_path(path))

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        return self.unpacked_nonmetadata_paths

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

    def _field_to_path(self, path: str, key: str) -> str:
        return path + self._field_delimiter + key

    def _sanitize_path(self, path: str) -> str:
        return path.rstrip(PATH_DELIMITER).lstrip(PATH_DELIMITER)

    def _pack_field(self, path: str, key: str, value: str) -> None:
        type_index = self._index_map["str"]
        data_list = self._packed[type_index]
        data_list.append(value)  # type: ignore[arg-type]
        self._packed_paths.append(self._field_to_path(path, key))
        self._packed_type_index.append(type_index)
        self._packed_position_index.append(len(data_list) - 1)

    def _pack_path(self, path: str) -> None:
        path = "" if path == PATH_DELIMITER else path  # i.e. sanitize the root
        self._packed_paths.append(path)

    def _unpack_field(self, path: str, key: str) -> str | None:
        try:
            return self.maybe_decode(
                self._read_pathlike(self._field_to_path(path, key))
            )
        except IndexError:
            return None

    def _read_pathlike(self, path: str) -> object:
        # A real path or one with the field delimiter to find a metadata field
        packing_index = self.path_to_index.get(path, None)
        if packing_index is None:
            raise IndexError(
                f"Couldn't find {path} among {self.unpacked_paths}"
            )
        type_index = self.unpacked_type_index[packing_index]
        group_name = self._index_map.inverse[type_index]
        position_index = self.unpacked_position_index[packing_index]
        with self:
            value = self.file[group_name][position_index]
        return value

    @staticmethod
    def maybe_decode(attr: str | bytes) -> str:
        return attr if isinstance(attr, str) else attr.decode("utf-8")

    def pack_empty(self, path: str) -> None:
        self._pack_path(path)
        self._packed_type_index.append(self._index_map["empty"])
        self._packed_position_index.append(0)

    def _pack_thing(self, obj: PackedThingType, type_name: str, path: str) -> None:
        type_index = self._index_map[type_name]
        group = self._packed[type_index]
        group.append(obj)  # type: ignore[arg-type]
        self._pack_path(path)
        self._packed_type_index.append(type_index)
        self._packed_position_index.append(len(group) - 1)

    def pack_string(self, obj: str, path: str) -> None:
        self._pack_thing(obj, "str", path)

    def unpack_string(self, path: str) -> str:
        return self.maybe_decode(self._read_pathlike(path))

    def pack_bool(self, obj: bool, path: str) -> None:
        self._pack_thing(obj, "bool", path)

    def unpack_bool(self, path: str) -> bool:
        return bool(self._read_pathlike(path))

    def pack_long(self, obj: int, path: str) -> None:
        self._pack_thing(obj, "long", path)

    def unpack_long(self, path: str) -> int:
        return int(self._read_pathlike(path))

    def pack_float(self, obj: float, path: str) -> None:
        self._pack_thing(obj, "float", path)

    def unpack_float(self, path: str) -> float:
        return float(self._read_pathlike(path))

    def pack_complex(self, obj: complex, path: str) -> None:
        real_index = self._index_map["complex_real"]
        real_group = self._packed[real_index]
        real_group.append(obj.real)  # type: ignore[arg-type]
        imag_index = self._index_map["complex_imag"]
        imag_group = self._packed[imag_index]
        imag_group.append(obj.imag)  # type: ignore[arg-type]
        self._pack_path(path)
        self._packed_type_index.append(real_index)
        self._packed_position_index.append(len(real_group) - 1)

    def unpack_complex(self, path: str) -> complex:
        packing_index = np.argwhere(self.unpacked_paths == path)[0][0]
        position_index = self.unpacked_position_index[packing_index]
        with self:
            value = complex(
                self.file["complex_real"][position_index],
                self.file["complex_imag"][position_index],
            )
        return value

    def pack_bytes(self, obj: bytes, path: str) -> None:
        self._pack_thing(obj, "bytes", path)

    def unpack_bytes(self, path: str) -> bytes:
        return self._read_pathlike(path).tobytes()

    def pack_bytearray(self, obj: bytearray, path: str) -> None:
        self._pack_thing(obj, "bytearray", path)

    def unpack_bytearray(self, path: str) -> bytearray:
        return bytearray(self._read_pathlike(path))

    def create_group(self, path: str) -> None:
        type_index = self._index_map["group"]
        self._pack_path(path)
        self._packed_type_index.append(type_index)
        self._packed_position_index.append(0)

    def open_group(self, path: str) -> set[str]:
        prefix = path if path.endswith(PATH_DELIMITER) else path + PATH_DELIMITER
        subpaths = self.trie.keys(prefix=path)[1:]
        children = {
            key[len(prefix):].split(PATH_DELIMITER, 1)[0] for key in subpaths
        }
        return children

    # def get_bespoke_content_class(self, obj: object) -> type[BespokeItem[Any, Self]] | None:
    def get_bespoke_content_class(
        self, obj: object
    ) -> type[BespokeItem[Any, Self]] | None:
        if type(obj) is np.ndarray and obj.dtype in H5PY_DTYPE_WHITELIST:
            return cast(type[BespokeItem[Any, Self]], Array)
        return None

    def pack_array(self, obj: ArrayType, path: str) -> None:
        self._pack_thing(obj, "array", path)

    def unpack_array(self, path: str) -> ArrayType:
        packing_index = np.argwhere(self.unpacked_paths == path)[0][0]
        position_index = self.unpacked_position_index[packing_index]
        with self:
            value = cast(ArrayType, self.file[f"ndarrays/i{position_index}"][:])
        return value
