from __future__ import annotations

import pathlib
from typing import Any, ClassVar, Self, TypeAlias, TypeVar, cast

import bidict
import h5py
import numpy as np
import pygtrie

from bagofholding.bag import PATH_DELIMITER, Bag, BagInfo
from bagofholding.content import BespokeItem
from bagofholding.h5.bag import H5Info
from bagofholding.h5.content import Array, ArrayPacker, ArrayType
from bagofholding.h5.context import HasH5FileContext
from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST, H5Scalar, IntTypesAlias
from bagofholding.metadata import Metadata, VersionScrapingMap, VersionValidatorType
from bagofholding.trie import decompose_stringtrie, reconstruct_stringtrie

PackedThingType = TypeVar("PackedThingType", str, bool, int, float, bytes, bytearray)

StringArrayType: TypeAlias = np.ndarray[tuple[int, ...], np.dtype[np.str_]]
IntArrayType: TypeAlias = np.ndarray[tuple[int, ...], IntTypesAlias]


class TrieH5Bag(Bag, HasH5FileContext, ArrayPacker):
    """
    A bag using HDF5 files based on `h5py`.

    Uses a trie structure to flatten the stored object. Compared to
    :class:`bagofholding.h5.bag.H5Bag`, this is advantageous for file sizes but
    (currently)  has worse scaling for save times.

    The resulting HDF5 file cannot be directly related to the structure of the stored
    object, but must be re-mapped via mapping fields. Metadata is also lumped in with
    other string data to minimize the number of different h5 groups.
    """

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

    @classmethod
    def get_bag_info(cls) -> BagInfo:
        return H5Info(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=cls.get_version(),
            libver_str=cls.libver_str,
        )

    @classmethod
    def _bag_info_class(cls) -> type[BagInfo]:
        return H5Info

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        self._file = None
        self._context_depth = 0
        self._unpacked_paths: StringArrayType | None = None
        self._unpacked_type_index: IntArrayType | None = None
        self._unpacked_position_index: IntArrayType | None = None
        self._unpacked_nonmetadata_paths: list[str] | None = None
        self._path_to_index: dict[str, int] | None = None
        self._unpacked_trie: pygtrie.StringTrie | None = None
        super().__init__(filepath)
        self._packed_trie: pygtrie.StringTrie = pygtrie.StringTrie()
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
    def unpacked_trie(self) -> pygtrie.StringTrie:
        if self._unpacked_trie is None:
            with self:
                self._unpacked_trie = reconstruct_stringtrie(
                    self.file["trie_segments"][:].astype("str"),
                    self.file["trie_parents"][:],
                    [v.tolist() for v in self.file["trie_values"][:]],
                    [-1, -1],
                )
        return self._unpacked_trie

    def _write(self) -> None:
        str_type = h5py.string_dtype(encoding="utf-8")

        self.open("w")
        segments, parents, values = decompose_stringtrie(
            self._packed_trie, null_value=(-1, -1)
        )
        self.file.create_dataset(
            "trie_segments",
            data=np.array(segments, dtype=h5py.string_dtype(encoding="utf-8")),
        )
        self.file.create_dataset("trie_parents", data=np.array(parents, dtype=np.int32))
        self.file.create_dataset("trie_values", data=np.array(values, dtype=np.int32))

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
        # Groups don't need to be packed -- they are just holders so meta info is enough

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
            return super().__getitem__(self._sanitize_path(path))

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        if self._unpacked_nonmetadata_paths is None:
            paths = self.unpacked_trie.keys()
            self._unpacked_nonmetadata_paths = [
                self._sanitize_path(p)
                for p in np.array(paths)[
                    ~np.char.find(paths, self._field_delimiter) >= 0
                ]
            ]
        return self._unpacked_nonmetadata_paths

    def __enter__(self) -> Self:
        self._context_depth += 1
        if self._file is None:
            self.open("r")
        return self

    def _pack_trie(self, path: str, type_index: int, position_index: int) -> None:
        self._packed_trie[PATH_DELIMITER + path] = [type_index, position_index]

    def _read_trie(self, path: str) -> tuple[int, int]:
        return cast(
            tuple[int, int],
            self.unpacked_trie.values(prefix=PATH_DELIMITER + path, shallow=True)[0],
        )

    def _field_to_path(self, path: str, key: str) -> str:
        return self._sanitize_path(path) + self._field_delimiter + key

    def _sanitize_path(self, path: str) -> str:
        return path.rstrip(PATH_DELIMITER).lstrip(PATH_DELIMITER)

    def _pack_field(self, path: str, key: str, value: str) -> None:
        type_index = self._index_map["str"]
        data_list = self._packed[type_index]
        data_list.append(value)  # type: ignore[arg-type]
        self._pack_trie(self._field_to_path(path, key), type_index, len(data_list) - 1)

    def _unpack_field(self, path: str, key: str) -> str | None:
        try:
            return self.maybe_decode(
                cast(str, self._read_pathlike(self._field_to_path(path, key)))
            )
        except KeyError:
            return None

    def _read_pathlike(self, path: str) -> H5Scalar:
        # A real path or one with the field delimiter to find a metadata field
        type_index, position_index = self._read_trie(path)
        group_name = self._index_map.inverse[type_index]
        with self:
            value = cast(H5Scalar, self.file[group_name][position_index])
        return value

    def pack_empty(self, path: str) -> None:
        self._pack_trie(path, self._index_map["empty"], -1)

    def _pack_thing(
        self, obj: PackedThingType | ArrayType, type_name: str, path: str
    ) -> None:
        type_index = self._index_map[type_name]
        group = self._packed[type_index]
        group.append(obj)  # type: ignore[arg-type]
        self._pack_trie(path, type_index, len(group) - 1)

    def pack_string(self, obj: str, path: str) -> None:
        self._pack_thing(obj, "str", path)

    def unpack_string(self, path: str) -> str:
        return self.maybe_decode(cast(str, self._read_pathlike(path)))

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
        self._pack_trie(path, real_index, len(real_group) - 1)

    def unpack_complex(self, path: str) -> complex:
        _, position_index = self._read_trie(path)
        with self:
            value = complex(
                self.file["complex_real"][position_index],
                self.file["complex_imag"][position_index],
            )
        return value

    def pack_bytes(self, obj: bytes, path: str) -> None:
        self._pack_thing(obj, "bytes", path)

    def unpack_bytes(self, path: str) -> bytes:
        return cast(bytes, self._read_pathlike(path).tobytes())

    def pack_bytearray(self, obj: bytearray, path: str) -> None:
        self._pack_thing(obj, "bytearray", path)

    def unpack_bytearray(self, path: str) -> bytearray:
        return bytearray(self._read_pathlike(path))

    def create_group(self, path: str) -> None:
        self._pack_trie(path, self._index_map["group"], -1)

    def open_group(self, path: str) -> set[str]:
        prefix = PATH_DELIMITER + path
        subpaths = self.unpacked_trie.keys(prefix=prefix, shallow=False)
        next_depth_index = 1
        children = {
            part[next_depth_index]
            for key in subpaths
            if (part := key[len(prefix) :].split(PATH_DELIMITER, next_depth_index + 1))
            and len(part) > next_depth_index
            and self._field_delimiter not in part[next_depth_index]
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
        _, position_index = self._read_trie(path)
        with self:
            value = cast(ArrayType, self.file[f"ndarrays/i{position_index}"][:])
        return value
