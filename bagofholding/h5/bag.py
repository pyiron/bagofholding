from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, ClassVar, Self, cast

import h5py
import numpy as np

from bagofholding.bag import PATH_DELIMITER, Bag, BagInfo
from bagofholding.content import BespokeItem, has_surrogates
from bagofholding.exceptions import BagMismatchError, NotAGroupError
from bagofholding.h5 import format_versions
from bagofholding.h5.content import Array, ArrayPacker, ArrayType, int_overflows
from bagofholding.h5.context import HasH5FileContext
from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST
from bagofholding.metadata import Metadata, VersionScrapingMap, VersionValidatorType


@dataclasses.dataclass(frozen=True)
class H5Info(BagInfo):
    libver_str: str | None = dataclasses.field(default="latest", compare=False)


class H5Bag(Bag, HasH5FileContext, ArrayPacker):
    """
    A bag using HDF5 files based on `h5py`.

    The underlying file structure is directly representative of the structure of the
    decomposed object being stored, and `attrs` are used to store metadata.

    ``save`` accepts a ``libver`` option (default :attr:`max_libver`) controlling
    the HDF5 file-format version written, which is also stored in the bag info.
    Only codes up to :attr:`max_libver` are accepted; ``"latest"`` is not.
    """

    _content_key: ClassVar[str] = "content_type"

    @classmethod
    def get_bag_info(cls) -> BagInfo:
        return H5Info(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=cls.get_version(),
            libver_str=cls.max_libver,
        )

    @classmethod
    def _bag_info_class(cls) -> type[BagInfo]:
        return H5Info

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        self._file = None
        self._context_depth = 0
        self._parsed_path = None
        self._working_root = None
        self._write_libver = self.max_libver
        super().__init__(filepath, *args, **kwargs)

    def _load_existing_bag_info(self) -> BagInfo | None:
        file_path, group_path = self._parse_path()
        if not file_path.is_file():
            return None
        self._file = h5py.File(file_path, "r")
        try:
            if group_path != "/" and group_path not in self._file:
                return None
            self._context_depth = 1
            try:
                info = self._unpack_bag_info()
            except (KeyError, OSError, RuntimeError) as e:
                raise BagMismatchError(
                    f"Could not read the bag metadata at {self.filepath}; it may "
                    f"have been written by a newer bagofholding/HDF5. Update "
                    f"bagofholding (and h5py)."
                ) from e
            finally:
                self._context_depth = 0
            return info if info.qualname is not None else None
        finally:
            self.close()

    def _validate_loaded_bag_info(
        self, info: BagInfo, bag_version_validator: VersionValidatorType
    ) -> None:
        format_versions.check_hdf5_supports(self.max_libver)
        if bag_version_validator != "none":
            format_versions.check_stored_libver(
                cast(H5Info, info).libver_str, self.max_libver
            )

    @classmethod
    def _new_for_save(
        cls,
        filepath: str | pathlib.Path,
        overwrite_existing: bool,
        libver: str | None = None,
        **backend_options: Any,
    ) -> Self:
        if backend_options:
            raise TypeError(
                f"Unexpected save option(s) for {cls.__name__}: "
                f"{sorted(backend_options)}"
            )
        format_versions.check_hdf5_supports(cls.max_libver)
        resolved = format_versions.resolve_save_libver(libver, cls.max_libver)
        bag = cls(filepath, _skip_load=True)
        bag._write_libver = resolved
        bag._open_for_write(overwrite_existing)
        return bag

    def _pack_bag_info(self) -> None:
        info = dataclasses.replace(
            cast(H5Info, self.get_bag_info()), libver_str=self._write_libver
        )
        self._pack_fields(info, PATH_DELIMITER)

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

    def __enter__(self) -> Self:
        self._context_depth += 1
        if self._file is None:
            self.open("r")
        return self

    def _resolve(self, path: str) -> h5py.Group | h5py.Dataset:
        return self.file if path in ("/", "") else self.file[path]

    def _pack_field(self, path: str, key: str, value: str) -> None:
        self._resolve(path).attrs[key] = value

    def _unpack_field(self, path: str, key: str) -> str | None:
        try:
            attrs = self._resolve(path).attrs
        except KeyError:
            return None
        if key not in attrs:
            return None
        return self.maybe_decode(attrs[key])

    def pack_empty(self, path: str) -> None:
        self.file.create_dataset(path, data=h5py.Empty(dtype="f"))

    def pack_string(self, obj: str, path: str) -> None:
        if has_surrogates(obj):
            encoded = obj.encode("utf-16", errors="surrogatepass")
            self.file.create_dataset(path, data=np.void(encoded))
            self.file[path].attrs["_surrogate_str"] = True
        else:
            self.file.create_dataset(
                path, data=obj, dtype=h5py.string_dtype(encoding="utf-8")
            )

    def unpack_string(self, path: str) -> str:
        if self.file[path].attrs.get("_surrogate_str", False):
            return cast(
                str,
                self.file[path][()].tobytes().decode("utf-16", errors="surrogatepass"),
            )
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
        if int_overflows(obj):
            self.file.create_dataset(
                path, data=str(obj), dtype=h5py.string_dtype(encoding="utf-8")
            )
            self.file[path].attrs["_bigint"] = True
        else:
            self._pack_raw(obj, path)

    def unpack_long(self, path: str) -> int:
        if self.file[path].attrs.get("_bigint", False):
            return int(self._unpack_raw(path).decode("utf-8"))
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
        if obj == b"":
            special = h5py.special_dtype(vlen=bytes)
            self.file.create_dataset(path, data=b"", dtype=special)
        else:
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
    def get_bespoke_content_class(
        self, obj: object
    ) -> type[BespokeItem[Any, Self]] | None:
        if type(obj) is np.ndarray and obj.dtype.type in H5PY_DTYPE_WHITELIST:
            return cast(type[BespokeItem[Any, Self]], Array)
        return None

    def pack_array(self, obj: ArrayType, path: str) -> None:
        self.file.create_dataset(path, data=obj)

    def unpack_array(self, path: str) -> ArrayType:
        return cast(ArrayType, self.file[path][()])
