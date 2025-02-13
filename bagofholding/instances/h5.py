from pathlib import Path
from typing import Any, TypeAlias, Union

import numpy as np
from h5py import Dataset, File, string_dtype
from h5py import Group as H5Group

from bagofholding.exception import BagOfHoldingError
from bagofholding.instances.bag import Bag
from bagofholding.instances.content import (
    ComplexItem,
    Content,
    DirectItem,
    Global,
    Group,
    Item,
    LongItem,
    NoneItem,
    Reducible,
    Reference,
    StrItem,
    minimal_dispatcher,
)
from bagofholding.metadata import Metadata

h5py_dtype_whitelist = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.bool_,
    np.bytes_,
    np.str_,
)

H5DtypeAlias: TypeAlias = Union[
    np.dtype[np.int8],
    np.dtype[np.int16],
    np.dtype[np.int32],
    np.dtype[np.int64],
    np.dtype[np.uint8],
    np.dtype[np.uint16],
    np.dtype[np.uint32],
    np.dtype[np.uint64],
    np.dtype[np.float16],
    np.dtype[np.float32],
    np.dtype[np.float64],
    np.dtype[np.complex64],
    np.dtype[np.complex128],
    np.dtype[np.bool_],
    np.dtype[np.bytes_],
    np.dtype[np.str_],
]


class ArrayItem(DirectItem[np.ndarray[tuple[int, ...], H5DtypeAlias]]):
    pass


def dispatch_array(obj: object) -> type[ArrayItem] | type[Reducible] | None:
    if type(obj) is np.ndarray:
        if obj.dtype in h5py_dtype_whitelist:
            return ArrayItem
        return Reducible
    return None


class InstanceH5Bag(Bag):

    def __init__(self, filepath: str | Path, *args: object, **kwargs: Any) -> None:
        super().__init__(filepath)
        self.file = File(filepath, mode="r")

    def _close(self) -> None:
        self.file.close()

    def __del__(self) -> None:
        self._close()

    @classmethod
    def dispatch(cls, obj: object) -> type[Content[Any, Any]] | None:
        con_class = minimal_dispatcher(obj)
        return dispatch_array(obj) if con_class is None else con_class

    @classmethod
    def _write(cls, content: Content[Any, Any], filename: Path) -> None:
        with File(filename, "w") as f:
            cls._write_entry(f, content)

    @classmethod
    def _write_entry(cls, file: File, content: Content[Any, Any]) -> None:
        if isinstance(content, Group):
            entry = cls._write_group(file, content)
        elif isinstance(content, Item):
            entry = cls._write_item(file, content)
        else:
            raise BagOfHoldingError(f"Expected a group or item, but got {content}")
        entry.attrs["content_type"] = (
            f"{content.__class__.__module__}.{content.__class__.__qualname__}"
        )
        if content.metadata is not None:
            cls._write_metadata(entry, content.metadata)

    @staticmethod
    def _write_metadata(entry: H5Group | Dataset, metadata: Metadata | None) -> None:
        if metadata is not None:
            for k, v in metadata.field_items():
                if v is not None:
                    entry.attrs[k] = v

    @classmethod
    def _write_group(
        cls, file: File, content: Group[Any, Any, Content[Any, Any]]
    ) -> H5Group:
        group = file.create_group(content.path)
        for subcontent in content.values():
            cls._write_entry(file, subcontent)
        return group

    @classmethod
    def _write_item(cls, file: File, item: Item[Any, Any, Any]) -> Dataset:
        if isinstance(item.stored, str):
            dataset = file.create_dataset(
                item.path, data=item.stored, dtype=string_dtype(encoding="utf-8")
            )
        elif item.stored is None:
            dataset = file.create_dataset(item.path, data=0)
        elif isinstance(item.stored, complex):
            dataset = file.create_dataset(
                item.path,
                data=np.array([item.stored.real, item.stored.imag]),
            )
        else:
            dataset = file.create_dataset(item.path, data=item.stored)
        return dataset

    def read_stored_item(self, item: Item[Any, Any, Any]) -> Any:
        entry = self.file[item.path]
        if isinstance(item, NoneItem):  # TODO: Is it getting overwritten?
            return None
        elif isinstance(item, (Global, Reference, StrItem)):  # TODO: Brittle...
            return entry[()].decode("utf-8")
        elif isinstance(item, ComplexItem):
            return complex(entry[0], entry[1])
        elif isinstance(item, LongItem):
            return int(entry[()])
        # H5 is casting everything to numpy datatypes
        # For, e.g., indices, I _need_ the original type
        # TODO: Treat type conversions more robustly
        else:
            return entry[()]

    def _maybe_decode(self, attr: str | bytes) -> Any:
        return attr if isinstance(attr, str) else attr.decode("utf-8")

    def __getitem__(self, path: str) -> Content[Any, Any]:
        entry = self.file[path]
        content = self._instantiate_content(
            self._maybe_decode(entry.attrs["content_type"]),
            path,
            self._read_metadata(entry),
        )
        if isinstance(content, Group):
            for subgroup in entry:
                content[subgroup] = self[content.relative(subgroup)]
        return content

    def _read_metadata(self, entry: H5Group | Dataset) -> Metadata | None:
        metadata = {}
        has_metadata = False
        for meta_key in Metadata.__dataclass_fields__:
            try:
                metadata[meta_key] = self._maybe_decode(entry.attrs[meta_key])
                has_metadata = True
            except KeyError:
                metadata[meta_key] = ""
        return Metadata(**metadata) if has_metadata else None

    def list_paths(self) -> list[str]:
        """A list of all available content paths."""
        paths: list[str] = []
        self.file.visit(paths.append)
        return paths
