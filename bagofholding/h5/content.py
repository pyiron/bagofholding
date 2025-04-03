from __future__ import annotations

import abc
import collections.abc
import operator
import types
from collections.abc import Callable, Iterator
from types import BuiltinFunctionType, FunctionType
from typing import Any, ClassVar, Generic, TypeAlias, TypeVar, cast

import bidict
import h5py
import numpy as np

from bagofholding.exception import BagOfHoldingError
from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST, H5DtypeAlias
from bagofholding.metadata import Metadata, get_metadata
from bagofholding.retrieve import import_from_string

PackingMemoAlias: TypeAlias = bidict.bidict[int, str]
ReferencesAlias: TypeAlias = list[object]
UnpackingMemoAlias: TypeAlias = dict[str, Any]


PackingType = TypeVar("PackingType", bound=Any)
UnpackingType = TypeVar("UnpackingType", bound=Any)


class NotData:
    pass


class Content(Generic[PackingType, UnpackingType], abc.ABC):

    @classmethod
    @abc.abstractmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> UnpackingType:
        # TODO: Optionally first read the metadata and verify that your env is viable
        pass

    @classmethod
    def _write_type(cls, entry: h5py.Group | h5py.Dataset) -> None:
        entry.attrs["content_type"] = cls.__module__ + "." + cls.__name__

    @staticmethod
    def _write_metadata(
        entry: h5py.Group | h5py.Dataset, metadata: Metadata | None
    ) -> None:
        if metadata is not None:
            for k, v in metadata.field_items():
                if v is not None:
                    entry.attrs[k] = v


class Item(
    Content[PackingType, UnpackingType], Generic[PackingType, UnpackingType], abc.ABC
):
    @classmethod
    @abc.abstractmethod
    def write_item(
        cls,
        obj: PackingType,
        file: h5py.File,
        path: str,
    ) -> None:
        pass


class Reference(Item[str, Any]):
    @classmethod
    def write_item(
        cls,
        obj: str,
        file: h5py.File,
        path: str,
    ) -> None:
        entry = file.create_dataset(
            path, data=obj, dtype=h5py.string_dtype(encoding="utf-8")
        )
        cls._write_type(entry)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> Any:
        reference = file[path][()].decode("utf-8")
        from_memo = memo.get(reference, NotData)
        if from_memo is not NotData:
            return from_memo
        else:
            return unpack(file, reference, memo)


GlobalType: TypeAlias = type[type] | FunctionType | str


class Global(Item[GlobalType, Any]):
    @classmethod
    def write_item(
        cls,
        obj: GlobalType,
        file: h5py.File,
        path: str,
    ) -> None:
        value: str
        if isinstance(obj, str):
            value = "builtins." + obj if "." not in obj else obj
        else:
            value = obj.__module__ + "." + obj.__qualname__
        entry = file.create_dataset(
            path, data=value, dtype=h5py.string_dtype(encoding="utf-8")
        )
        cls._write_type(entry)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> Any:
        import_string = file[path][()].decode("utf-8")
        return import_from_string(import_string)


class NoneItem(Item[type[None], None]):
    @classmethod
    def write_item(
        cls,
        obj: type[None],
        file: h5py.File,
        path: str,
    ) -> None:
        entry = file.create_dataset(path, data=h5py.Empty(dtype="f"))
        cls._write_type(entry)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> None:
        return None


ItemType = TypeVar("ItemType", bound=Any)


class SimpleItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    pass


class Complex(SimpleItem[complex]):
    @classmethod
    def write_item(
        cls,
        obj: complex,
        file: h5py.File,
        path: str,
    ) -> None:
        entry = file.create_dataset(path, data=np.array([obj.real, obj.imag]))
        cls._write_type(entry)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> complex:
        entry = file[path]
        return complex(entry[0], entry[1])


class Str(SimpleItem[str]):
    @classmethod
    def write_item(
        cls,
        obj: str,
        file: h5py.File,
        path: str,
    ) -> None:
        entry = file.create_dataset(
            path, data=obj, dtype=h5py.string_dtype(encoding="utf-8")
        )
        cls._write_type(entry)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> str:
        return cast(str, file[path][()].decode("utf-8"))


class NativeItem(SimpleItem[ItemType], Generic[ItemType], abc.ABC):
    recast: type[ItemType]

    @classmethod
    def write_item(
        cls,
        obj: ItemType,
        file: h5py.File,
        path: str,
    ) -> None:
        entry = file.create_dataset(path, data=obj)
        cls._write_type(entry)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> ItemType:
        return cast(ItemType, cls.recast(file[path][()]))


class Bool(NativeItem[bool]):
    recast = bool


class Long(NativeItem[int]):
    recast = int


class Float(NativeItem[float]):
    recast = float


class Bytes(NativeItem[bytes]):
    recast = bytes


class Bytearray(NativeItem[bytearray]):
    recast = bytearray


class ComplexItem(SimpleItem[ItemType], Generic[ItemType], abc.ABC):
    @classmethod
    def write_item(
        cls,
        obj: ItemType,
        file: h5py.File,
        path: str,
    ) -> None:
        entry = cls._write_item(obj, file, path)
        cls._write_type(entry)
        cls._write_metadata(entry, get_metadata(obj))

    @classmethod
    @abc.abstractmethod
    def _write_item(
        cls,
        obj: ItemType,
        file: h5py.File,
        path: str,
    ) -> h5py.Dataset:
        pass


class Array(ComplexItem[np.ndarray[tuple[int, ...], H5DtypeAlias]]):
    @classmethod
    def _write_item(
        cls,
        obj: np.ndarray[tuple[int, ...], H5DtypeAlias],
        file: h5py.File,
        path: str,
    ) -> h5py.Dataset:
        return file.create_dataset(path, data=obj)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> np.ndarray[tuple[int, ...], H5DtypeAlias]:
        return cast(np.ndarray[tuple[int, ...], H5DtypeAlias], file[path][()])


class Group(
    Content[PackingType, UnpackingType], Generic[PackingType, UnpackingType], abc.ABC
):
    @classmethod
    @abc.abstractmethod
    def write_group(
        cls,
        obj: PackingType,
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
        **kwargs: Any,
    ) -> None:
        pass


# __reduce__ return values
# per https://docs.python.org/3/library/pickle.html#object.__reduce__
ConstructorType: TypeAlias = Callable[..., object]
ConstructorArgsType: TypeAlias = tuple[object, ...]
StateType: TypeAlias = object
ListItemsType: TypeAlias = Iterator[object]
DictItemsType: TypeAlias = Iterator[tuple[object, object]]
SetStateCallableType: TypeAlias = Callable[[object, object], None]
ReduceReturnType: TypeAlias = (
    tuple[ConstructorType, ConstructorArgsType]
    | tuple[ConstructorType, ConstructorArgsType, StateType | None]
    | tuple[
        ConstructorType, ConstructorArgsType, StateType | None, ListItemsType | None
    ]
    | tuple[
        ConstructorType,
        ConstructorArgsType,
        StateType | None,
        ListItemsType | None,
        DictItemsType | None,
    ]
    | tuple[
        ConstructorType | None,
        ConstructorArgsType | None,
        StateType | None,
        ListItemsType | None,
        DictItemsType | None,
        SetStateCallableType | None,
    ]
)
PickleHint: TypeAlias = str | tuple[Any, ...]


class Reducible(Group[object, object]):

    reduction_fields: ClassVar[tuple[str, str, str, str, str, str]] = (
        "constructor",
        "args",
        "state",
        "item_iterator",
        "kv_iterator",
        "setter",
    )

    @classmethod
    def write_group(
        cls,
        obj: object,
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
        reduced_value: ReduceReturnType | PickleHint | None = None,
        **kwargs: Any,
    ) -> None:
        reduced_value = obj.__reduce__() if reduced_value is None else reduced_value
        entry = file.create_group(path)
        cls._write_type(entry)
        cls._write_metadata(entry, get_metadata(obj))
        for subpath, value in zip(cls.reduction_fields, reduced_value, strict=False):
            pack(value, file, relative(path, subpath), memo, references)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> object:
        constructor = cast(
            ConstructorType, unpack(file, relative(path, "constructor"), memo)
        )
        constructor_args = cast(
            ConstructorArgsType, unpack(file, relative(path, "args"), memo)
        )
        obj: object = constructor(*constructor_args)
        memo[path] = obj
        rv = (constructor, constructor_args) + tuple(
            unpack(file, relative(path, k), memo)
            for k in cls.reduction_fields[2 : len(file[path])]
        )
        n_items = len(rv)
        if n_items >= 3 and rv[2] is not None:
            if n_items == 6 and rv[5] is not None:
                cast(SetStateCallableType, rv[5])(obj, rv[2])
            elif hasattr(obj, "__setstate__"):
                obj.__setstate__(rv[2])
            else:
                # "If the object has no such method then, the value must be a dictionary"
                obj.__dict__.update(cast(dict[Any, Any], rv[2]))
        if n_items >= 4 and rv[3] is not None:
            if hasattr(obj, "append"):
                for item in cast(ListItemsType, rv[3]):
                    obj.append(item)
            elif hasattr(obj, "extend"):
                obj.extend(list(cast(ListItemsType, rv[3])))
                # TODO: look into efficiency choices for optional usage of extend even
                #  when append exists
            else:
                raise AttributeError(f"{obj} has neither append nor extend methods")
        if n_items >= 5 and rv[4] is not None and hasattr(obj, "__setitem__"):
            for k, v in cast(DictItemsType, rv[4]):
                obj[k] = v

        return obj


GroupType = TypeVar("GroupType", bound=Any)  # Bind to container?


class SimpleGroup(Group[GroupType, GroupType], Generic[GroupType], abc.ABC):
    @classmethod
    def write_group(
        cls,
        obj: PackingType,
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
        **kwargs: Any,
    ) -> None:
        entry = file.create_group(path)
        cls._write_type(entry)
        cls._write_subcontent(obj, file, path, memo, references)

    @classmethod
    @abc.abstractmethod
    def _write_subcontent(
        cls,
        obj: PackingType,
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> h5py.Group:
        pass


class Dict(SimpleGroup[dict[Any, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[Any, Any],
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> None:
        pack(tuple(obj.keys()), file, relative(path, "keys"), memo, references)
        pack(tuple(obj.values()), file, relative(path, "values"), memo, references)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> dict[Any, Any]:
        return dict(
            zip(
                cast(tuple[Any], unpack(file, relative(path, "keys"), memo)),
                cast(tuple[Any], unpack(file, relative(path, "values"), memo)),
                strict=True,
            )
        )


class StrKeyDict(SimpleGroup[dict[str, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[Any, Any],
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> None:
        for k, v in obj.items():
            pack(v, file, relative(path, k), memo, references)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> dict[Any, Any]:
        return {k: unpack(file, relative(path, k), memo) for k in file[path]}


class Union(SimpleGroup[types.UnionType]):
    """
    :class:`types.UnionType` has no :meth:`__reduce__` method. Pickle actually gets
    around this with bespoke logic, and so we need to too.
    """

    @classmethod
    def _write_subcontent(
        cls,
        obj: types.UnionType,
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> None:
        for i, v in enumerate(obj.__args__):
            pack(v, file, relative(path, f"i{i}"), memo, references)

    @staticmethod
    def _recursive_or(args: collections.abc.Iterable[object]) -> types.UnionType:
        it = iter(args)
        try:
            first = next(it)
            second = next(it)
        except StopIteration:
            raise ValueError("Expected at least two elements for a UnionType") from None

        union: types.UnionType = operator.or_(first, second)

        for arg in it:
            union = operator.or_(union, arg)

        return union

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> types.UnionType:
        return cls._recursive_or(
            unpack(file, relative(path, f"i{i}"), memo) for i in range(len(file[path]))
        )


IndexableType = TypeVar(
    "IndexableType", tuple[Any, ...], list[Any], set[Any], frozenset[Any]
)


class Indexable(SimpleGroup[IndexableType], Generic[IndexableType], abc.ABC):
    recast: type[IndexableType]

    @classmethod
    def _write_subcontent(
        cls,
        obj: IndexableType,
        file: h5py.File,
        path: str,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> None:
        for i, v in enumerate(obj):
            pack(v, file, relative(path, f"i{i}"), memo, references)

    @classmethod
    def read(
        cls,
        file: h5py.File,
        path: str,
        memo: UnpackingMemoAlias,
    ) -> IndexableType:
        return cls.recast(
            unpack(file, relative(path, f"i{i}"), memo) for i in range(len(file[path]))
        )


class Tuple(Indexable[tuple[Any, ...]]):
    recast = tuple


class List(Indexable[list[Any]]):
    recast = list


class Set(Indexable[set[Any]]):
    recast = set


class FrozenSet(Indexable[frozenset[Any]]):
    recast = frozenset


def pack(
    obj: object,
    file: h5py.File,
    path: str,
    memo: PackingMemoAlias,
    references: ReferencesAlias,
) -> None:
    t = type if isinstance(obj, type) else type(obj)
    simple_class = KNOWN_ITEM_MAP.get(t)
    if simple_class is not None:
        simple_class.write_item(obj, file, path)
        return

    obj_id = id(obj)
    reference = memo.get(obj_id)
    if reference is not None:
        Reference.write_item(reference, file, path)
        return
    else:
        memo[obj_id] = path
        references.append(obj)

    complex_class = get_complex_content_class(obj)
    if complex_class is not None:
        complex_class.write_item(obj, file, path)
        return

    group_class = get_group_content_class(obj)
    if group_class is not None:
        group_class.write_group(obj, file, path, memo, references)
        return

    rv = obj.__reduce__()  # TODO: handle __reduce_ex__ for pickle compliance
    if isinstance(rv, str):
        Global.write_item(
            _get_importable_string_from_string_reduction(rv, obj),
            file,
            path,
        )
        return
    else:
        Reducible.write_group(obj, file, path, memo, references, reduced_value=rv)
        return


def _get_importable_string_from_string_reduction(
    string_reduction: str, reduced_object: object
) -> str:
    """
    Per the pickle docs:

    > If a string is returned, the string should be interpreted as the name of a global
      variable. It should be the object’s local name relative to its module; the pickle
      module searches the module namespace to determine the object’s module. This
      behaviour is typically useful for singletons.

    To then import such an object from a non-local caller, we try scoping the string
    with the module of the object which returned it.
    """
    try:
        import_from_string(string_reduction)
        importable = string_reduction
    except ModuleNotFoundError:
        importable = reduced_object.__module__ + "." + string_reduction
        try:
            import_from_string(importable)
        except (ModuleNotFoundError, AttributeError) as e:
            raise BagOfHoldingError(
                f"Couldn't import {string_reduction} after scoping it as {importable}. "
                f"Please contact the developers so we can figure out how to handle "
                f"this edge case."
            ) from e
    return importable


KNOWN_ITEM_MAP: dict[
    type | FunctionType | BuiltinFunctionType, type[Item[Any, Any]]
] = {
    type: Global,
    FunctionType: Global,
    type(all): Global,
    type(None): NoneItem,
    bool: Bool,
    int: Long,
    float: Float,
    complex: Complex,
    bytes: Bytes,
    bytearray: Bytearray,
    str: Str,
}


def get_complex_content_class(obj: object) -> type[ComplexItem[Any]] | None:
    if type(obj) is np.ndarray and obj.dtype in H5PY_DTYPE_WHITELIST:
        return Array
    return None


KNOWN_GROUP_MAP: dict[type, type[Group[Any, Any]]] = {
    dict: Dict,
    types.UnionType: Union,
    tuple: Tuple,
    list: List,
    set: Set,
    frozenset: FrozenSet,
}


def get_group_content_class(obj: object) -> type[Group[Any, Any]] | None:
    t = type(obj)
    if t is dict and all(isinstance(k, str) for k in cast(dict[str, Any], obj)):
        return StrKeyDict

    return KNOWN_GROUP_MAP.get(t)


def unpack(
    file: h5py.File,
    path: str,
    memo: UnpackingMemoAlias,
) -> object:
    memo_value = memo.get(path, NotData)
    if memo_value is NotData:
        content_class_string = maybe_decode(file[path].attrs["content_type"])
        content_class = import_from_string(content_class_string)
        value = content_class.read(file, path, memo)
        if path not in memo:
            memo[path] = value
        return value
    return memo_value


def read_metadata(entry: h5py.Group | h5py.Dataset) -> Metadata | None:
    metadata = {}
    has_metadata = False
    for meta_key in Metadata.__dataclass_fields__:
        try:
            metadata[meta_key] = maybe_decode(entry.attrs[meta_key])
            has_metadata = True
        except KeyError:
            metadata[meta_key] = ""
    return Metadata(**metadata) if has_metadata else None


def maybe_decode(attr: str | bytes) -> str:
    return attr if isinstance(attr, str) else attr.decode("utf-8")


PATH_DELIMITER = "/"


def relative(path: str, subpath: str) -> str:
    return path + PATH_DELIMITER + subpath
