from __future__ import annotations

import abc
import collections.abc
import dataclasses
import operator
import pickle
import types
from collections.abc import Callable, Iterator
from types import BuiltinFunctionType, FunctionType
from typing import Any, ClassVar, Generic, SupportsIndex, TypeAlias, TypeVar, cast

import bidict
import h5py
import numpy as np

from bagofholding.exception import BagOfHoldingError
from bagofholding.h5.dtypes import H5PY_DTYPE_WHITELIST, H5DtypeAlias
from bagofholding.metadata import (
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_metadata,
    validate_version,
)
from bagofholding.retrieve import import_from_string

PackingMemoAlias: TypeAlias = bidict.bidict[int, str]
ReferencesAlias: TypeAlias = list[object]
UnpackingMemoAlias: TypeAlias = dict[str, Any]


PackingType = TypeVar("PackingType", bound=Any)
UnpackingType = TypeVar("UnpackingType", bound=Any)

PATH_DELIMITER = "/"


@dataclasses.dataclass
class Location:
    file: h5py.File
    path: str

    def relative_path(self, subpath: str) -> str:
        return self.path + PATH_DELIMITER + subpath

    @property
    def entry(self) -> h5py.Group | h5py.Dataset:
        return self.file[self.path]

    def create_dataset(self, **kwargs: Any) -> h5py.Dataset:
        return self.file.create_dataset(self.path, **kwargs)


@dataclasses.dataclass
class GroupPackingArguments:
    loc: Location
    memo: PackingMemoAlias
    references: ReferencesAlias
    version_scraping: VersionScrapingMap | None
    _pickle_protocol: SupportsIndex


@dataclasses.dataclass
class UnpackingArguments:
    loc: Location
    memo: UnpackingMemoAlias
    version_validator: VersionValidatorType
    version_scraping: VersionScrapingMap | None


class NotData:
    pass


class Content(Generic[PackingType, UnpackingType], abc.ABC):

    @classmethod
    @abc.abstractmethod
    def read(cls, unpacking_args: UnpackingArguments) -> UnpackingType:
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
    def write_item(cls, obj: PackingType, loc: Location) -> None:
        pass


class Reference(Item[str, Any]):
    @classmethod
    def write_item(cls, obj: str, loc: Location) -> None:
        entry = loc.create_dataset(data=obj, dtype=h5py.string_dtype(encoding="utf-8"))
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> Any:
        reference = unpacking_args.loc.entry[()].decode("utf-8")
        from_memo = unpacking_args.memo.get(reference, NotData)
        if from_memo is not NotData:
            return from_memo
        else:
            return unpack(
                unpacking_args.loc.file,
                reference,
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            )


GlobalType: TypeAlias = type[type] | FunctionType | str


class Global(Item[GlobalType, Any]):
    @classmethod
    def write_item(cls, obj: GlobalType, loc: Location) -> None:
        value: str
        if isinstance(obj, str):
            value = "builtins." + obj if "." not in obj else obj
        else:
            value = obj.__module__ + "." + obj.__qualname__
        entry = loc.create_dataset(
            data=value, dtype=h5py.string_dtype(encoding="utf-8")
        )
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> Any:
        import_string = unpacking_args.loc.entry[()].decode("utf-8")
        return import_from_string(import_string)


class NoneItem(Item[type[None], None]):
    @classmethod
    def write_item(cls, obj: type[None], loc: Location) -> None:
        entry = loc.create_dataset(data=h5py.Empty(dtype="f"))
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> None:
        return None


ItemType = TypeVar("ItemType", bound=Any)


class SimpleItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    pass


class Complex(SimpleItem[complex]):
    @classmethod
    def write_item(cls, obj: complex, loc: Location) -> None:
        entry = loc.create_dataset(data=np.array([obj.real, obj.imag]))
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> complex:
        entry = unpacking_args.loc.entry
        return complex(entry[0], entry[1])


class Str(SimpleItem[str]):
    @classmethod
    def write_item(cls, obj: str, loc: Location) -> None:
        entry = loc.create_dataset(data=obj, dtype=h5py.string_dtype(encoding="utf-8"))
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> str:
        return cast(str, unpacking_args.loc.entry[()].decode("utf-8"))


class Bytes(SimpleItem[bytes]):
    @classmethod
    def write_item(cls, obj: bytes, loc: Location) -> None:
        entry = loc.create_dataset(data=np.void(obj))
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> bytes:
        return bytes(unpacking_args.loc.entry[()])


class NativeItem(SimpleItem[ItemType], Generic[ItemType], abc.ABC):
    recast: type[ItemType]

    @classmethod
    def write_item(cls, obj: ItemType, loc: Location) -> None:
        entry = loc.create_dataset(data=obj)
        cls._write_type(entry)

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> ItemType:
        return cast(ItemType, cls.recast(unpacking_args.loc.entry[()]))


class Bool(NativeItem[bool]):
    recast = bool


class Long(NativeItem[int]):
    recast = int


class Float(NativeItem[float]):
    recast = float


class Bytearray(NativeItem[bytearray]):
    recast = bytearray


class ComplexItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    @classmethod
    def write_item(
        cls,
        obj: ItemType,
        loc: Location,
        version_scraping: VersionScrapingMap | None = None,
    ) -> None:
        entry = cls._write_item(obj, loc)
        cls._write_type(entry)
        cls._write_metadata(
            entry,
            get_metadata(obj, {} if version_scraping is None else version_scraping),
        )

    @classmethod
    @abc.abstractmethod
    def _write_item(cls, obj: ItemType, loc: Location) -> h5py.Dataset:
        pass


class Array(ComplexItem[np.ndarray[tuple[int, ...], H5DtypeAlias]]):
    @classmethod
    def _write_item(
        cls,
        obj: np.ndarray[tuple[int, ...], H5DtypeAlias],
        loc: Location,
    ) -> h5py.Dataset:
        return loc.create_dataset(data=obj)

    @classmethod
    def read(
        cls,
        unpacking_args: UnpackingArguments,
    ) -> np.ndarray[tuple[int, ...], H5DtypeAlias]:
        return cast(
            np.ndarray[tuple[int, ...], H5DtypeAlias],
            unpacking_args.loc.entry[()],
        )


class Group(
    Content[PackingType, UnpackingType], Generic[PackingType, UnpackingType], abc.ABC
):
    @classmethod
    @abc.abstractmethod
    def write_group(
        cls,
        obj: PackingType,
        packing_args: GroupPackingArguments,
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
        packing_args: GroupPackingArguments,
        **kwargs: Any,
    ) -> None:
        try:
            reduced_value = obj.__reduce_ex__(pickle.DEFAULT_PROTOCOL)
        except AttributeError:
            reduced_value = obj.__reduce__() if reduced_value is None else reduced_value
        entry = packing_args.loc.file.create_group(packing_args.loc.path)
        cls._write_type(entry)
        cls._write_metadata(
            entry,
            get_metadata(
                obj,
                (
                    {}
                    if packing_args.version_scraping is None
                    else packing_args.version_scraping
                ),
            ),
        )
        for subpath, value in zip(cls.reduction_fields, reduced_value, strict=False):
            pack(
                value,
                packing_args.loc.file,
                packing_args.loc.relative_path(subpath),
                packing_args.memo,
                packing_args.references,
                version_scraping=packing_args.version_scraping,
                _pickle_protocol=packing_args._pickle_protocol,
            )

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> object:
        constructor = cast(
            ConstructorType,
            unpack(
                unpacking_args.loc.file,
                unpacking_args.loc.relative_path("constructor"),
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            ),
        )
        constructor_args = cast(
            ConstructorArgsType,
            unpack(
                unpacking_args.loc.file,
                unpacking_args.loc.relative_path("args"),
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            ),
        )
        obj: object = constructor(*constructor_args)
        unpacking_args.memo[unpacking_args.loc.path] = obj
        rv = (constructor, constructor_args) + tuple(
            unpack(
                unpacking_args.loc.file,
                unpacking_args.loc.relative_path(k),
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            )
            for k in cls.reduction_fields[2 : len(unpacking_args.loc.entry)]
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
        packing_args: GroupPackingArguments,
        **kwargs: Any,
    ) -> None:
        entry = packing_args.loc.file.create_group(packing_args.loc.path)
        cls._write_type(entry)
        cls._write_subcontent(obj, packing_args)

    @classmethod
    @abc.abstractmethod
    def _write_subcontent(
        cls,
        obj: PackingType,
        packing_args: GroupPackingArguments,
    ) -> h5py.Group:
        pass


class Dict(SimpleGroup[dict[Any, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[Any, Any],
        packing_args: GroupPackingArguments,
    ) -> None:
        pack(
            tuple(obj.keys()),
            packing_args.loc.file,
            packing_args.loc.relative_path("keys"),
            packing_args.memo,
            packing_args.references,
            version_scraping=packing_args.version_scraping,
            _pickle_protocol=packing_args._pickle_protocol,
        )
        pack(
            tuple(obj.values()),
            packing_args.loc.file,
            packing_args.loc.relative_path("values"),
            packing_args.memo,
            packing_args.references,
            version_scraping=packing_args.version_scraping,
            _pickle_protocol=packing_args._pickle_protocol,
        )

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> dict[Any, Any]:
        return dict(
            zip(
                cast(
                    tuple[Any],
                    unpack(
                        unpacking_args.loc.file,
                        unpacking_args.loc.relative_path("keys"),
                        unpacking_args.memo,
                        version_validator=unpacking_args.version_validator,
                        version_scraping=unpacking_args.version_scraping,
                    ),
                ),
                cast(
                    tuple[Any],
                    unpack(
                        unpacking_args.loc.file,
                        unpacking_args.loc.relative_path("values"),
                        unpacking_args.memo,
                        version_validator=unpacking_args.version_validator,
                        version_scraping=unpacking_args.version_scraping,
                    ),
                ),
                strict=True,
            )
        )


class StrKeyDict(SimpleGroup[dict[str, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[str, Any],
        packing_args: GroupPackingArguments,
    ) -> None:
        for k, v in obj.items():
            pack(
                v,
                packing_args.loc.file,
                packing_args.loc.relative_path(k),
                packing_args.memo,
                packing_args.references,
                version_scraping=packing_args.version_scraping,
                _pickle_protocol=packing_args._pickle_protocol,
            )

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> dict[str, Any]:
        return {
            k: unpack(
                unpacking_args.loc.file,
                unpacking_args.loc.relative_path(k),
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            )
            for k in unpacking_args.loc.entry
        }


class Union(SimpleGroup[types.UnionType]):
    """
    :class:`types.UnionType` has no :meth:`__reduce__` method. Pickle actually gets
    around this with bespoke logic, and so we need to too.
    """

    @classmethod
    def _write_subcontent(
        cls,
        obj: types.UnionType,
        packing_args: GroupPackingArguments,
    ) -> None:
        for i, v in enumerate(obj.__args__):
            pack(
                v,
                packing_args.loc.file,
                packing_args.loc.relative_path(f"i{i}"),
                packing_args.memo,
                packing_args.references,
                version_scraping=packing_args.version_scraping,
                _pickle_protocol=packing_args._pickle_protocol,
            )

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
    def read(cls, unpacking_args: UnpackingArguments) -> types.UnionType:
        return cls._recursive_or(
            unpack(
                unpacking_args.loc.file,
                unpacking_args.loc.relative_path(f"i{i}"),
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            )
            for i in range(len(unpacking_args.loc.entry))
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
        packing_args: GroupPackingArguments,
    ) -> None:
        for i, v in enumerate(obj):
            pack(
                v,
                packing_args.loc.file,
                packing_args.loc.relative_path(f"i{i}"),
                packing_args.memo,
                packing_args.references,
                version_scraping=packing_args.version_scraping,
                _pickle_protocol=packing_args._pickle_protocol,
            )

    @classmethod
    def read(cls, unpacking_args: UnpackingArguments) -> IndexableType:
        return cls.recast(
            unpack(
                unpacking_args.loc.file,
                unpacking_args.loc.relative_path(f"i{i}"),
                unpacking_args.memo,
                version_validator=unpacking_args.version_validator,
                version_scraping=unpacking_args.version_scraping,
            )
            for i in range(len(unpacking_args.loc.entry))
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
    version_scraping: VersionScrapingMap | None,
    _pickle_protocol: SupportsIndex = pickle.HIGHEST_PROTOCOL,
) -> None:
    loc = Location(file=file, path=path)
    packing_args = GroupPackingArguments(
        loc=loc,
        memo=memo,
        references=references,
        version_scraping=version_scraping,
        _pickle_protocol=_pickle_protocol,
    )

    t = type if isinstance(obj, type) else type(obj)
    simple_class = KNOWN_ITEM_MAP.get(t)
    if simple_class is not None:
        simple_class.write_item(obj, loc)
        return

    obj_id = id(obj)
    reference = memo.get(obj_id)
    if reference is not None:
        Reference.write_item(reference, loc)
        return
    else:
        memo[obj_id] = path
        references.append(obj)

    complex_class = get_complex_content_class(obj)
    if complex_class is not None:
        complex_class.write_item(obj, loc, version_scraping=version_scraping)
        return

    group_class = get_group_content_class(obj)
    if group_class is not None:
        group_class.write_group(obj, packing_args)
        return

    try:
        rv = obj.__reduce_ex__(_pickle_protocol)
    except AttributeError:
        rv = obj.__reduce__()
    if isinstance(rv, str):
        Global.write_item(_get_importable_string_from_string_reduction(rv, obj), loc)
        return
    else:
        Reducible.write_group(obj, packing_args)
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
    version_validator: VersionValidatorType,
    version_scraping: VersionScrapingMap | None,
) -> object:
    memo_value = memo.get(path, NotData)
    if memo_value is NotData:
        entry = file[path]
        content_class_string = maybe_decode(entry.attrs["content_type"])
        content_class = import_from_string(content_class_string)
        metadata = read_metadata(entry)
        if metadata is not None:
            validate_version(
                metadata, validator=version_validator, version_scraping=version_scraping
            )
        value = content_class.read(
            UnpackingArguments(
                loc=Location(file=file, path=path),
                memo=memo,
                version_validator=version_validator,
                version_scraping=version_scraping,
            )
        )
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
