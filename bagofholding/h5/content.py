from __future__ import annotations

import abc
import collections.abc
import dataclasses
import operator
import pickle
import types
from collections.abc import Callable, Iterator
from types import BuiltinFunctionType, FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
)

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
from bagofholding.retrieve import (
    get_importable_string_from_string_reduction,
    import_from_string,
)

if TYPE_CHECKING:
    from bagofholding.h5.bag import H5Bag

PackingMemoAlias: TypeAlias = bidict.bidict[int, str]
ReferencesAlias: TypeAlias = list[object]
UnpackingMemoAlias: TypeAlias = dict[str, Any]


PackingType = TypeVar("PackingType", bound=Any)
UnpackingType = TypeVar("UnpackingType", bound=Any)


@dataclasses.dataclass
class Location:
    bag: H5Bag
    path: str

    @property
    def entry(self) -> h5py.Group | h5py.Dataset:
        return self.bag.file[self.path]

    def create_dataset(self, **kwargs: Any) -> h5py.Dataset:
        return self.bag.file.create_dataset(self.path, **kwargs)

    def create_group(self) -> h5py.Group:
        return self.bag.file.create_group(self.path)


@dataclasses.dataclass
class PackingArguments:
    memo: PackingMemoAlias
    references: ReferencesAlias
    require_versions: bool
    forbidden_modules: list[str] | tuple[str, ...]
    version_scraping: VersionScrapingMap | None
    _pickle_protocol: SupportsIndex


@dataclasses.dataclass
class UnpackingArguments:
    memo: UnpackingMemoAlias
    version_validator: VersionValidatorType
    version_scraping: VersionScrapingMap | None


class NotData:
    pass


class Content(Generic[PackingType, UnpackingType], abc.ABC):

    @classmethod
    @abc.abstractmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> UnpackingType:
        # TODO: Optionally first read the metadata and verify that your env is viable
        pass

    @classmethod
    @abc.abstractmethod
    def write(
        cls, obj: PackingType, location: Location, packing: PackingArguments
    ) -> None:
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
    pass


class Reference(Item[str, Any]):
    @classmethod
    def write(cls, obj: str, location: Location, packing: PackingArguments) -> None:
        entry = location.create_dataset(
            data=obj, dtype=h5py.string_dtype(encoding="utf-8")
        )
        cls._write_type(entry)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> Any:
        reference = location.entry[()].decode("utf-8")
        from_memo = unpacking.memo.get(reference, NotData)
        if from_memo is not NotData:
            return from_memo
        else:
            return unpack(
                location.bag,
                reference,
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )


GlobalType: TypeAlias = type[type] | FunctionType | str


class Global(Item[GlobalType, Any]):
    @classmethod
    def write(
        cls, obj: GlobalType, location: Location, packing: PackingArguments
    ) -> None:
        value: str
        if isinstance(obj, str):
            value = "builtins." + obj if "." not in obj else obj
        else:
            value = obj.__module__ + "." + obj.__qualname__
        entry = location.create_dataset(
            data=value, dtype=h5py.string_dtype(encoding="utf-8")
        )
        cls._write_type(entry)
        cls._write_metadata(
            entry,
            get_metadata(
                obj,
                packing.require_versions,
                packing.forbidden_modules,
                {} if packing.version_scraping is None else packing.version_scraping,
            ),
        )

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> Any:
        import_string = location.entry[()].decode("utf-8")
        return import_from_string(import_string)


class NoneItem(Item[type[None], None]):
    @classmethod
    def write(
        cls, obj: type[None], location: Location, packing: PackingArguments
    ) -> None:
        entry = location.create_dataset(data=h5py.Empty(dtype="f"))
        cls._write_type(entry)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> None:
        return None


ItemType = TypeVar("ItemType", bound=Any)


class SimpleItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    @classmethod
    def write(
        cls, obj: ItemType, location: Location, packing: PackingArguments
    ) -> None:
        entry = cls._make_dataset(obj, location)
        cls._write_type(entry)

    @classmethod
    @abc.abstractmethod
    def _make_dataset(cls, obj: ItemType, location: Location) -> h5py.Dataset:
        pass


class Complex(SimpleItem[complex]):
    @classmethod
    def _make_dataset(cls, obj: complex, location: Location) -> h5py.Dataset:
        return location.create_dataset(data=np.array([obj.real, obj.imag]))

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> complex:
        entry = location.entry
        return complex(entry[0], entry[1])


class Str(SimpleItem[str]):
    @classmethod
    def _make_dataset(cls, obj: str, location: Location) -> h5py.Dataset:
        return location.create_dataset(
            data=obj, dtype=h5py.string_dtype(encoding="utf-8")
        )

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> str:
        return cast(str, location.entry[()].decode("utf-8"))


class Bytes(SimpleItem[bytes]):
    @classmethod
    def _make_dataset(cls, obj: bytes, location: Location) -> h5py.Dataset:
        return location.create_dataset(data=np.void(obj))

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> bytes:
        return bytes(location.entry[()])


class NativeItem(SimpleItem[ItemType], Generic[ItemType], abc.ABC):
    recast: type[ItemType]

    @classmethod
    def _make_dataset(cls, obj: ItemType, location: Location) -> h5py.Dataset:
        return location.create_dataset(data=obj)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> ItemType:
        return cast(ItemType, cls.recast(location.entry[()]))


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
    def write(
        cls,
        obj: ItemType,
        location: Location,
        packing: PackingArguments,
    ) -> None:
        entry = cls._make_dataset(obj, location)
        cls._write_type(entry)
        cls._write_metadata(
            entry,
            get_metadata(
                obj,
                packing.require_versions,
                packing.forbidden_modules,
                {} if packing.version_scraping is None else packing.version_scraping,
            ),
        )

    @classmethod
    @abc.abstractmethod
    def _make_dataset(cls, obj: ItemType, location: Location) -> h5py.Dataset:
        pass


class Array(ComplexItem[np.ndarray[tuple[int, ...], H5DtypeAlias]]):
    @classmethod
    def _make_dataset(
        cls,
        obj: np.ndarray[tuple[int, ...], H5DtypeAlias],
        location: Location,
    ) -> h5py.Dataset:
        return location.create_dataset(data=obj)

    @classmethod
    def read(
        cls,
        location: Location,
        unpacking: UnpackingArguments,
    ) -> np.ndarray[tuple[int, ...], H5DtypeAlias]:
        return cast(
            np.ndarray[tuple[int, ...], H5DtypeAlias],
            location.entry[()],
        )


class Group(
    Content[PackingType, UnpackingType], Generic[PackingType, UnpackingType], abc.ABC
):
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
    def write(
        cls,
        obj: object,
        location: Location,
        packing: PackingArguments,
        rv: ReduceReturnType | None = None,
    ) -> None:
        reduced_value = (
            obj.__reduce_ex__(packing._pickle_protocol) if rv is None else rv
        )
        entry = location.create_group()
        cls._write_type(entry)
        cls._write_metadata(
            entry,
            get_metadata(
                obj,
                packing.require_versions,
                packing.forbidden_modules,
                get_metadata(
                    {} if packing.version_scraping is None else packing.version_scraping
                ),
            ),
        )
        for subpath, value in zip(cls.reduction_fields, reduced_value, strict=False):
            pack(
                value,
                location.bag,
                location.bag.join(location.path, subpath),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> object:
        constructor = cast(
            ConstructorType,
            unpack(
                location.bag,
                location.bag.join(location.path, "constructor"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            ),
        )
        constructor_args = cast(
            ConstructorArgsType,
            unpack(
                location.bag,
                location.bag.join(location.path, "args"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            ),
        )
        obj: object = constructor(*constructor_args)
        unpacking.memo[location.path] = obj
        rv = (constructor, constructor_args) + tuple(
            unpack(
                location.bag,
                location.bag.join(location.path, k),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for k in cls.reduction_fields[2 : len(location.entry)]
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
    def write(
        cls,
        obj: PackingType,
        location: Location,
        packing: PackingArguments,
    ) -> None:
        entry = location.create_group()
        cls._write_type(entry)
        cls._write_subcontent(obj, location, packing)

    @classmethod
    @abc.abstractmethod
    def _write_subcontent(
        cls,
        obj: PackingType,
        location: Location,
        packing: PackingArguments,
    ) -> h5py.Group:
        pass


class Dict(SimpleGroup[dict[Any, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[Any, Any],
        location: Location,
        packing: PackingArguments,
    ) -> None:
        pack(
            tuple(obj.keys()),
            location.bag,
            location.bag.join(location.path, "keys"),
            packing.memo,
            packing.references,
            packing.require_versions,
            packing.forbidden_modules,
            packing.version_scraping,
            _pickle_protocol=packing._pickle_protocol,
        )
        pack(
            tuple(obj.values()),
            location.bag,
            location.bag.join(location.path, "values"),
            packing.memo,
            packing.references,
            packing.require_versions,
            packing.forbidden_modules,
            packing.version_scraping,
            _pickle_protocol=packing._pickle_protocol,
        )

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> dict[Any, Any]:
        return dict(
            zip(
                cast(
                    tuple[Any],
                    unpack(
                        location.bag,
                        location.bag.join(location.path, "keys"),
                        unpacking.memo,
                        version_validator=unpacking.version_validator,
                        version_scraping=unpacking.version_scraping,
                    ),
                ),
                cast(
                    tuple[Any],
                    unpack(
                        location.bag,
                        location.bag.join(location.path, "values"),
                        unpacking.memo,
                        version_validator=unpacking.version_validator,
                        version_scraping=unpacking.version_scraping,
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
        location: Location,
        packing: PackingArguments,
    ) -> None:
        for k, v in obj.items():
            pack(
                v,
                location.bag,
                location.bag.join(location.path, k),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> dict[str, Any]:
        return {
            k: unpack(
                location.bag,
                location.bag.join(location.path, k),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for k in location.entry
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
        location: Location,
        packing: PackingArguments,
    ) -> None:
        for i, v in enumerate(obj.__args__):
            pack(
                v,
                location.bag,
                location.bag.join(location.path, f"i{i}"),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
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
    def read(cls, location: Location, unpacking: UnpackingArguments) -> types.UnionType:
        return cls._recursive_or(
            unpack(
                location.bag,
                location.bag.join(location.path, f"i{i}"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for i in range(len(location.entry))
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
        location: Location,
        packing: PackingArguments,
    ) -> None:
        for i, v in enumerate(obj):
            pack(
                v,
                location.bag,
                location.bag.join(location.path, f"i{i}"),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> IndexableType:
        return cls.recast(
            unpack(
                location.bag,
                location.bag.join(location.path, f"i{i}"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for i in range(len(location.entry))
        )


class Tuple(Indexable[tuple[Any, ...]]):
    recast = tuple


class List(Indexable[list[Any]]):
    recast = list


class Set(Indexable[set[Any]]):
    recast = set


class FrozenSet(Indexable[frozenset[Any]]):
    recast = frozenset


class PickleProtocolError(BagOfHoldingError, ValueError):
    pass


def pack(
    obj: object,
    bag: H5Bag,
    path: str,
    memo: PackingMemoAlias,
    references: ReferencesAlias,
    require_versions: bool,
    forbidden_modules: list[str] | tuple[str, ...],
    version_scraping: VersionScrapingMap | None,
    _pickle_protocol: SupportsIndex = pickle.DEFAULT_PROTOCOL,
) -> None:
    if _pickle_protocol not in (4, 3, 2, 1, 0):
        raise PickleProtocolError(
            f"pickle protocol must be <= 4, got {_pickle_protocol}"
        )

    location = Location(bag=bag, path=path)
    packing_args = PackingArguments(
        memo=memo,
        references=references,
        require_versions=require_versions,
        forbidden_modules=forbidden_modules,
        version_scraping=version_scraping,
        _pickle_protocol=_pickle_protocol,
    )

    t = type if isinstance(obj, type) else type(obj)
    simple_class = KNOWN_ITEM_MAP.get(t)
    if simple_class is not None:
        simple_class.write(obj, location, packing_args)
        return

    obj_id = id(obj)
    reference = memo.get(obj_id)
    if reference is not None:
        Reference.write(reference, location, packing_args)
        return
    else:
        memo[obj_id] = path
        references.append(obj)

    complex_class = get_complex_content_class(obj)
    if complex_class is not None:
        complex_class.write(obj, location, packing_args)
        return

    group_class = get_group_content_class(obj)
    if group_class is not None:
        group_class.write(obj, location, packing_args)
        return

    rv = obj.__reduce_ex__(_pickle_protocol)
    if isinstance(rv, str):
        Global.write(
            get_importable_string_from_string_reduction(rv, obj), location, packing_args
        )
        return
    else:
        Reducible.write(obj, location, packing_args, rv=rv)
        return


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
    bag: H5Bag,
    path: str,
    memo: UnpackingMemoAlias,
    version_validator: VersionValidatorType,
    version_scraping: VersionScrapingMap | None,
) -> object:
    memo_value = memo.get(path, NotData)
    if memo_value is NotData:
        entry = bag.file[path]
        content_class_string = maybe_decode(entry.attrs["content_type"])
        content_class = import_from_string(content_class_string)
        metadata = read_metadata(entry)
        if metadata is not None:
            validate_version(
                metadata, validator=version_validator, version_scraping=version_scraping
            )
        value = content_class.read(
            Location(bag=bag, path=path),
            UnpackingArguments(
                memo=memo,
                version_validator=version_validator,
                version_scraping=version_scraping,
            ),
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
