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

    key: ClassVar[str] = "content_type"

    @classmethod
    @abc.abstractmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> UnpackingType:
        # TODO: Optionally first read the metadata and verify that your env is viable
        pass

    @classmethod
    @abc.abstractmethod
    def write(
        cls,
        obj: PackingType,
        location: Location,
        packing: PackingArguments,
        metadata: Metadata | None = None,
    ) -> None:
        pass

    @classmethod
    def full_name(cls) -> str:
        return cls.__module__ + "." + cls.__name__


class Item(
    Content[PackingType, UnpackingType], Generic[PackingType, UnpackingType], abc.ABC
):
    @classmethod
    def write(
        cls,
        obj: PackingType,
        location: Location,
        packing: PackingArguments,
        metadata: Metadata | None = None,
    ) -> None:
        cls._write_item(obj, location)
        location.bag.pack_content_type(cls, location.path)
        location.bag.pack_metadata(metadata, location.path)

    @classmethod
    @abc.abstractmethod
    def _write_item(cls, obj: PackingType, location: Location) -> None:
        pass


class Reference(Item[str, Any]):
    @classmethod
    def _write_item(cls, obj: str, location: Location) -> None:
        location.bag.pack_string(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> Any:
        reference = location.bag.unpack_string(location.path)
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
    def _write_item(cls, obj: GlobalType, location: Location) -> None:
        value: str
        if isinstance(obj, str):
            value = "builtins." + obj if "." not in obj else obj
        else:
            value = obj.__module__ + "." + obj.__qualname__
        location.bag.pack_string(value, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> Any:
        import_string = location.bag.unpack_string(location.path)
        return import_from_string(import_string)


class NoneItem(Item[type[None], None]):
    @classmethod
    def _write_item(cls, obj: type[None], location: Location) -> None:
        location.bag.pack_empty(location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> None:
        return None


ItemType = TypeVar("ItemType", bound=Any)


class SimpleItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    pass


class Complex(SimpleItem[complex]):
    @classmethod
    def _write_item(cls, obj: complex, location: Location) -> None:
        location.bag.pack_complex(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> complex:
        return location.bag.unpack_complex(location.path)


class Str(SimpleItem[str]):
    @classmethod
    def _write_item(cls, obj: str, location: Location) -> None:
        location.bag.pack_string(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> str:
        return location.bag.unpack_string(location.path)


class Bytes(SimpleItem[bytes]):
    @classmethod
    def _write_item(cls, obj: bytes, location: Location) -> None:
        location.bag.pack_bytes(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> bytes:
        return location.bag.unpack_bytes(location.path)


class Bool(SimpleItem[bool]):
    @classmethod
    def _write_item(cls, obj: bool, location: Location) -> None:
        location.bag.pack_bool(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> bool:
        return location.bag.unpack_bool(location.path)


class Long(SimpleItem[int]):
    @classmethod
    def _write_item(cls, obj: int, location: Location) -> None:
        location.bag.pack_long(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> int:
        return location.bag.unpack_long(location.path)


class Float(SimpleItem[float]):
    @classmethod
    def _write_item(cls, obj: float, location: Location) -> None:
        location.bag.pack_float(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> float:
        return location.bag.unpack_float(location.path)


class Bytearray(SimpleItem[bytearray]):
    @classmethod
    def _write_item(cls, obj: bytearray, location: Location) -> None:
        location.bag.pack_bytearray(obj, location.path)

    @classmethod
    def read(cls, location: Location, unpacking: UnpackingArguments) -> bytearray:
        return location.bag.unpack_bytearray(location.path)


class ComplexItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    pass


class Array(ComplexItem[np.ndarray[tuple[int, ...], H5DtypeAlias]]):
    @classmethod
    def _write_item(
        cls, obj: np.ndarray[tuple[int, ...], H5DtypeAlias], location: Location
    ) -> None:
        location.bag.file.create_dataset(location.path, data=obj)

    @classmethod
    def read(
        cls,
        location: Location,
        unpacking: UnpackingArguments,
    ) -> np.ndarray[tuple[int, ...], H5DtypeAlias]:
        return cast(
            np.ndarray[tuple[int, ...], H5DtypeAlias],
            location.bag.file[location.path][()],
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
        metadata: Metadata | None = None,
        rv: ReduceReturnType | None = None,
    ) -> None:
        reduced_value = (
            obj.__reduce_ex__(packing._pickle_protocol) if rv is None else rv
        )
        location.bag.pack_group(location.path)
        location.bag.pack_content_type(cls, location.path)
        location.bag.pack_metadata(metadata, location.path)
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
        metadata: Metadata | None = None,
    ) -> None:
        location.bag.pack_group(location.path)
        location.bag.pack_content_type(cls, location.path)
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
        simple_class.write(
            obj,
            location,
            packing_args,
            metadata=(
                get_metadata(obj, require_versions, forbidden_modules, version_scraping)
                if simple_class is Global
                else None
            ),
        )
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
        complex_class.write(
            obj,
            location,
            packing_args,
            metadata=get_metadata(
                obj, require_versions, forbidden_modules, version_scraping
            ),
        )
        return

    group_class = get_group_content_class(obj)
    if group_class is not None:
        group_class.write(obj, location, packing_args)
        return

    rv = obj.__reduce_ex__(_pickle_protocol)
    if isinstance(rv, str):
        Global.write(
            get_importable_string_from_string_reduction(rv, obj),
            location,
            packing_args,
            metadata=get_metadata(
                obj, require_versions, forbidden_modules, version_scraping
            ),
        )
        return
    else:
        Reducible.write(
            obj,
            location,
            packing_args,
            metadata=get_metadata(
                obj, require_versions, forbidden_modules, version_scraping
            ),
            rv=rv,
        )
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
        content_class_string = bag.unpack_content_type(path)
        content_class = import_from_string(content_class_string)
        metadata = bag.unpack_metadata(path)
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
