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

from bagofholding import ModuleForbiddenError, NoVersionError, PickleProtocolError
from bagofholding.h5.dtypes import H5DtypeAlias
from bagofholding.metadata import (
    Metadata,
    VersionScrapingMap,
    VersionValidatorType,
    get_module,
    get_qualname,
    get_version,
    validate_version,
)
from bagofholding.retrieve import (
    get_importable_string_from_string_reduction,
    import_from_string,
)
from bagofholding.types import BuiltinGroupType, BuiltinItemType

if TYPE_CHECKING:
    from bagofholding.h5.bag import H5Bag

PackingMemoAlias: TypeAlias = bidict.bidict[int, str]
ReferencesAlias: TypeAlias = list[object]
UnpackingMemoAlias: TypeAlias = dict[str, Any]


PackingType = TypeVar("PackingType", bound=Any)
UnpackingType = TypeVar("UnpackingType", bound=Any)


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

    _rich_metadata: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def read(
        cls, bag: H5Bag, path: str, unpacking: UnpackingArguments
    ) -> UnpackingType:
        # TODO: Optionally first read the metadata and verify that your env is viable
        pass

    @classmethod
    @abc.abstractmethod
    def write(
        cls,
        obj: PackingType,
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        pass

    @classmethod
    def _get_metadata(cls, obj: PackingType, packing: PackingArguments) -> Metadata:
        if cls._rich_metadata:
            module = get_module(obj)
            if module == "builtins":
                return Metadata(cls.full_name())
            else:
                if module.split(".")[0] in packing.forbidden_modules:
                    raise ModuleForbiddenError(
                        f"Module '{module}' is forbidden as a source of stored objects. Change "
                        f"the `forbidden_modules` or move this object to an allowed module."
                    )

                version = get_version(module, packing.version_scraping)
                if packing.require_versions and version is None:
                    raise NoVersionError(
                        f"Could not find a version for {module}. Either disable "
                        f"`require_versions`, use `version_scraping` to find an existing "
                        f"version for this package, or add versioning to the unversioned "
                        f"package."
                    )

                return Metadata(
                    cls.full_name(),
                    qualname=get_qualname(obj),
                    module=module,
                    version=version,
                    meta=(
                        str(obj.__metadata__) if hasattr(obj, "__metadata__") else None
                    ),
                )
        else:
            return Metadata(cls.full_name())

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
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        cls._write_item(obj, bag, path)
        bag.pack_metadata(cls._get_metadata(obj, packing), path)

    @classmethod
    @abc.abstractmethod
    def _write_item(cls, obj: PackingType, bag: H5Bag, path: str) -> None:
        pass


class Reference(Item[str, Any]):
    @classmethod
    def _write_item(cls, obj: str, bag: H5Bag, path: str) -> None:
        bag.pack_string(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> Any:
        reference = bag.unpack_string(path)
        from_memo = unpacking.memo.get(reference, NotData)
        if from_memo is not NotData:
            return from_memo
        else:
            return unpack(
                bag,
                reference,
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )


GlobalType: TypeAlias = type[type] | FunctionType | str


class Global(Item[GlobalType, Any]):
    _rich_metadata = True

    @classmethod
    def _write_item(cls, obj: GlobalType, bag: H5Bag, path: str) -> None:
        value: str
        if isinstance(obj, str):
            value = "builtins." + obj if "." not in obj else obj
        else:
            value = obj.__module__ + "." + obj.__qualname__
        bag.pack_string(value, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> Any:
        import_string = bag.unpack_string(path)
        return import_from_string(import_string)


class NoneItem(Item[type[None], None]):
    @classmethod
    def _write_item(cls, obj: type[None], bag: H5Bag, path: str) -> None:
        bag.pack_empty(path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> None:
        return None


ItemType = TypeVar("ItemType", bound=Any)


class ReflexiveItem(Item[ItemType, ItemType], Generic[ItemType], abc.ABC):
    pass


class BuiltinItem(ReflexiveItem[BuiltinItemType], Generic[BuiltinItemType], abc.ABC):
    pass


class Complex(BuiltinItem[complex]):
    @classmethod
    def _write_item(cls, obj: complex, bag: H5Bag, path: str) -> None:
        bag.pack_complex(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> complex:
        return bag.unpack_complex(path)


class Str(BuiltinItem[str]):
    @classmethod
    def _write_item(cls, obj: str, bag: H5Bag, path: str) -> None:
        bag.pack_string(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> str:
        return bag.unpack_string(path)


class Bytes(BuiltinItem[bytes]):
    @classmethod
    def _write_item(cls, obj: bytes, bag: H5Bag, path: str) -> None:
        bag.pack_bytes(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> bytes:
        return bag.unpack_bytes(path)


class Bool(BuiltinItem[bool]):
    @classmethod
    def _write_item(cls, obj: bool, bag: H5Bag, path: str) -> None:
        bag.pack_bool(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> bool:
        return bag.unpack_bool(path)


class Long(BuiltinItem[int]):
    @classmethod
    def _write_item(cls, obj: int, bag: H5Bag, path: str) -> None:
        bag.pack_long(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> int:
        return bag.unpack_long(path)


class Float(BuiltinItem[float]):
    @classmethod
    def _write_item(cls, obj: float, bag: H5Bag, path: str) -> None:
        bag.pack_float(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> float:
        return bag.unpack_float(path)


class Bytearray(BuiltinItem[bytearray]):
    @classmethod
    def _write_item(cls, obj: bytearray, bag: H5Bag, path: str) -> None:
        bag.pack_bytearray(obj, path)

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> bytearray:
        return bag.unpack_bytearray(path)


class BespokeItem(ReflexiveItem[ItemType], Generic[ItemType], abc.ABC):
    _rich_metadata = True


class Array(BespokeItem[np.ndarray[tuple[int, ...], H5DtypeAlias]]):
    @classmethod
    def _write_item(
        cls, obj: np.ndarray[tuple[int, ...], H5DtypeAlias], bag: H5Bag, path: str
    ) -> None:
        bag.file.create_dataset(path, data=obj)

    @classmethod
    def read(
        cls,
        bag: H5Bag,
        path: str,
        unpacking: UnpackingArguments,
    ) -> np.ndarray[tuple[int, ...], H5DtypeAlias]:
        return cast(
            np.ndarray[tuple[int, ...], H5DtypeAlias],
            bag.file[path][()],
        )


class Group(
    Content[PackingType, UnpackingType], Generic[PackingType, UnpackingType], abc.ABC
):
    pass


GroupType = TypeVar("GroupType", bound=Any)  # Bind to container?


class ReflexiveGroup(Group[GroupType, GroupType], Generic[GroupType], abc.ABC):
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


class Reducible(ReflexiveGroup[object]):
    _rich_metadata = True
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
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
        rv: ReduceReturnType | None = None,
    ) -> None:
        reduced_value = (
            obj.__reduce_ex__(packing._pickle_protocol) if rv is None else rv
        )
        bag.create_group(path)
        bag.pack_metadata(cls._get_metadata(obj, packing), path)
        for subpath, value in zip(cls.reduction_fields, reduced_value, strict=False):
            pack(
                value,
                bag,
                bag.join(path, subpath),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def read(cls, bag: H5Bag, path: str, unpacking: UnpackingArguments) -> object:
        constructor = cast(
            ConstructorType,
            unpack(
                bag,
                bag.join(path, "constructor"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            ),
        )
        constructor_args = cast(
            ConstructorArgsType,
            unpack(
                bag,
                bag.join(path, "args"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            ),
        )
        obj: object = constructor(*constructor_args)
        unpacking.memo[path] = obj
        rv = (constructor, constructor_args) + tuple(
            unpack(
                bag,
                bag.join(path, k),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for k in cls.reduction_fields[2 : len(bag.open_group(path))]
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


class BuiltinGroup(
    ReflexiveGroup[BuiltinGroupType], Generic[BuiltinGroupType], abc.ABC
):
    @classmethod
    def write(
        cls,
        obj: PackingType,
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        bag.create_group(path)
        bag.pack_metadata(cls._get_metadata(obj, packing), path)
        cls._write_subcontent(obj, bag, path, packing)

    @classmethod
    @abc.abstractmethod
    def _write_subcontent(
        cls,
        obj: PackingType,
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> h5py.Group:
        pass


class Dict(BuiltinGroup[dict[Any, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[Any, Any],
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        pack(
            tuple(obj.keys()),
            bag,
            bag.join(path, "keys"),
            packing.memo,
            packing.references,
            packing.require_versions,
            packing.forbidden_modules,
            packing.version_scraping,
            _pickle_protocol=packing._pickle_protocol,
        )
        pack(
            tuple(obj.values()),
            bag,
            bag.join(path, "values"),
            packing.memo,
            packing.references,
            packing.require_versions,
            packing.forbidden_modules,
            packing.version_scraping,
            _pickle_protocol=packing._pickle_protocol,
        )

    @classmethod
    def read(
        cls, bag: H5Bag, path: str, unpacking: UnpackingArguments
    ) -> dict[Any, Any]:
        return dict(
            zip(
                cast(
                    tuple[Any],
                    unpack(
                        bag,
                        bag.join(path, "keys"),
                        unpacking.memo,
                        version_validator=unpacking.version_validator,
                        version_scraping=unpacking.version_scraping,
                    ),
                ),
                cast(
                    tuple[Any],
                    unpack(
                        bag,
                        bag.join(path, "values"),
                        unpacking.memo,
                        version_validator=unpacking.version_validator,
                        version_scraping=unpacking.version_scraping,
                    ),
                ),
                strict=True,
            )
        )


class StrKeyDict(BuiltinGroup[dict[str, Any]]):
    @classmethod
    def _write_subcontent(
        cls,
        obj: dict[str, Any],
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        for k, v in obj.items():
            pack(
                v,
                bag,
                bag.join(path, k),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def read(
        cls, bag: H5Bag, path: str, unpacking: UnpackingArguments
    ) -> dict[str, Any]:
        return {
            k: unpack(
                bag,
                bag.join(path, k),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for k in bag.open_group(path)
        }


class Union(BuiltinGroup[types.UnionType]):
    """
    :class:`types.UnionType` has no :meth:`__reduce__` method. Pickle actually gets
    around this with bespoke logic, and so we need to too.
    """

    @classmethod
    def _write_subcontent(
        cls,
        obj: types.UnionType,
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        for i, v in enumerate(obj.__args__):
            pack(
                v,
                bag,
                bag.join(path, f"i{i}"),
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
    def read(
        cls, bag: H5Bag, path: str, unpacking: UnpackingArguments
    ) -> types.UnionType:
        return cls._recursive_or(
            unpack(
                bag,
                bag.join(path, f"i{i}"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for i in range(len(bag.open_group(path)))
        )


IndexableType = TypeVar(
    "IndexableType", tuple[Any, ...], list[Any], set[Any], frozenset[Any]
)


class Indexable(BuiltinGroup[IndexableType], Generic[IndexableType], abc.ABC):
    recast: type[IndexableType]

    @classmethod
    def _write_subcontent(
        cls,
        obj: IndexableType,
        bag: H5Bag,
        path: str,
        packing: PackingArguments,
    ) -> None:
        for i, v in enumerate(obj):
            pack(
                v,
                bag,
                bag.join(path, f"i{i}"),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def read(
        cls, bag: H5Bag, path: str, unpacking: UnpackingArguments
    ) -> IndexableType:
        return cls.recast(
            unpack(
                bag,
                bag.join(path, f"i{i}"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for i in range(len(bag.open_group(path)))
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
            bag,
            path,
            packing_args,
        )
        return

    obj_id = id(obj)
    reference = memo.get(obj_id)
    if reference is not None:
        Reference.write(reference, bag, path, packing_args)
        return
    else:
        memo[obj_id] = path
        references.append(obj)

    complex_class = bag.get_bespoke_content_class(obj)
    if complex_class is not None:
        complex_class.write(obj, bag, path, packing_args)
        return

    group_class = get_group_content_class(obj)
    if group_class is not None:
        group_class.write(obj, bag, path, packing_args)
        return

    rv = obj.__reduce_ex__(_pickle_protocol)
    if isinstance(rv, str):
        Global.write(
            get_importable_string_from_string_reduction(rv, obj),
            bag,
            path,
            packing_args,
        )
        return
    else:
        Reducible.write(obj, bag, path, packing_args, rv=rv)
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
        metadata = bag.unpack_metadata(path)
        content_class = import_from_string(metadata.content_type)
        if metadata is not None:
            validate_version(
                metadata, validator=version_validator, version_scraping=version_scraping
            )
        value = content_class.read(
            bag,
            path,
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
