"""
Content controls the decomposition of python objects into their component parts, along
with a record of what type of object they are, and gathering any relevant metadata.

Content is not, of itself, concerned with what you _do_ with this decompositon.
That is handled via the "packer" protocol (for us, the relevant implementation of this
is "bags").
"""

from __future__ import annotations

import abc
import collections.abc
import dataclasses
import operator
import pickle
import types
from collections.abc import Callable, Iterable, Iterator, Sized
from typing import (
    Any,
    ClassVar,
    Generic,
    Protocol,
    Self,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
)

import bidict
import h5py

from bagofholding.exceptions import (
    ModuleForbiddenError,
    NoVersionError,
    PickleProtocolError,
    StringNotImportableError,
)
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

PackingMemoAlias: TypeAlias = bidict.bidict[int, str]
ReferencesAlias: TypeAlias = list[object]
UnpackingMemoAlias: TypeAlias = dict[str, Any]


PackingType = TypeVar("PackingType", bound=Any)
UnpackingType = TypeVar("UnpackingType", bound=Any)


class HasContents(Sized, Iterable[str], Protocol): ...


class Packer(Protocol):
    def join(self, *paths: str) -> str: ...

    def pack_metadata(self, metadata: Metadata, path: str) -> None: ...
    def unpack_metadata(self, path: str) -> Metadata: ...

    def pack_empty(self, path: str) -> None: ...

    def pack_string(self, obj: str, path: str) -> None: ...
    def unpack_string(self, path: str) -> str: ...

    def pack_bool(self, obj: bool, path: str) -> None: ...
    def unpack_bool(self, path: str) -> bool: ...

    def pack_long(self, obj: int, path: str) -> None: ...
    def unpack_long(self, path: str) -> int: ...

    def pack_float(self, obj: float, path: str) -> None: ...
    def unpack_float(self, path: str) -> float: ...

    def pack_complex(self, obj: complex, path: str) -> None: ...
    def unpack_complex(self, path: str) -> complex: ...

    def pack_bytes(self, obj: bytes, path: str) -> None: ...
    def unpack_bytes(self, path: str) -> bytes: ...

    def pack_bytearray(self, obj: bytearray, path: str) -> None: ...
    def unpack_bytearray(self, path: str) -> bytearray: ...

    def create_group(self, path: str) -> None: ...
    def open_group(self, path: str) -> HasContents: ...

    def get_bespoke_content_class(
        self, obj: object
    ) -> type[BespokeItem[Any, Self]] | None: ...


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


PackerType = TypeVar("PackerType", bound=Packer)


class Content(Generic[PackingType, UnpackingType, PackerType], abc.ABC):

    _rich_metadata: ClassVar[bool] = False

    @classmethod
    @abc.abstractmethod
    def pack(
        cls,
        obj: PackingType,
        packer: PackerType,
        path: str,
        packing: PackingArguments,
    ) -> None: ...

    @classmethod
    @abc.abstractmethod
    def unpack(
        cls, packer: PackerType, path: str, unpacking: UnpackingArguments
    ) -> UnpackingType: ...

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
    Content[PackingType, UnpackingType, PackerType],
    Generic[PackingType, UnpackingType, PackerType],
    abc.ABC,
):
    @classmethod
    def pack(
        cls,
        obj: PackingType,
        packer: PackerType,
        path: str,
        packing: PackingArguments,
    ) -> None:
        cls._pack_item(obj, packer, path)
        packer.pack_metadata(cls._get_metadata(obj, packing), path)

    @classmethod
    @abc.abstractmethod
    def _pack_item(cls, obj: PackingType, packer: PackerType, path: str) -> None: ...


class Reference(Item[str, Any, Packer]):
    @classmethod
    def _pack_item(cls, obj: str, packer: Packer, path: str) -> None:
        packer.pack_string(obj, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> Any:
        reference = packer.unpack_string(path)
        from_memo = unpacking.memo.get(reference, NotData)
        if from_memo is not NotData:
            return from_memo
        else:
            return unpack(
                packer,
                reference,
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )


GlobalType: TypeAlias = type[type] | types.FunctionType | str


class Global(Item[GlobalType, Any, Packer]):
    _rich_metadata = True

    @classmethod
    def _pack_item(cls, obj: GlobalType, packer: Packer, path: str) -> None:
        value: str
        if isinstance(obj, str):
            value = "builtins." + obj if "." not in obj else obj
        else:
            value = obj.__module__ + "." + obj.__qualname__

        if "<lambda>" in value:
            raise StringNotImportableError(
                f"Lambda functions are not re-importable, can't pack {obj}"
            )
        elif "<locals>" in value:
            raise StringNotImportableError(
                f"Local functions are not re-importable, can't pack {obj}"
            )
        packer.pack_string(value, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> Any:
        import_string = packer.unpack_string(path)
        return import_from_string(import_string)


class NoneItem(Item[type[None], None, Packer]):
    @classmethod
    def _pack_item(cls, obj: type[None], packer: Packer, path: str) -> None:
        packer.pack_empty(path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> None:
        return None


ItemType = TypeVar("ItemType", bound=Any)


class ReflexiveItem(
    Item[ItemType, ItemType, PackerType], Generic[ItemType, PackerType], abc.ABC
): ...


BuiltinItemType = TypeVar(
    "BuiltinItemType",
    str,
    bytes,
    bytearray,
    bool,
    int,
    float,
    complex,
)


class BuiltinItem(
    ReflexiveItem[BuiltinItemType, Packer], Generic[BuiltinItemType], abc.ABC
): ...


class Str(BuiltinItem[str]):
    @classmethod
    def _pack_item(cls, obj: str, packer: Packer, path: str) -> None:
        packer.pack_string(obj, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> str:
        return packer.unpack_string(path)


class Bool(BuiltinItem[bool]):
    @classmethod
    def _pack_item(cls, obj: bool, packer: Packer, path: str) -> None:
        packer.pack_bool(obj, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> bool:
        return packer.unpack_bool(path)


class Long(BuiltinItem[int]):
    @classmethod
    def _pack_item(cls, obj: int, packer: Packer, path: str) -> None:
        packer.pack_long(obj, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> int:
        return packer.unpack_long(path)


class Float(BuiltinItem[float]):
    @classmethod
    def _pack_item(cls, obj: float, packer: Packer, path: str) -> None:
        packer.pack_float(obj, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> float:
        return packer.unpack_float(path)


class Complex(BuiltinItem[complex]):
    @classmethod
    def _pack_item(cls, obj: complex, packer: Packer, path: str) -> None:
        packer.pack_complex(obj, path)

    @classmethod
    def unpack(
        cls, packer: Packer, path: str, unpacking: UnpackingArguments
    ) -> complex:
        return packer.unpack_complex(path)


class Bytes(BuiltinItem[bytes]):
    @classmethod
    def _pack_item(cls, obj: bytes, packer: Packer, path: str) -> None:
        packer.pack_bytes(obj, path)

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> bytes:
        return packer.unpack_bytes(path)


class Bytearray(BuiltinItem[bytearray]):
    @classmethod
    def _pack_item(cls, obj: bytearray, packer: Packer, path: str) -> None:
        packer.pack_bytearray(obj, path)

    @classmethod
    def unpack(
        cls, packer: Packer, path: str, unpacking: UnpackingArguments
    ) -> bytearray:
        return packer.unpack_bytearray(path)


class BespokeItem(
    ReflexiveItem[ItemType, PackerType], Generic[ItemType, PackerType], abc.ABC
):
    _rich_metadata = True


class Group(
    Content[PackingType, UnpackingType, Packer],
    Generic[PackingType, UnpackingType],
    abc.ABC,
): ...


GroupType = TypeVar("GroupType", bound=Any)  # Bind to container?


class ReflexiveGroup(Group[GroupType, GroupType], Generic[GroupType], abc.ABC): ...


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
    def pack(
        cls,
        obj: object,
        packer: Packer,
        path: str,
        packing: PackingArguments,
        rv: ReduceReturnType | None = None,
    ) -> None:
        reduced_value = (
            obj.__reduce_ex__(packing._pickle_protocol) if rv is None else rv
        )
        packer.create_group(path)
        packer.pack_metadata(cls._get_metadata(obj, packing), path)
        for subpath, value in zip(cls.reduction_fields, reduced_value, strict=False):
            pack(
                value,
                packer,
                packer.join(path, subpath),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def unpack(cls, packer: Packer, path: str, unpacking: UnpackingArguments) -> object:
        constructor = cast(
            ConstructorType,
            unpack(
                packer,
                packer.join(path, "constructor"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            ),
        )
        constructor_args = cast(
            ConstructorArgsType,
            unpack(
                packer,
                packer.join(path, "args"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            ),
        )
        obj: object = constructor(*constructor_args)
        unpacking.memo[path] = obj
        rv = (constructor, constructor_args) + tuple(
            unpack(
                packer,
                packer.join(path, k),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for k in cls.reduction_fields[2 : len(packer.open_group(path))]
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


BuiltinGroupType = TypeVar(
    "BuiltinGroupType",
    dict[Any, Any],
    dict[str, Any],
    types.UnionType,
    tuple[Any, ...],
    list[Any],
    set[Any],
    frozenset[Any],
)


class BuiltinGroup(
    ReflexiveGroup[BuiltinGroupType], Generic[BuiltinGroupType], abc.ABC
):
    @classmethod
    def pack(
        cls,
        obj: PackingType,
        packer: Packer,
        path: str,
        packing: PackingArguments,
    ) -> None:
        packer.create_group(path)
        packer.pack_metadata(cls._get_metadata(obj, packing), path)
        cls._pack_subcontent(obj, packer, path, packing)

    @classmethod
    @abc.abstractmethod
    def _pack_subcontent(
        cls,
        obj: PackingType,
        packer: Packer,
        path: str,
        packing: PackingArguments,
    ) -> h5py.Group: ...


class Dict(BuiltinGroup[dict[Any, Any]]):
    @classmethod
    def _pack_subcontent(
        cls,
        obj: dict[Any, Any],
        packer: Packer,
        path: str,
        packing: PackingArguments,
    ) -> None:
        pack(
            tuple(obj.keys()),
            packer,
            packer.join(path, "keys"),
            packing.memo,
            packing.references,
            packing.require_versions,
            packing.forbidden_modules,
            packing.version_scraping,
            _pickle_protocol=packing._pickle_protocol,
        )
        pack(
            tuple(obj.values()),
            packer,
            packer.join(path, "values"),
            packing.memo,
            packing.references,
            packing.require_versions,
            packing.forbidden_modules,
            packing.version_scraping,
            _pickle_protocol=packing._pickle_protocol,
        )

    @classmethod
    def unpack(
        cls, packer: Packer, path: str, unpacking: UnpackingArguments
    ) -> dict[Any, Any]:
        return dict(
            zip(
                cast(
                    tuple[Any],
                    unpack(
                        packer,
                        packer.join(path, "keys"),
                        unpacking.memo,
                        version_validator=unpacking.version_validator,
                        version_scraping=unpacking.version_scraping,
                    ),
                ),
                cast(
                    tuple[Any],
                    unpack(
                        packer,
                        packer.join(path, "values"),
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
    def _pack_subcontent(
        cls,
        obj: dict[str, Any],
        packer: Packer,
        path: str,
        packing: PackingArguments,
    ) -> None:
        for k, v in obj.items():
            pack(
                v,
                packer,
                packer.join(path, k),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def unpack(
        cls, packer: Packer, path: str, unpacking: UnpackingArguments
    ) -> dict[str, Any]:
        return {
            k: unpack(
                packer,
                packer.join(path, k),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for k in packer.open_group(path)
        }


class Union(BuiltinGroup[types.UnionType]):
    """
    :class:`types.UnionType` has no :meth:`__reduce__` method. Pickle actually gets
    around this with bespoke logic, and so we need to too.
    """

    @classmethod
    def _pack_subcontent(
        cls,
        obj: types.UnionType,
        packer: Packer,
        path: str,
        packing: PackingArguments,
    ) -> None:
        for i, v in enumerate(obj.__args__):
            pack(
                v,
                packer,
                packer.join(path, f"i{i}"),
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
    def unpack(
        cls, packer: Packer, path: str, unpacking: UnpackingArguments
    ) -> types.UnionType:
        return cls._recursive_or(
            unpack(
                packer,
                packer.join(path, f"i{i}"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for i in range(len(packer.open_group(path)))
        )


IndexableType = TypeVar(
    "IndexableType", tuple[Any, ...], list[Any], set[Any], frozenset[Any]
)


class Indexable(BuiltinGroup[IndexableType], Generic[IndexableType], abc.ABC):
    recast: type[IndexableType]

    @classmethod
    def _pack_subcontent(
        cls,
        obj: IndexableType,
        packer: Packer,
        path: str,
        packing: PackingArguments,
    ) -> None:
        for i, v in enumerate(obj):
            pack(
                v,
                packer,
                packer.join(path, f"i{i}"),
                packing.memo,
                packing.references,
                packing.require_versions,
                packing.forbidden_modules,
                packing.version_scraping,
                _pickle_protocol=packing._pickle_protocol,
            )

    @classmethod
    def unpack(
        cls, packer: Packer, path: str, unpacking: UnpackingArguments
    ) -> IndexableType:
        return cls.recast(
            unpack(
                packer,
                packer.join(path, f"i{i}"),
                unpacking.memo,
                version_validator=unpacking.version_validator,
                version_scraping=unpacking.version_scraping,
            )
            for i in range(len(packer.open_group(path)))
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
    packer: Packer,
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
        simple_class.pack(
            obj,
            packer,
            path,
            packing_args,
        )
        return

    obj_id = id(obj)
    reference = memo.get(obj_id)
    if reference is not None:
        Reference.pack(reference, packer, path, packing_args)
        return
    else:
        memo[obj_id] = path
        references.append(obj)

    complex_class = packer.get_bespoke_content_class(obj)
    if complex_class is not None:
        complex_class.pack(obj, packer, path, packing_args)
        return

    group_class = get_group_content_class(obj)
    if group_class is not None:
        group_class.pack(obj, packer, path, packing_args)
        return

    rv = obj.__reduce_ex__(_pickle_protocol)
    if isinstance(rv, str):
        Global.pack(
            get_importable_string_from_string_reduction(rv, obj),
            packer,
            path,
            packing_args,
        )
        return
    else:
        Reducible.pack(obj, packer, path, packing_args, rv=rv)
        return


KNOWN_ITEM_MAP: dict[
    type | types.FunctionType | types.BuiltinFunctionType, type[Item[Any, Any, Packer]]
] = {
    type: Global,
    types.FunctionType: Global,
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
    packer: Packer,
    path: str,
    memo: UnpackingMemoAlias,
    version_validator: VersionValidatorType,
    version_scraping: VersionScrapingMap | None,
) -> object:
    memo_value = memo.get(path, NotData)
    if memo_value is NotData:
        metadata = packer.unpack_metadata(path)
        content_class = import_from_string(metadata.content_type)
        if metadata is not None:
            validate_version(
                metadata, validator=version_validator, version_scraping=version_scraping
            )
        value = content_class.unpack(
            packer,
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
