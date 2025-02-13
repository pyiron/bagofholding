from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Iterator, MutableMapping
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Self,
    TypeAlias,
    TypeVar,
    cast,
)

from bidict import bidict

from bagofholding.exception import BagOfHoldingError
from bagofholding.metadata import Metadata, get_metadata
from bagofholding.retrieve import import_from_string

if TYPE_CHECKING:
    from bagofholding.instance.bag import Bag

DispatcherAlias: TypeAlias = Callable[[object], "type[Content[Any, Any]] | None"]
PackingMemoAlias: TypeAlias = bidict[int, str]
ReferencesAlias: TypeAlias = list[object]
UnpackingMemoAlias: TypeAlias = dict[str, Any]

PATH_DELIMITER: str = "/"


class NotData:
    pass


def pack_content(
    obj: Any,
    path: str,
    dispatcher: DispatcherAlias,
    memo: PackingMemoAlias | None = None,
    references: ReferencesAlias | None = None,
) -> Content[Any, Any]:
    memo = memo if memo is not None else bidict()
    references = references if references is not None else []

    content_class = dispatcher(obj)
    if content_class is not None and issubclass(content_class, DirectItem):
        return content_class(path, get_metadata(obj)).pack(
            obj, dispatcher, memo, references
        )

    if id(obj) in memo:
        return Reference(path, None).pack(memo[id(obj)], dispatcher, memo, references)
    else:
        memo[id(obj)] = path
        references.append(obj)  # Otherwise the gc might reuse the id

    if content_class is not None:
        return content_class(path, get_metadata(obj)).pack(
            obj, dispatcher, memo, references
        )

    rv = obj.__reduce__()  # TODO: pickle compatibility with __reduce_ex__

    if isinstance(rv, str):
        return Global(path, get_metadata(obj)).pack(
            _get_importable_string_from_string_reduction(rv, obj),
            dispatcher,
            memo,
            references,
        )
    else:
        # TODO: We inefficiently call `__reduce__` twice, as Reducible also invokes it
        return Reducible(path, get_metadata(obj)).pack(
            obj, dispatcher, memo, references
        )


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


def unpack_content(
    bag: Bag,
    path: str,
    memo: UnpackingMemoAlias,
) -> Any:
    memo_value = memo.get(path, NotData)
    if memo_value is NotData:
        value = bag[path].unpack(bag, memo)
        if path not in memo:
            memo[path] = value
        return value
    else:
        return memo_value


PackingType = TypeVar("PackingType", bound=Any)
UnpackingType = TypeVar("UnpackingType", bound=Any)


class Content(Generic[PackingType, UnpackingType], ABC):
    path: str
    metadata: Metadata | None

    def __init__(
        self, path: str, metadata: Metadata | None, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.path = path
        self.metadata = metadata

    @abstractmethod
    def pack(
        self,
        obj: PackingType,
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        pass

    @abstractmethod
    def unpack(
        self, bag: Bag, memo: UnpackingMemoAlias
    ) -> (
        UnpackingType
    ):  # I think I need to break content type into packing and unpacking types
        pass

    def relative(self, subpath: str) -> str:
        return self.path + PATH_DELIMITER + subpath


StoredType = TypeVar("StoredType", bound=Any)


class Item(
    Content[PackingType, UnpackingType],
    Generic[PackingType, StoredType, UnpackingType],
    ABC,
):
    stored: StoredType | NotData = NotData()

    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> UnpackingType:
        if isinstance(self.stored, NotData):
            self.stored = bag.read_stored_item(self)
        return self._unpack_item(bag, memo)

    @abstractmethod
    def _unpack_item(self, bag: Bag, memo: UnpackingMemoAlias) -> UnpackingType:
        pass


class Reference(Item[str, str, Any]):
    def pack(
        self,
        obj: str,
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        self.stored = obj
        return self

    def _unpack_item(self, bag: Bag, memo: UnpackingMemoAlias) -> Any:
        reference = cast(str, self.stored)
        from_memo = memo.get(reference, NotData)
        if from_memo is NotData:
            return bag[reference].unpack(bag, memo)
        return from_memo


GlobalType: TypeAlias = type[type] | FunctionType | str


class Global(Item[GlobalType, str, Any]):
    def pack(
        self,
        obj: GlobalType,
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        if isinstance(obj, str):
            self.stored = "builtins." + obj if "." not in obj else obj
        else:
            self.stored = obj.__module__ + "." + obj.__qualname__
        return self

    def _unpack_item(self, bag: Bag, memo: UnpackingMemoAlias) -> Any:
        import_string = cast(str, self.stored)
        return import_from_string(import_string)


ChildType = TypeVar("ChildType", bound=Content[Any, Any])


class Group(
    Content[PackingType, UnpackingType],
    Generic[PackingType, UnpackingType, ChildType],
    MutableMapping[str, ChildType],
    ABC,
):
    children: OrderedDict[str, ChildType]

    def __init__(
        self, path: str, metadata: Metadata | None, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(path, metadata)
        self.children = OrderedDict()

    def __getitem__(self, key: str) -> ChildType:
        return self.children[key]

    def __setitem__(self, key: str, value: ChildType) -> None:
        self.children[key] = value

    def __delitem__(self, key: str) -> None:
        del self.children[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.children)

    def __len__(self) -> int:
        return len(self.children)


# __reduce__ return values
# per https://docs.python.org/3/library/pickle.html#object.__reduce__
ConstructorType: TypeAlias = Callable[..., object]
ConstructorArgsType: TypeAlias = tuple[object, ...]
StateType: TypeAlias = None | object
ListItemsType: TypeAlias = None | Iterator[object]
DictItemsType: TypeAlias = None | Iterator[tuple[object, object]]
SetStateCallableType: TypeAlias = None | Callable[[object, object], None]
ReduceReturnType: TypeAlias = (
    tuple[ConstructorType, ConstructorArgsType]
    | tuple[ConstructorType, ConstructorArgsType, StateType]
    | tuple[ConstructorType, ConstructorArgsType, StateType, ListItemsType]
    | tuple[
        ConstructorType, ConstructorArgsType, StateType, ListItemsType, DictItemsType
    ]
    | tuple[
        ConstructorType,
        ConstructorArgsType,
        StateType,
        ListItemsType,
        DictItemsType,
        SetStateCallableType,
    ]
)


class Reducible(Group[Any, Any, Content[Any, Any]]):

    reduction_fields: ClassVar[tuple[str, str, str, str, str, str]] = (
        "constructor",
        "args",
        "state",
        "item_iterator",
        "kv_iterator",
        "setter",
    )

    def pack(
        self,
        obj: Any,
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        rv = obj.__reduce__()  # TODO: pickle compatibility with __reduce_ex__
        for subpath, item in zip(self.reduction_fields, rv, strict=False):
            self[subpath] = pack_content(
                item,
                self.relative(subpath),
                dispatcher,
                memo,
                references,
            )
        return self

    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> Any:
        constructor = self["constructor"].unpack(bag, memo)
        constructor_args = self["args"].unpack(bag, memo)
        obj = constructor(*constructor_args)
        memo[self.path] = obj

        rv = (constructor, constructor_args) + tuple(
            self[k].unpack(bag, memo) for k in self.reduction_fields[2 : len(self)]
        )
        n_items = len(rv)
        if n_items >= 3 and rv[2] is not None:
            if n_items == 6 and rv[5] is not None:
                rv[5](obj, rv[2])
            elif hasattr(obj, "__setstate__"):
                obj.__setstate__(rv[2])
            else:
                obj.__dict__.update(rv[2])
        if n_items >= 4 and rv[3] is not None:
            if hasattr(obj, "append"):
                for item in rv[3]:
                    obj.append(item)
            elif hasattr(obj, "extend"):
                obj.extend(list(rv[3]))
                # TODO: look into efficiency choices for optional usage of extend even
                #  when append exists
            else:
                raise AttributeError(f"{obj} has neither append nor extend methods")
        if n_items >= 5 and rv[4] is not None:
            for k, v in rv[4]:
                obj[k] = v

        return obj


class Dict(Group[dict[Any, Any], dict[Any, Any], "Tuple"]):
    def pack(
        self,
        obj: dict[Any, Any],
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        self["keys"] = Tuple(self.relative("keys"), None).pack(
            tuple(obj.keys()), dispatcher, memo, references
        )

        self["values"] = Tuple(self.relative("values"), None).pack(
            tuple(obj.values()), dispatcher, memo, references
        )

        return self

    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> dict[Any, Any]:
        return {
            k.unpack(bag, memo): v.unpack(bag, memo)
            for k, v in zip(
                self["keys"].values(),
                self["values"].values(),
                strict=True,
            )
        }


class StrKeyDict(Group[dict[str, Any], dict[str, Any], Content[Any, Any]]):
    def pack(
        self,
        obj: dict[str, Any],
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        for k, v in obj.items():
            self[k] = pack_content(v, self.relative(k), dispatcher, memo, references)
        return self

    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> dict[str, Any]:
        return {k: v.unpack(bag, memo) for k, v in self.items()}


IndexableType = TypeVar(
    "IndexableType", tuple[Any, ...], list[Any], set[Any], frozenset[Any]
)


class Indexable(
    Group[IndexableType, IndexableType, Content[Any, Any]], Generic[IndexableType], ABC
):
    def pack(
        self,
        obj: IndexableType,
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        for i, item in enumerate(obj):
            self[f"i{i}"] = pack_content(
                item, self.relative(f"i{i}"), dispatcher, memo, references
            )
        return self


class Tuple(Indexable[tuple[Any, ...]]):
    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> tuple[Any, ...]:
        return tuple(child.unpack(bag, memo) for child in self.values())


class List(Indexable[list[Any]]):
    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> list[Any]:
        return [child.unpack(bag, memo) for child in self.values()]


class Set(Indexable[set[Any]]):
    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> set[Any]:
        return {child.unpack(bag, memo) for child in self.values()}


class FrozenSet(Indexable[frozenset[Any]]):
    def unpack(self, bag: Bag, memo: UnpackingMemoAlias) -> frozenset[Any]:
        return frozenset(child.unpack(bag, memo) for child in self.values())


class DirectItem(
    Item[PackingType, PackingType, PackingType], Generic[PackingType], ABC
):
    def pack(
        self,
        obj: PackingType,
        dispatcher: DispatcherAlias,
        memo: PackingMemoAlias,
        references: ReferencesAlias,
    ) -> Self:
        self.stored = obj
        return self

    def _unpack_item(self, bag: Bag, memo: UnpackingMemoAlias) -> PackingType:
        return cast(PackingType, self.stored)


class NoneItem(DirectItem[type[None]]):
    pass


class BoolItem(DirectItem[bool]):
    pass


class LongItem(DirectItem[int]):
    pass


class FloatItem(DirectItem[float]):
    pass


class ComplexItem(DirectItem[complex]):
    pass


class BytesItem(DirectItem[bytes]):
    pass


class BytearrayItem(DirectItem[bytearray]):
    pass


class StrItem(DirectItem[str]):
    pass


def content_map() -> dict[type, type[Content[Any, Any]]]:
    return {
        type: Global,
        FunctionType: Global,
        type(all): Global,
        tuple: Tuple,
        list: List,
        set: Set,
        frozenset: FrozenSet,
        type(None): NoneItem,
        bool: BoolItem,
        int: LongItem,
        float: FloatItem,
        complex: ComplexItem,
        bytes: BytesItem,
        bytearray: BytearrayItem,
        str: StrItem,
    }


def dispatch_dictionary(obj: object) -> type[Dict] | type[StrKeyDict] | None:
    if type(obj) is dict:
        if all(isinstance(k, str) for k in obj):
            return StrKeyDict
        return Dict
    return None


def minimal_dispatcher(obj: object) -> type[Content[Any, Any]] | None:
    t = type if isinstance(obj, type) else type(obj)
    con_class = content_map().get(t)
    con_class = dispatch_dictionary(obj) if con_class is None else con_class
    return con_class
