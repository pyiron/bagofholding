from __future__ import annotations

import types
from typing import Any, TypeVar, Union

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
BuiltinItemUnion = Union[str, bytes, bytearray, bool, int, float, complex]

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
BuiltinGroupUnion = Union[
    dict[Any, Any],
    dict[str, Any],
    types.UnionType,
    tuple[Any, ...],
    list[Any],
    set[Any],
    frozenset[Any],
]
