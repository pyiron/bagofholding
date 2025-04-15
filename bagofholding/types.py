from __future__ import annotations

import types
from typing import Any, TypeVar

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
