"""
FROZEN once 0.1.16 is released: compatibility cases first supported in 0.1.16.

See `static.compat` for the freezing rule.
"""

from __future__ import annotations

from typing import Any

from static.compat import v0_1_15


class Holder:
    """A plain, hashable attribute holder for building reference cycles."""


def _held_by(container: Any) -> Holder:
    holder = Holder()
    holder.parent = container  # type: ignore[attr-defined]
    return holder


class _Cycle:
    """
    Wraps a container in a reference cycle. Equality checks the cycle's shape: the
    back-reference must resolve to the container itself, not to an equal copy.
    """

    container: Any

    def back_reference(self) -> Any:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        return type(other) is type(self) and other.back_reference() is other.container

    __hash__ = object.__hash__


class ListSelfCycle(_Cycle):
    def __init__(self) -> None:
        self.container: list[Any] = []
        self.container.append(self.container)

    def back_reference(self) -> Any:
        return self.container[0]


class ListCycle(_Cycle):
    def __init__(self) -> None:
        self.container: list[Any] = []
        self.container.append(_held_by(self.container))

    def back_reference(self) -> Any:
        return self.container[0].parent


class DictCycle(_Cycle):
    def __init__(self) -> None:
        self.container: dict[Any, Any] = {}
        self.container[0] = _held_by(self.container)

    def back_reference(self) -> Any:
        return self.container[0].parent


class StrKeyDictCycle(_Cycle):
    def __init__(self) -> None:
        self.container: dict[str, Any] = {}
        self.container["k"] = _held_by(self.container)

    def back_reference(self) -> Any:
        return self.container["k"].parent


class SetCycle(_Cycle):
    def __init__(self) -> None:
        self.container: set[Any] = set()
        self.container.add(_held_by(self.container))

    def back_reference(self) -> Any:
        return next(iter(self.container)).parent


class BoundBuiltinMethodCycle(_Cycle):
    """E.g. matplotlib artists hold their parent's `list.remove`."""

    def __init__(self) -> None:
        self.container: list[Any] = []
        holder = Holder()
        holder.remove = self.container.remove  # type: ignore[attr-defined]
        self.container.append(holder)

    def back_reference(self) -> Any:
        return self.container[0].remove.__self__


def build() -> dict[str, Any]:
    return {
        **v0_1_15.build(),
        "cycle_list_self": ListSelfCycle(),
        "cycle_list": ListCycle(),
        "cycle_dict": DictCycle(),
        "cycle_str_key_dict": StrKeyDictCycle(),
        "cycle_set": SetCycle(),
        "cycle_bound_builtin_method": BoundBuiltinMethodCycle(),
    }
