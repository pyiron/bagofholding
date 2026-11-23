"""
FROZEN once 0.1.17 is released: compatibility cases first supported in 0.1.17.

See `static.compat` for the freezing rule.
"""

from __future__ import annotations

from typing import Any

from static.compat import v0_1_16


class Slotted:
    """Only `__slots__`, no `__dict__`; one slot is deliberately left unset."""

    __slots__ = ("a", "b", "unset")

    def __init__(self, a: Any = 1, b: Any = "two") -> None:
        self.a = a
        self.b = b

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is type(self)
            and other.a == self.a
            and other.b == self.b
            and hasattr(other, "unset") == hasattr(self, "unset")
        )

    __hash__ = object.__hash__


class SlottedWithDict(Slotted):
    """No `__slots__` of its own, so instances have both slots and a `__dict__`."""

    def __init__(self, a: Any = 3, b: Any = "four") -> None:
        super().__init__(a, b)
        self.extra = [5]

    def __eq__(self, other: object) -> bool:
        return super().__eq__(other) and other.extra == self.extra  # type: ignore[attr-defined]

    __hash__ = object.__hash__


def build() -> dict[str, Any]:
    return {
        **v0_1_16.build(),
        "slotted": Slotted(),
        "slotted_with_dict": SlottedWithDict(),
    }
