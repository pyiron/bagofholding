from __future__ import annotations

import dataclasses
from typing import SupportsIndex

import numpy as np
from pyiron_snippets.singleton import Singleton


@dataclasses.dataclass
class SomeData:
    foo: str = "foo"
    bar: int = 42


class CustomReduce:
    def __init__(self, value, items: list[str] | None = None):
        self.value = value
        self.child = SubCustomReduce(value * 2)
        self.items = [] if items is None else items

    def __reduce__(self):
        reconstructor = self.__class__
        args = (self.value,)
        state = {"extra_info": "meta", "child_state": self.child.get_state()}
        return (
            reconstructor,
            args,
            state,
            iter(self.items),
            None,
        )

    def append(self, item):
        self.items.append(item)

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and other.value == self.value
            and other.child == self.child
            and other.items == self.items
        )


class SubCustomReduce:
    def __init__(self, value):
        self.value = value

    def __reduce__(self):
        return (self.__class__, (self.value,), self.get_state())

    def get_state(self):
        return {"info": f"sub value is {self.value}"}

    def __eq__(self, other):
        return other.__class__ == self.__class__ and other.value == self.value


class Parent:
    def __init__(self, name="p", data=None):
        self.name = name
        self.data = [10, 11, 12] if data is None else data
        self.child = Child("c", self)

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and other.name == self.name
            and other.data == self.data
            and other.child == self.child
        )


class Child:
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
        self.data = parent.data
        self.modified_data = parent.data[0]

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and other.name == self.name
            and other.parent.__class__ == self.parent.__class__
            and other.parent.name == self.parent.name
            and other.parent.data == self.parent.data
        )


class NestedParent:
    """This is importable right from the module."""

    class NestedChild:
        """This is not."""

        def __init__(self):
            self.data = "You can't import this from storables"

        def __eq__(self, other):
            return other.__class__ == self.__class__ and other.data == self.data


class Draco(metaclass=Singleton):  # type: ignore[misc]
    def __init__(self, how_many="I am the last one"):
        self.how_many = how_many

    def __eq__(self, other):
        return other is self

    def __reduce__(self):
        return "DRAGON"


DRAGON = Draco()


class ExReducta:
    def __init__(self, n: int):
        self.n = n

    def __reduce_ex__(self, protocol: SupportsIndex):
        return self.__class__, (self.n + 1,)

    def __reduce__(self):
        raise RuntimeError(
            "Python should never get here because an explicit __reduce_ex__ takes precedence"
        )

    def __eq__(self, other):
        # Equality is designed for a reloaded object to equate a stored object
        return other.__class__ == self.__class__ and other.n == self.n + 1


class Recursing:

    def __init__(self, n: int):
        self.child: Recursing | None

        if n > 0:
            self.child = Recursing(n - 1)
        else:
            self.child = None

        self.data = np.arange(n)
        self.label = f"Recursive{n}"

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return (
            other.__class__ == self.__class__
            and np.all(other.data == self.data)
            and other.label == self.label
            and other.child == self.child
        )


is_a_lambda = lambda x: isinstance(x, int)  # noqa: E731
