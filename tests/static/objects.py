from __future__ import annotations

import collections
from typing import Any

from static.compat import v0_1_15
from static.compat.v0_1_0 import DRAGON as DRAGON
from static.compat.v0_1_0 import Child as Child
from static.compat.v0_1_0 import CustomReduce as CustomReduce
from static.compat.v0_1_0 import Draco as Draco
from static.compat.v0_1_0 import ExReducta as ExReducta
from static.compat.v0_1_0 import MyTestStr as MyTestStr
from static.compat.v0_1_0 import NestedParent as NestedParent
from static.compat.v0_1_0 import Parent as Parent
from static.compat.v0_1_0 import Recursing as Recursing
from static.compat.v0_1_0 import SomeData as SomeData
from static.compat.v0_1_0 import SubCustomReduce as SubCustomReduce
from static.compat.v0_1_0 import SubList as SubList


def build_cases() -> dict[str, Any]:
    """All named round-trip cases, i.e. those of the newest compat module."""
    cases: dict[str, Any] = v0_1_15.build()
    return cases


is_a_lambda = lambda x: isinstance(x, int)  # noqa: E731


def make_namedtuple_class() -> type:
    """
    A class factory: the resulting class has a module and qualname, but is not actually
    importable from there.
    """
    return collections.namedtuple("FactoryMade", "x")


# Mimic re-running a definition (e.g. a notebook cell) after the original is in use
class Redefined:
    pass


STALE_CLASS = Redefined


class Redefined:  # type: ignore[no-redef]  # noqa: F811
    pass


class Sentinel:
    def __reduce__(self):
        return "SENTINEL"


SENTINEL = Sentinel()
STALE_SENTINEL = SENTINEL
SENTINEL = Sentinel()


class Holder:
    """A plain, hashable attribute holder for building reference cycles."""
