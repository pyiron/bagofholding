from __future__ import annotations

import collections
from typing import Any

from static.compat import v0_1_9
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


def _post_compat_cases() -> dict[str, Any]:
    """Cases no probed release handles for every bag class (not frozen)."""
    return {
        "dict_empty_key": {"": 42},
        "global_nonetype": type(None),
        "global_ellipsis_type": type(...),
        "global_notimplemented_type": type(NotImplemented),
        "global_ellipsis": ...,
        "global_notimplemented": NotImplemented,
        "builtin_subclass": SubList([1, 2, 3]),
    }


def build_cases() -> dict[str, Any]:
    """All named round-trip cases: frozen compat sets plus newer ones."""
    return {**v0_1_9.build(), **_post_compat_cases()}


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
