"""
FROZEN once 0.1.15 is released: compatibility cases first supported in 0.1.15.

See `static.compat` for the freezing rule.
"""

from __future__ import annotations

from typing import Any

from static.compat import v0_1_0, v0_1_9


def build() -> dict[str, Any]:
    return {
        **v0_1_9.build(),
        "dict_empty_key": {"": 42},
        "global_nonetype": type(None),
        "global_ellipsis_type": type(...),
        "global_notimplemented_type": type(NotImplemented),
        "global_ellipsis": ...,
        "global_notimplemented": NotImplemented,
        "builtin_subclass": v0_1_0.SubList([1, 2, 3]),
    }
