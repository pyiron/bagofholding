"""
FROZEN: compatibility cases saved by bagofholding 0.1.9 (H5Bag and TrieH5Bag).

See `static.compat` for the freezing rule. Do not edit.
"""

from __future__ import annotations

from typing import Any

from static.compat import v0_1_0


def build() -> dict[str, Any]:
    return {
        **v0_1_0.build(),
        "bytes_empty": b"",
        "int_below_int64": -(2**63) - 1,
        "int_above_uint64": 2**64,
        "dict_slash_key": {"forty/two": 42},
        "dict_str_subclass_key": {v0_1_0.MyTestStr("forty/two"): 42},
        "dict_root_slash_key": {"/": None},
        "dict_trailing_slash_key": {"0/": None},
        "dict_inner_slash_key": {"0/0": None},
        "dict_surrogate_key": {"\ud800": None},
    }
