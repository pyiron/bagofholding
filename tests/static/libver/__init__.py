"""
Frozen payload for the HDF5 write-format (libver) artefacts.

`artefacts/` holds one bag per allowed libver code per bag class, and `future/`
holds one bag per class claiming an unsupported (newer) code. The payload covers
the types whose on-disk encoding depends on the HDF5 format version. Changing
`build` requires regenerating the artefacts with `tests/libver/generate.py`.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from bagofholding import H5Bag, TrieH5Bag
from bagofholding import bag as bag_module

ARTEFACT_DIR = pathlib.Path(__file__).resolve().parent / "artefacts"
FUTURE_DIR = pathlib.Path(__file__).resolve().parent / "future"

BAG_CLASSES: dict[str, type[bag_module.Bag]] = {"H5Bag": H5Bag, "TrieH5Bag": TrieH5Bag}


def build() -> dict[str, Any]:
    return {
        "complex128_array": np.array([1 + 2j]),
        "complex64_array": np.array([1j], dtype=np.complex64),
        "bool": True,
        "np_bool": np.bool_(False),
        "bool_array": np.array([True, False]),
        "surrogate_str": "\ud800",
        "bigint": 2**70,
        "complex": 1 + 2j,
        "nested": {"a": [1, True, 2.5, "s"]},
    }
