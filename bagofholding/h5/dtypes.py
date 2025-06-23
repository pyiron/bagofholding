"""
h5py supports efficient treatment of a subset of possible array.

Here, we whitelist these types.
"""

from typing import TypeAlias

import numpy as np

H5PY_DTYPE_WHITELIST = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.bool_,
    np.bytes_,
    np.str_,
)


H5Scalar: TypeAlias = (
    np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
    | np.complex64
    | np.complex128
    | np.bool_
    | np.bytes_
    | np.str_
)


IntTypesAlias: TypeAlias = (
    np.dtype[np.int8]
    | np.dtype[np.int16]
    | np.dtype[np.int32]
    | np.dtype[np.int64]
    | np.dtype[np.uint8]
    | np.dtype[np.uint16]
    | np.dtype[np.uint32]
    | np.dtype[np.uint64]
)


H5DtypeAlias: TypeAlias = (
    IntTypesAlias
    | np.dtype[np.float16]
    | np.dtype[np.float32]
    | np.dtype[np.float64]
    | np.dtype[np.complex64]
    | np.dtype[np.complex128]
    | np.dtype[np.bool_]
    | np.dtype[np.bytes_]
    | np.dtype[np.str_]
)
