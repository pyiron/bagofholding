# Designed and intended for use with H5 implementations, which treat arrays very nicely
# but since bags merely follow a protocol, there is nothing intrinsically-h5 here
from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np

from bagofholding.content import BespokeItem, Packer, UnpackingArguments
from bagofholding.h5.dtypes import H5DtypeAlias

ArrayType: TypeAlias = np.ndarray[tuple[int, ...], H5DtypeAlias]


class ArrayPacker(Packer, Protocol):
    def pack_array(self, obj: ArrayType, path: str) -> None: ...
    def unpack_array(self, path: str) -> ArrayType: ...


class Array(BespokeItem[ArrayType, ArrayPacker]):
    @classmethod
    def _pack_item(cls, obj: ArrayType, packer: ArrayPacker, path: str) -> None:
        packer.pack_array(obj, path)

    @classmethod
    def unpack(
        cls, packer: ArrayPacker, path: str, unpacking: UnpackingArguments
    ) -> ArrayType:
        return packer.unpack_array(path)
