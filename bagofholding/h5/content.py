from __future__ import annotations

from typing import cast

import numpy as np

from bagofholding.content import BespokeItem, UnpackingArguments
from bagofholding.h5.bag import H5Bag
from bagofholding.h5.dtypes import H5DtypeAlias


class Array(BespokeItem[np.ndarray[tuple[int, ...], H5DtypeAlias], H5Bag]):
    @classmethod
    def _pack_item(
        cls, obj: np.ndarray[tuple[int, ...], H5DtypeAlias], bag: H5Bag, path: str
    ) -> None:
        bag.file.create_dataset(path, data=obj)

    @classmethod
    def unpack(
        cls,
        bag: H5Bag,
        path: str,
        unpacking: UnpackingArguments,
    ) -> np.ndarray[tuple[int, ...], H5DtypeAlias]:
        return cast(
            np.ndarray[tuple[int, ...], H5DtypeAlias],
            bag.file[path][()],
        )
