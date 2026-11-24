"""
Generate the HDF5 write-format (libver) artefacts.

Usage, from the repository root, in the dev environment:

    python tests/libver/generate.py

Writes one bag per allowed libver code per bag class to
`static.libver.ARTEFACT_DIR`, plus one "future" bag per class to
`static.libver.FUTURE_DIR`. The future bags hold a payload written at the newest
format the generating HDF5 supports but claim the unsupported code "v999".

Needs an HDF5 that can write `max_libver`. Never run in CI. Rerun whenever
`max_libver` is raised.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Any

TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent

FUTURE_CODE = "v999"


def newest_supported_code() -> str:
    import h5py

    from bagofholding.h5 import format_versions

    found = tuple(h5py.version.hdf5_version_tuple[:2])
    supported = [
        code
        for code, version in format_versions.LIBVER_HDF5_VERSIONS.items()
        if version <= found
    ]
    return supported[-1]


def future_bag_class(base: Any) -> Any:
    """
    A throwaway subclass of `base` that reports `base`'s exact bag info, writes
    its payload at the newest supported format, and stores a fake future code.
    """
    from bagofholding import bag as bag_module

    info = base.get_bag_info()

    def _pack_bag_info(self: Any) -> None:
        self._pack_fields(
            dataclasses.replace(info, libver_str=FUTURE_CODE),
            bag_module.PATH_DELIMITER,
        )

    return type(
        base.__name__,
        (base,),
        {
            "max_libver": newest_supported_code(),
            "get_bag_info": classmethod(lambda cls: info),
            "_pack_bag_info": _pack_bag_info,
        },
    )


def main() -> None:
    sys.path.insert(0, str(TESTS_DIR))
    import static.libver as payload

    from bagofholding.h5 import format_versions

    obj = payload.build()
    payload.ARTEFACT_DIR.mkdir(exist_ok=True)
    payload.FUTURE_DIR.mkdir(exist_ok=True)
    for name, bag_class in payload.BAG_CLASSES.items():
        for code in format_versions.allowed_libvers(bag_class.max_libver):
            path = payload.ARTEFACT_DIR / f"{code}.{name}.h5"
            bag_class.save(obj, path, libver=code)
            print(f"Wrote {path}")
        path = payload.FUTURE_DIR / f"future.{name}.h5"
        future_bag_class(bag_class).save(obj, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
