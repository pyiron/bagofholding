"""
Pinned HDF5 write-format ("libver") codes and the checks around them.
"""

from __future__ import annotations

import h5py

from bagofholding import exceptions

LIBVER_HDF5_VERSIONS: dict[str, tuple[int, int]] = {
    "v108": (1, 8),
    "v110": (1, 10),
    "v112": (1, 12),
    "v114": (1, 14),
    "v200": (2, 0),
}


def _require_known(max_libver: str) -> None:
    if max_libver not in LIBVER_HDF5_VERSIONS:
        raise ValueError(
            f"Invalid max_libver {max_libver!r}; must be one of: "
            f"{', '.join(LIBVER_HDF5_VERSIONS)}"
        )


def allowed_libvers(max_libver: str) -> tuple[str, ...]:
    """The libver codes no newer than `max_libver`, oldest first."""
    _require_known(max_libver)
    codes = tuple(LIBVER_HDF5_VERSIONS)
    return codes[: codes.index(max_libver) + 1]


def resolve_save_libver(libver: str | None, max_libver: str) -> str:
    """The libver to save with: `max_libver` if `None`, else `libver` if allowed."""
    if libver is None:
        return max_libver
    allowed = allowed_libvers(max_libver)
    if isinstance(libver, str) and libver in allowed:
        return libver
    options = ", ".join(allowed)
    if libver in ("latest", "earliest"):
        raise ValueError(
            f"libver '{libver}' is not accepted: bagofholding pins HDF5 formats "
            f"explicitly. Use one of: {options}"
        )
    raise ValueError(f"Unsupported libver {libver!r}; use one of: {options}")


def check_hdf5_supports(max_libver: str) -> None:
    """Raise if the installed HDF5 is too old to write `max_libver`."""
    _require_known(max_libver)
    required = LIBVER_HDF5_VERSIONS[max_libver]
    found = tuple(h5py.version.hdf5_version_tuple[:2])
    if found < required:
        found_str = ".".join(str(n) for n in h5py.version.hdf5_version_tuple)
        raise exceptions.EnvironmentMismatchError(
            f"bagofholding requires HDF5 >= {required[0]}.{required[1]} "
            f"(for libver '{max_libver}'), found {found_str}; update h5py"
        )


def check_stored_libver(stored: str | None, max_libver: str) -> None:
    """Raise if a bag's stored libver is neither "latest" nor an allowed code."""
    if stored == "latest" or stored in allowed_libvers(max_libver):
        return
    raise exceptions.BagMismatchError(
        f"This bag was written with HDF5 format {stored!r}, which this "
        f"bagofholding does not support (supports up to {max_libver}); "
        f"update bagofholding"
    )
