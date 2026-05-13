from __future__ import annotations

import os
import pathlib
from types import TracebackType
from typing import ClassVar, Literal

import h5py

from bagofholding.exceptions import FileAlreadyOpenError, FileNotOpenError


def _parse_h5_path(
    filepath: str | pathlib.Path, file_extensions: tuple[str, ...]
) -> tuple[pathlib.Path, str]:
    """Split a path into (filesystem path, interior HDF5 group path).

    See :meth:`HasH5FileContext._parse_path` for semantics; this is the same
    logic exposed as a free function for use outside an instance.
    """
    full = pathlib.Path(filepath).absolute()
    candidate = full
    while True:
        if candidate.suffix in file_extensions or candidate.is_file():
            interior_rel = full.relative_to(candidate)
            interior = str(interior_rel)
            if interior in (".", ""):
                return candidate, "/"
            return candidate, "/" + interior.replace("\\", "/")
        if candidate.parent == candidate:
            return pathlib.Path(filepath), "/"
        candidate = candidate.parent


def h5_target_exists(
    filepath: str | pathlib.Path, file_extensions: tuple[str, ...]
) -> bool:
    """Whether a bag is stored at the given (possibly interior) path."""
    file_path, group_path = _parse_h5_path(filepath, file_extensions)
    if group_path == "/":
        return os.path.isfile(filepath)
    if not file_path.is_file():
        return False
    with h5py.File(file_path, "r") as f:
        return group_path in f


def h5_prepare_save_target(
    filepath: str | pathlib.Path,
    overwrite_existing: bool,
    file_extensions: tuple[str, ...],
) -> None:
    """Clear an existing target before writing a bag.

    For top-level paths, this removes the file (or raises). For interior
    paths, this removes the target group inside the file (or raises) and
    leaves the rest of the file untouched.
    """
    file_path, group_path = _parse_h5_path(filepath, file_extensions)
    if group_path == "/":
        if os.path.exists(filepath):
            if overwrite_existing and os.path.isfile(filepath):
                os.remove(filepath)
            else:
                raise FileExistsError(
                    f"{filepath} already exists or is not a file."
                )
        return
    if not file_path.is_file():
        return
    with h5py.File(file_path, "a") as f:
        if group_path in f:
            if overwrite_existing:
                del f[group_path]
            else:
                raise FileExistsError(
                    f"Group {group_path!r} already exists in {file_path}."
                )


class HasH5FileContext:
    """
    A mixin class for context management with an :class:`h5py.File` object.

    Supports addressing a group inside an HDF5 file by extending the filepath
    past a recognized file extension (e.g., ``folder/file.h5/group/sub``
    refers to the group ``/group/sub`` inside ``folder/file.h5``). This allows
    storing multiple bags in a single HDF5 file. The set of recognized
    extensions is controlled by :attr:`file_extensions`.
    """

    libver_str: ClassVar[str] = "latest"
    file_extensions: ClassVar[tuple[str, ...]] = (".h5", ".hdf5")

    filepath: pathlib.Path
    _file: h5py.File | None
    _context_depth: int

    @property
    def file(self) -> h5py.Group:
        """The bag's working root group.

        When the bag's filepath has no interior group component, this is the
        HDF5 file's root group (i.e., the :class:`h5py.File` itself, which is
        an :class:`h5py.Group`). When the bag is rooted at an interior path,
        this is the corresponding sub-group.
        """
        if self._file is None:
            raise FileNotOpenError(f"{self.filepath} is not open; use `open` or `with`")
        group_path = self.h5_group_path
        if group_path == "/":
            return self._file
        return self._file[group_path]

    @file.setter
    def file(self, new_file: h5py.File | None) -> None:
        self._file = new_file

    def _parse_path(self) -> tuple[pathlib.Path, str]:
        """Split :attr:`filepath` into the filesystem path and an interior group path.

        Walks up the filepath looking for a component whose suffix matches one
        of :attr:`file_extensions` or which already exists as a file. Returns
        the (file path, interior group path) pair. The interior group path is
        always returned with a leading ``"/"``; ``"/"`` itself indicates no
        interior path (the bag is rooted at the file root).
        """
        return _parse_h5_path(self.filepath, self.file_extensions)

    @property
    def h5_file_path(self) -> pathlib.Path:
        """The filesystem path to the underlying HDF5 file."""
        return self._parse_path()[0]

    @property
    def h5_group_path(self) -> str:
        """The interior group path inside the HDF5 file.

        Returns ``"/"`` when the bag is rooted at the file root.
        """
        return self._parse_path()[1]

    @property
    def is_subpath(self) -> bool:
        """Whether the filepath points inside an HDF5 file rather than at its root."""
        return self.h5_group_path != "/"

    def open(self, mode: Literal["r", "r+", "w", "w-", "x", "a"]) -> h5py.Group:
        if self._file is not None:
            raise FileAlreadyOpenError(f"The bag at {self.filepath} is already open")
        file_path, group_path = self._parse_path()
        self._file = h5py.File(file_path, mode, libver=self.libver_str)
        if group_path == "/":
            return self._file
        if group_path in self._file:
            return self._file[group_path]
        if mode == "r":
            raise KeyError(
                f"Group {group_path!r} not found in {file_path}"
            )
        return self._file.create_group(group_path)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._context_depth -= 1
        if self._context_depth == 0:
            self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def maybe_decode(attr: str | bytes) -> str:
        return attr if isinstance(attr, str) else attr.decode("utf-8")
