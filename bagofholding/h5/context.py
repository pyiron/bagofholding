from __future__ import annotations

import os
import pathlib
from types import TracebackType
from typing import ClassVar, Literal

import h5py

from bagofholding.exceptions import FileAlreadyOpenError, FileNotOpenError


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
    _parsed_path: tuple[pathlib.Path, str] | None
    _working_root: h5py.Group | None

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
        if self._working_root is None:
            group_path = self.h5_group_path
            self._working_root = (
                self._file if group_path == "/" else self._file[group_path]
            )
        return self._working_root

    @file.setter
    def file(self, new_file: h5py.File | None) -> None:
        self._file = new_file
        self._working_root = None

    def _parse_path(self) -> tuple[pathlib.Path, str]:
        """Split :attr:`filepath` into the filesystem path and an interior group path.

        Walks up the filepath looking for a component whose suffix matches one
        of :attr:`file_extensions` or which already exists as a file. Returns
        the (file path, interior group path) pair. The interior group path is
        always returned with a leading ``"/"``; ``"/"`` itself indicates no
        interior path (the bag is rooted at the file root).

        Cached on first call: :attr:`filepath` is treated as fixed for the
        lifetime of the bag, and the parse is hot enough during packing that
        recomputing each access measurably costs stack frames (pathlib's
        ``relative_to`` is recursive on Python 3.12).
        """
        if self._parsed_path is not None:
            return self._parsed_path
        full = self.filepath.absolute()
        candidate = full
        while True:
            if candidate.suffix in self.file_extensions or candidate.is_file():
                interior_rel = full.relative_to(candidate)
                interior = str(interior_rel)
                if interior in (".", ""):
                    self._parsed_path = (candidate, "/")
                    return self._parsed_path
                self._parsed_path = (candidate, "/" + interior.replace("\\", "/"))
                return self._parsed_path
            if candidate.parent == candidate:
                self._parsed_path = (self.filepath, "/")
                return self._parsed_path
            candidate = candidate.parent

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
            raise KeyError(f"Group {group_path!r} not found in {file_path}")
        return self._file.create_group(group_path)

    def _open_for_write(self, overwrite_existing: bool) -> None:
        """Open the underlying file and prepare the target group for a fresh write.

        Combines validating the target, clearing existing data, and opening
        the file into a single :func:`h5py.File` call so a save touches the
        file only once.
        """
        file_path, group_path = self._parse_path()
        if group_path == "/":
            if os.path.exists(self.filepath):
                if overwrite_existing and os.path.isfile(self.filepath):
                    os.remove(self.filepath)
                else:
                    raise FileExistsError(
                        f"{self.filepath} already exists or is not a file."
                    )
            self._file = h5py.File(file_path, "w", libver=self.libver_str)
            return
        self._file = h5py.File(file_path, "a", libver=self.libver_str)
        try:
            if group_path in self._file:
                if overwrite_existing:
                    del self._file[group_path]
                else:
                    raise FileExistsError(
                        f"Group {group_path!r} already exists in {file_path}."
                    )
            self._file.create_group(group_path)
        except BaseException:
            self.close()
            raise

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
            self._working_root = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def maybe_decode(attr: str | bytes) -> str:
        return attr if isinstance(attr, str) else attr.decode("utf-8")
