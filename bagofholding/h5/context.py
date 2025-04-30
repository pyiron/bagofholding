import pathlib
from types import TracebackType
from typing import ClassVar, Literal

import h5py

from bagofholding.exceptions import FileAlreadyOpenError, FileNotOpenError


class HasH5FileContext:
    """
    A mixin class for context management with an :class:`h5py.File` object.
    """

    libver_str: ClassVar[str] = "latest"

    filepath: pathlib.Path
    _file: h5py.File | None
    _context_depth: int

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            raise FileNotOpenError(f"{self.filepath} is not open; use `open` or `with`")
        return self._file

    @file.setter
    def file(self, new_file: h5py.File | None) -> None:
        self._file = new_file

    def open(self, mode: Literal["r", "r+", "w", "w-", "x", "a"]) -> h5py.File:
        if self._file is None:
            self.file = h5py.File(self.filepath, mode, libver=self.libver_str)
            return self.file
        else:
            raise FileAlreadyOpenError(f"The bag at {self.filepath} is already open")

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
            self.file.close()
            self._file = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def maybe_decode(attr: str | bytes) -> str:
        return attr if isinstance(attr, str) else attr.decode("utf-8")
