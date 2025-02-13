from __future__ import annotations

import abc
import pathlib
from collections.abc import Iterator, Mapping
from typing import Any, ClassVar

from bagofholding.metadata import Metadata


class Bag(Mapping[str, Metadata | None], abc.ABC):
    """
    Bags are the user-facing object.
    """

    storage_root: ClassVar[str] = "object"
    filepath: pathlib.Path

    @classmethod
    @abc.abstractmethod
    def save(cls, obj: Any, filepath: str | pathlib.Path) -> None:
        pass

    def __init__(
        self, filepath: str | pathlib.Path, *args: object, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = pathlib.Path(filepath)

    @abc.abstractmethod
    def load(self, path: str = storage_root) -> Any:
        pass

    @abc.abstractmethod
    def __getitem__(self, path: str) -> Metadata | None:
        pass

    @abc.abstractmethod
    def list_paths(self) -> list[str]:
        """A list of all available content paths."""

    def __len__(self) -> int:
        return len(self.list_paths())

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_paths())
