from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, ClassVar, TypeAlias

from bagofholding.exception import BagOfHoldingError
from bagofholding.instances.content import (
    Content,
    Item,
    pack_content,
    unpack_content,
)
from bagofholding.metadata import Metadata
from bagofholding.retrieve import import_from_string

RetrieverAlias: TypeAlias = dict[type[Item[Any, Any, Any]], Callable[[Path, str], Any]]


class Bag(Mapping[str, Content[Any, Any]], ABC):

    storage_root: ClassVar[str] = "object"
    filepath: Path

    @classmethod
    def save(cls, obj: Any, filepath: str | Path) -> None:
        cls._write(cls._pack(obj), Path(filepath))

    @classmethod
    def _pack(cls, obj: Any) -> Content[Any, Any]:
        return pack_content(obj, cls.storage_root, cls.dispatch)

    @classmethod
    @abstractmethod
    def dispatch(cls, obj: object) -> type[Content[Any, Any]] | None:
        pass

    @classmethod
    @abstractmethod
    def _write(cls, content: Content[Any, Any], filepath: Path) -> None:
        pass

    def __init__(self, filepath: str | Path, *args: object, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.filepath = Path(filepath)

    def load(self, path: str = storage_root) -> Any:
        return unpack_content(self, path, {})

    @abstractmethod
    def read_stored_item(self, item: Item[Any, Any, Any]) -> Any:
        pass

    @abstractmethod
    def __getitem__(self, path: str) -> Content[Any, Any]:
        pass

    @abstractmethod
    def list_paths(self) -> list[str]:
        """A list of all available content paths."""

    def __len__(self) -> int:
        return len(self.list_paths())

    def __iter__(self) -> Iterator[str]:
        return iter(self.list_paths())

    def _instantiate_content(
        self, content_class_string: str, path: str, metadata: Metadata | None
    ) -> Content[Any, Any]:
        content = import_from_string(content_class_string)(path, metadata)
        if isinstance(content, Content):
            return content
        else:
            raise BagOfHoldingError(
                f"Expected to import a content class, but got {content}"
            )
