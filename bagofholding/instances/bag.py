from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeAlias

from bagofholding.bag import Bag
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


class InstanceBag(Bag, ABC):

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

    def load(self, path: str = Bag.storage_root) -> Any:
        return unpack_content(self, path, {})

    @abstractmethod
    def read_stored_item(self, item: Item[Any, Any, Any]) -> Any:
        pass

    def __getitem__(self, path: str) -> Metadata | None:
        return self.get_content(path).metadata

    @abstractmethod
    def get_content(self, path: str) -> Content[Any, Any]:
        pass

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
