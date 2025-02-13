from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, TypeAlias, cast

from bagofholding.instances.bag import Bag
from bagofholding.instances.content import (
    PATH_DELIMITER,
    Content,
    Group,
    Item,
    minimal_dispatcher,
)
from bagofholding.metadata import Metadata

NestedDictAlias: TypeAlias = dict[str, "str | NestedDictAlias"]

BOH_PREFIX: str = "BoH::"


class PickleBag(Bag):
    content: Content[Any, Any]

    def __init__(self, filepath: str | Path, *args: object, **kwargs: Any) -> None:
        super().__init__(filepath, *args, **kwargs)
        self.content = self._read_content()

    @classmethod
    def dispatch(cls, obj: object) -> type[Content[Any, Any]] | None:
        return minimal_dispatcher(obj)

    @classmethod
    def _write(cls, content: Content[Any, Any], filepath: Path) -> None:
        writeable, flat_data = cls.to_write(content, {})
        with open(filepath, "wb") as f:
            pickle.dump(
                (writeable, {k: pickle.dumps(v) for k, v in flat_data.items()}), f
            )

    @staticmethod
    def to_write(
        content: Content[Any, Any], flat_data: dict[str, Any]
    ) -> tuple[NestedDictAlias, dict[str, Any]]:
        writeable: NestedDictAlias = {
            f"{BOH_PREFIX}content_type": f"{content.__class__.__module__}.{content.__class__.__qualname__}",
        }
        if content.metadata is not None:
            for k, v in content.metadata.field_items():
                if v is not None:
                    writeable[f"{BOH_PREFIX}{k}"] = v
        if isinstance(content, Item):
            flat_data[content.path] = content.stored
        elif isinstance(content, Group):
            for k, subcontents in content.items():
                writeable[k], flat_data = PickleBag.to_write(subcontents, flat_data)
        return writeable, flat_data

    def _read_content(self) -> Any:
        with open(self.filepath, "rb") as f:
            content_dict, _ = pickle.load(f)
        return self.read_content(content_dict)

    def read_content(
        self, stringy_dict: NestedDictAlias, path: str = "object"
    ) -> Content[Any, Any]:
        # THIS IS WHERE WE DO VERSION CHECKING USING THE METADATA

        content = self._instantiate_content(
            cast(str, stringy_dict["BoH::content_type"]),
            path,
            Metadata(**self._get_metadata_kwargs(stringy_dict)),
        )
        if isinstance(content, Group):
            for k, v in stringy_dict.items():
                if isinstance(v, dict):
                    content[k] = self.read_content(v, f"{path}{PATH_DELIMITER}{k}")
        return content

    def _get_metadata_kwargs(
        self, stringy_dict: NestedDictAlias
    ) -> dict[str, str | None]:
        metadata_kwargs = {}
        for k in Metadata.__dataclass_fields__:
            v = stringy_dict.get(f"{BOH_PREFIX}{k}", None)
            if not isinstance(v, dict):
                metadata_kwargs[k] = v
        return metadata_kwargs

    def __getitem__(self, path: str) -> Content[Any, Any]:
        if not path.startswith(self.storage_root):
            raise ValueError(
                f"Invalid path: {path}. Paths must start with {self.storage_root}"
            )
        base_key, _, remaining_path = path.partition(PATH_DELIMITER)
        content = self.content
        traversed = base_key
        while len(remaining_path) > 0:
            if not isinstance(content, Group):
                raise TypeError(
                    f"Cannot read path {path} because {traversed} is not a {Group}"
                )
            child_key, _, remaining_path = remaining_path.partition(PATH_DELIMITER)
            content = content[child_key]
            traversed += PATH_DELIMITER + child_key
        return content

    def list_paths(self) -> list[str]:
        return self._get_paths(self.content, [])

    def _get_paths(self, content: Content[Any, Any], paths: list[str]) -> list[str]:
        paths.append(content.path)
        if isinstance(content, Group):
            for subgroup in content.values():
                self._get_paths(subgroup, paths)
        return paths

    def read_stored_item(self, item: Item[Any, Any, Any]) -> Any:
        with open(self.filepath, "rb") as f:
            _, data_dict = pickle.load(f)
        return pickle.loads(data_dict[item.path])
