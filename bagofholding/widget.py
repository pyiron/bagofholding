from __future__ import annotations

from typing import TYPE_CHECKING

import ipytree
import traitlets

from bagofholding.content import Reducible

if TYPE_CHECKING:
    from bagofholding.bag import Bag


class BagTree(ipytree.Tree):  # type: ignore
    # Silence complaints about subclassing `Any` because of upstream issues
    """
    A widget for more convenient bag browsing inside notebooks.
    """

    # pyiron_snippets.import_alarm.ImportAlarm.__call__  is not correctly passing on
    # the hint
    def __init__(self, bag: Bag) -> None:
        super().__init__(multiple_selection=False)
        self.bag = bag
        self.root_path = bag.storage_root
        self.selected_entry = None
        self.observe(self._on_select, names=["selected_nodes"])

        self.root = ipytree.Node("Bag", open=True, disabled=True, icon="shopping-bag")
        self.add_node(self.root)

        for key, value in bag.bag_info.field_items():
            self.root.add_node(
                ipytree.Node(f"{key}: {value}", disabled=True, icon="info")
            )
        self.object = self._create_node(self.root_path)
        self.root.add_node(self.object)

    def _create_node(self, path: str) -> ipytree.Node:
        metadata = self.bag[path]
        try:
            subentries = tuple(k for k in self.bag.open_group(path))
        except TypeError:
            subentries = None

        label_base = path.split("/")[-1]
        truncated_content = metadata.content_type.lstrip(Reducible.__module__)
        label = f"{label_base} ({truncated_content})"

        icon = "file"
        style = "default"
        if metadata.content_type == f"{Reducible.__module__}.{Reducible.__qualname__}":
            icon = "code"
            style = "success"
        elif subentries is not None:
            icon = "folder"
            style = "success" if len(subentries) > 0 else "warning"

        node = ipytree.Node(
            label,
            [],
            opened=False,
            icon=icon,
            open_icon_style=style,
            close_icon_style="danger",
        )
        node.entry_metadata = metadata
        node.entry_subentries = subentries
        node.tree_metadata = {"path": path, "loaded": False}

        # Placeholder for lazy loading subentries
        if subentries and len(subentries) > 0:
            node.add_node(ipytree.Node("Loading...", disabled=True))
            node.observe(self._load_subentries, names=["opened"])

        return node

    def _load_subentries(self, change: traitlets.Bunch) -> None:
        node = change["owner"]

        if node.tree_metadata["loaded"]:
            return

        # Clear placeholder
        node.nodes = []

        if node.entry_metadata:
            for key, value in node.entry_metadata.field_items():
                if value is not None and value != "":
                    node.add_node(
                        ipytree.Node(f"{key}: {value}", disabled=True, icon="info")
                    )

        subentries = node.entry_subentries
        for sub in subentries:
            sub_node = self._create_node(f"{node.tree_metadata['path']}/{sub}")
            node.add_node(sub_node)

        node.tree_metadata["loaded"] = True

    def _on_select(self, change: traitlets.Bunch) -> None:
        if change["new"]:
            self.selected_entry = change["new"][0].tree_metadata["path"]

    def load_selected(self) -> object:
        if self.selected_entry is None:
            raise ValueError("No entry selected")
        print(f"Loading {self.selected_entry}")
        return self.bag.load(self.selected_entry)
