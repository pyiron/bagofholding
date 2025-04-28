import random
from typing import cast

import numpy as np
import pygtrie


def decompose_stringtrie[
    ValueType
](trie: pygtrie.StringTrie, null_value: ValueType) -> tuple[
    list[str], list[int], list[ValueType]
]:
    segments: list[str] = []
    parents: list[int] = []
    values: list[ValueType] = []

    stack = [("", -1)]
    while stack:
        key, parent_idx = stack.pop()
        idx = len(segments)
        segment = key.split(trie._separator)[-1] if key else ""
        segments.append(segment)
        parents.append(parent_idx)
        values.append(
            trie.values(prefix=key, shallow=True)[0]
            if trie.has_key(key)
            else null_value
        )

        prefix = key
        children = set()
        for child_key in trie.keys(prefix=prefix):
            remainder = child_key[len(prefix) :]
            if remainder:
                next_ = remainder.split(trie._separator)[1]
                to_stack = prefix + trie._separator + next_
                children.add(to_stack)

        for child in children:
            stack.append((child, idx))

    return segments, parents, values


def reconstruct_stringtrie[ValueType](
        segments: list[str],
        parents: list[int],
        values: list[ValueType],
        null_value: ValueType,
        separator: str = "/",
) -> pygtrie.StringTrie:
    trie = pygtrie.StringTrie(separator=separator)
    keys = [""] * len(segments)

    for i in range(len(segments)):
        if parents[i] == -1:
            keys[i] = ""
        else:
            p = keys[parents[i]]
            keys[i] = p + ("" if p == "" else trie._separator) + segments[i]

    for i, key in enumerate(keys):
        if values[i] != null_value:
            trie[trie._separator + key] = values[i]

    return trie


class Helper:
    @staticmethod
    def compute_softmax_weights(n: int, depth_propensity: float, temperature: float = 0.1) -> list[float]:
        pos = np.linspace(0, 1, n)
        scores = (2 * depth_propensity - 1) * pos / temperature
        exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
        return cast(list[float], (exp_scores / exp_scores.sum()).tolist())

    @classmethod
    def generate_paths(
        cls, n_paths: int, depth_propensity: float = 0.5, temperature: float = 0.1, seed: int | None = None, separator: str = "/"
    ) -> list[str]:
        if seed is not None:
            random.seed(seed)

        paths: list[str] = []
        next_id = 0

        def new_segment() -> str:
            nonlocal next_id
            seg = f"seg{next_id}"
            next_id += 1
            return seg

        frontier: list[list[str]] = [[]]  # list of existing path prefixes

        while len(paths) < n_paths:
            # Choose existing prefix with bias towards depth
            weights = cls.compute_softmax_weights(len(frontier), depth_propensity, temperature)
            prefix = random.choices(frontier, weights=weights, k=1)[0]

            new_path = prefix + [new_segment()]
            paths.append(separator + separator.join(new_path))

            # Allow extending this path later
            frontier.append(new_path)

        return paths

    @classmethod
    def make_stochastic_trie(cls, n_paths: int, depth_propensity: float = 0.5, temperature: float = 0.1) -> tuple[pygtrie.StringTrie, int]:
        trie = pygtrie.StringTrie()
        for i, k in enumerate(cls.generate_paths(n_paths, depth_propensity, temperature)):
            trie[k] = i
        null = -1  # real values are >0
        return trie, null
