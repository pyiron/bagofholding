import random
from typing import TypeVar, cast

import numpy as np
import pygtrie

ValueType = TypeVar("ValueType")  # python <3.12 compatibility
# def not_allowed_to_have_inline[FunctionGenerics](in_python=3.12):


def decompose_stringtrie(
    trie: pygtrie.StringTrie, null_value: ValueType
) -> tuple[list[str], list[int], list[ValueType]]:
    """
    While tries are a useful representation of path-like tree data, storing the
    individual keys is not space-efficient.

    This function decomposes the trie into a linked-list structure of individual path
    segments and their upstream
    components. Each segment then gets its associated value. Since not all individual
    segments are necessarily valid
    paths in the original tree, we also require a "null" value to keep all components
    the same length.

    Args:
        trie (pygtrie.StringTrie): The trie to decompose.
        null_value (ValueType): What to store in lieu of a value for segments which do
            not correspond to trie paths. Null values must implement a meaningful `==`
            operator with the values of the trie.

    Returns:
        list[str]: The path segments.
        list[int]: Each segment's parent id in the segments list.
        list[ValueType]: The value associated with each segment (taking the null value
            whenever the segment has no associated path in the source trie).
    """
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


def reconstruct_stringtrie(
    segments: list[str],
    parents: list[int],
    values: list[ValueType],
    null_value: ValueType,
    separator: str = "/",
) -> pygtrie.StringTrie:
    """

    While tries are a useful representation of path-like tree data, storing the
    individual keys is not space-efficient.

    This function reconstructs a trie based on a linked-list structure of individual
    path segments and their upstream parent segments.
    Since it is possible that not all segments are intended to be terminal paths in the
    final trie, any segments whose value matches the provided "null" value will not
    actually be added to the resulting trie.

    Args:
        segments (list[str]): The path segments.
        parents (list[int]): The parent ids in the segments list.
        values (list[ValueType]): The corresponding values to assign to the trie paths
            (or do not add if equal to null values).
        null_value (ValueType): The null value against which to compare values to see
            if they should be excluded from the final trie. Null values must implement
            a meaningful `==` operator with elements of the values list.
        separator (str): The path separator.

    Returns:
        (pygtrie.StringTrie): The resulting trie.
    """

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
    """
    Useful tools for building tries on which to test the decomposition of path-like
    tree data.
    """

    @staticmethod
    def compute_softmax_weights(
        n: int, depth_propensity: float, temperature: float = 0.1
    ) -> list[float]:
        pos = np.linspace(0, 1, n)
        scores = (2 * depth_propensity - 1) * pos / temperature
        exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
        return cast(list[float], (exp_scores / exp_scores.sum()).tolist())

    @classmethod
    def generate_paths(
        cls,
        n_paths: int,
        depth_propensity: float = 0.5,
        temperature: float = 0.1,
        seed: int | None = None,
        separator: str = "/",
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
            weights = cls.compute_softmax_weights(
                len(frontier), depth_propensity, temperature
            )
            prefix = random.choices(frontier, weights=weights, k=1)[0]

            new_path = prefix + [new_segment()]
            paths.append(separator + separator.join(new_path))

            # Allow extending this path later
            frontier.append(new_path)

        return paths

    @classmethod
    def make_stochastic_trie(
        cls, n_paths: int, depth_propensity: float = 0.5, temperature: float = 0.1
    ) -> tuple[pygtrie.StringTrie, int]:
        trie = pygtrie.StringTrie()
        for i, k in enumerate(
            cls.generate_paths(n_paths, depth_propensity, temperature)
        ):
            trie[k] = i
        null = -1  # real values are >0
        return trie, null
