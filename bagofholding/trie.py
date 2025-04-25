import random

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


def reconstruct_stringtrie(segments, parents, values, null_value, separator="/"):
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


def compute_softmax_weights(n, depth_propensity, temperature=0.1):
    pos = np.linspace(0, 1, n)
    scores = (2 * depth_propensity - 1) * pos / temperature
    exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
    return (exp_scores / exp_scores.sum()).tolist()


def generate_paths(
    n_paths, depth_propensity=0.5, temperature=0.1, seed=None, separator="/"
):
    if seed is not None:
        random.seed(seed)

    paths = []
    next_id = 0

    def new_segment():
        nonlocal next_id
        seg = f"seg{next_id}"
        next_id += 1
        return seg

    frontier = [[]]  # list of existing path prefixes

    while len(paths) < n_paths:
        # Choose existing prefix with bias towards depth
        weights = compute_softmax_weights(len(frontier), depth_propensity, temperature)
        prefix = random.choices(frontier, weights=weights, k=1)[0]

        new_path = prefix + [new_segment()]
        paths.append(separator + separator.join(new_path))

        # Allow extending this path later
        frontier.append(new_path)

    return paths


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    def recursive_paths(n, root="object", delimiter="/"):
        paths = [delimiter + root]
        for i in range(n):
            paths.append(paths[-1] + delimiter + f"{root}{i}")
        return paths

    def make_trie(
        n: int, path_function=None, **path_kwargs
    ) -> tuple[pygtrie.StringTrie, int]:
        fnc = recursive_paths if path_function is None else path_function
        trie = pygtrie.StringTrie()
        for i, k in enumerate(fnc(n, **path_kwargs)):
            trie[k] = i
        null = -1  # real values are >0
        return trie, null

    t = pygtrie.StringTrie()
    t["/a/b/c"] = 0
    t["/a/b/d"] = 1
    t["/x/y"] = 2
    null = -1

    s, p, v = decompose_stringtrie(t, null)
    t2 = reconstruct_stringtrie(s, p, v, null)

    assert set(t2.keys()) == set(t.keys())

    trie, null = make_trie(5)
    arrays = decompose_stringtrie(trie, null)
    renewed = reconstruct_stringtrie(*arrays, null)
    print(trie)
    print(renewed)
    assert trie == renewed

    candidates = [
        decompose_stringtrie
    ]  # , trie_to_arrays_brute, trie_to_arrays_sorted] #, trie_to_arrays_direct]
    trie, null = make_trie(5)
    assert trie == reconstruct_stringtrie(*decompose_stringtrie(trie, null), null)

    depth_propensities = [0, 0.25, 0.5, 0.75, 1]
    sizes = np.array([2**n for n in range(2, 11)])

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i, _ in enumerate(depth_propensities)]

    for colour, depth in zip(colours, depth_propensities):
        times = []
        n_trials = 10
        for n in sizes:
            trie, null = make_trie(
                n, path_function=generate_paths, depth_propensity=depth
            )
            trials = []
            for _ in range(n_trials):
                t0 = time.perf_counter_ns()
                s, p, v = decompose_stringtrie(trie, null)
                reconstruct_stringtrie(s, p, v, null)
                dt = time.perf_counter_ns() - t0
                trials.append(dt)
            times.append(np.mean(trials))

        ax.plot(sizes, times, label="depth=" + str(depth), marker="o", color=colour)
        m, b = np.polyfit(sizes, times, 1)
        print(f"depth {depth}: {b:.3f} +/- {m:.3f}", times, m * sizes + b)
        ax.plot(sizes, m * sizes + b, linestyle="--", color=colour)
    ax.legend()
    plt.show()
