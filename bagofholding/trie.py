import random

import h5py
import numpy as np
import pygtrie


def decompose_stringtrie(trie: pygtrie.StringTrie):
    segments: list[str] = []
    parents: list[int] = []
    is_terminal: list[bool] = []

    stack = [("", -1)]
    while stack:
        key, parent_idx = stack.pop()
        idx = len(segments)
        segment = key.split(trie._separator)[-1] if key else ""
        segments.append(segment)
        parents.append(parent_idx)
        is_terminal.append(trie.has_key(key))

        prefix = key
        children = set()
        for child_key in trie.keys(prefix=prefix):
            remainder = child_key[len(prefix):]
            if remainder:
                next_ = remainder.split(trie._separator)[1]
                to_stack = prefix + trie._separator + next_
                children.add(to_stack)

        for child in children:
            stack.append((child, idx))

    return (
        np.array(segments, dtype=h5py.string_dtype(encoding='utf-8')),
        np.array(parents, dtype=np.int32),
        np.array(is_terminal, dtype=bool)
    )


def reconstruct_stringtrie(segments, parents, is_terminal):
    trie = pygtrie.StringTrie(separator='/')
    keys = [''] * len(segments)

    for i in range(len(segments)):
        if parents[i] == -1:
            keys[i] = ''
        else:
            p = keys[parents[i]]
            keys[i] = p + ('' if p == '' else trie._separator) + segments[i]

    for i, key in enumerate(keys):
        if is_terminal[i]:
            trie[trie._separator + key] = True

    return trie


def compute_softmax_weights(n, depth_propensity, temperature=0.1):
    pos = np.linspace(0, 1, n)
    scores = (2 * depth_propensity - 1) * pos / temperature
    exp_scores = np.exp(scores - np.max(scores))  # for numerical stability
    return (exp_scores / exp_scores.sum()).tolist()

def generate_paths(n_paths, depth_propensity=0.5, temperature=0.1, seed=None, separator="/"):
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
        prefix = random.choices(
            frontier,
            weights=weights,
            k=1
        )[0]

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


    def make_trie(n: int, path_function=None, **path_kwargs) -> pygtrie.StringTrie:
        fnc = recursive_paths if path_function is None else path_function
        trie = pygtrie.StringTrie()
        for k in fnc(n, **path_kwargs):
            trie[k] = True
        return trie

    t = pygtrie.StringTrie()
    t['/a/b/c'] = True
    t['/a/b/d'] = True
    t['/x/y'] = True

    s, p, term = decompose_stringtrie(t)
    t2 = reconstruct_stringtrie(s, p, term)

    assert set(t2.keys()) == set(t.keys())

    trie = make_trie(5)
    arrays = decompose_stringtrie(trie)
    renewed = reconstruct_stringtrie(*arrays)
    assert(trie == renewed)
    print(trie)
    print(renewed)

    candidates = [decompose_stringtrie]  #, trie_to_arrays_brute, trie_to_arrays_sorted] #, trie_to_arrays_direct]
    trie = make_trie(5)
    for to_arrays in candidates:
        assert(trie == reconstruct_stringtrie(*to_arrays(trie)))


    depth_propensities = [0, 0.25, 0.5, 0.75, 1]
    sizes = np.array([2 ** n for n in range(2, 11)])

    fig, ax = plt.subplots()
    cmap = plt.get_cmap("tab10")
    colours = [cmap(i) for i, _ in enumerate(depth_propensities)]

    for colour, depth in zip(colours, depth_propensities):
        times = {c.__name__: [] for c in candidates}
        n_trials = 10
        for n in sizes:
            trie = make_trie(n, path_function=generate_paths, depth_propensity=depth)
            for fnc in candidates:
                trials = []
                for _ in range(n_trials):

                    t0 = time.perf_counter_ns()
                    arrays = fnc(trie)
                    reconstruct_stringtrie(*arrays)
                    dt = time.perf_counter_ns() - t0
                    trials.append(dt)
                times[fnc.__name__].append(np.mean(trials))

        for name, result in times.items():
            ax.plot(sizes, result, label=name+str(depth), marker="o", color=colour)
            m, b = np.polyfit(sizes, result, 1)
            print(f"{name}: {b:.3f} +/- {m:.3f}", result, m*sizes + b)
            ax.plot(sizes, m * sizes + b, linestyle="--",  color=colour)
    ax.legend()
    plt.show()