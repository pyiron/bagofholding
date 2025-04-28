import time
import unittest

import matplotlib.pyplot as plt
import numpy as np

from bagofholding.trie import decompose_stringtrie, reconstruct_stringtrie, Helper


class TestTrieScaling(unittest.TestCase):
    def test_scaling(self):
        depth_propensities = [0, 0.25, 0.5, 0.75, 1]
        sizes = np.array([2 ** n for n in range(2, 11)])

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        colours = [cmap(i) for i, _ in enumerate(depth_propensities)]

        for colour, depth in zip(colours, depth_propensities):
            times = []
            n_trials = 20
            for n in sizes:
                trie, null = Helper.make_stochastic_trie(n, depth_propensity=depth)
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