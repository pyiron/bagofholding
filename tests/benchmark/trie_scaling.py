import time
import unittest

import numpy as np

from bagofholding.trie import Helper, decompose_stringtrie, reconstruct_stringtrie


class TestTrieScaling(unittest.TestCase):
    def test_scaling(self):
        """
        The de/reconstruction of the trie should move from a best-case scenario of
        nearly O(N) for breadth-like tries, to a worst-case scenario of O(N^2) scaling
        of depth-like tries.
        """

        depth_propensities = [0, 0.33, 0.67, 1]
        sizes = np.array([2**n for n in range(2, 12)])
        quadratic_coeffs = []

        for depth in depth_propensities:
            times = []
            n_trials = 50
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

            coeffs = np.polyfit(sizes, times, 2)  # quadratic fit
            quadratic_coeffs.append(coeffs[0])  # leading coefficient

        for earlier, later in zip(quadratic_coeffs, quadratic_coeffs[1:]):
            self.assertLessEqual(
                earlier,
                later,
                f"Quadratic coefficient did not increase monotonically with dept propensity. {quadratic_coeffs}",
            )
