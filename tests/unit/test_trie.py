import unittest

from bagofholding.trie import Helper, decompose_stringtrie, reconstruct_stringtrie


class TestTrie(unittest.TestCase):
    def test_round_trip(self):
        trie, null = Helper.make_stochastic_trie(50)
        segments, parents, values = decompose_stringtrie(trie, null)
        renewed = reconstruct_stringtrie(segments, parents, values, null)
        self.assertEqual(renewed, trie)
