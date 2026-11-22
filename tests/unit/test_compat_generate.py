import unittest

from compat import generate


class TestGenerate(unittest.TestCase):
    def test_can_generate(self):
        self.assertTrue(generate.can_generate("0.1.0", "v0_1_0"))
        self.assertTrue(
            generate.can_generate("0.1.15.dev10+g412e88a7a", "v0_1_15"),
            msg="Dev builds of the next release generate its unreleased module",
        )
        self.assertFalse(generate.can_generate("0.1.9", "v0_1_0"))
        self.assertFalse(generate.can_generate("0.1.15.dev10+gabc", "v0_1_1"))
        self.assertFalse(generate.can_generate("0.1.150", "v0_1_15"))


if __name__ == "__main__":
    unittest.main()
