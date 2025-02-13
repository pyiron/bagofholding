import unittest

import bagofholding


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = bagofholding.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
