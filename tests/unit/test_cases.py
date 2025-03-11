import pathlib
import unittest

import bagofholding as boh


class TestCases(unittest.TestCase):
    def setUp(self):
        self.save = pathlib.Path(__file__).parent.resolve() / "save_case.h5"

    def tearDown(self):
        self.save.unlink(missing_ok=True)

    def test_uniontype(self):
        sub_union = str | bytes
        union_type = int | float | sub_union
        boh.ClassH5Bag.save(union_type, self.save)
        reloaded = boh.ClassH5Bag(self.save).load()
        self.assertEqual(union_type, reloaded)
