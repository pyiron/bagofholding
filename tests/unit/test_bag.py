"""
Testing bag features.
"""

import contextlib
import os
import unittest

from bagofholding.bag import BagMismatchError
from bagofholding.h5.bag import H5Bag, H5Info


class BagVariant(H5Bag):
    @classmethod
    def get_bag_info(cls) -> H5Info:
        return H5Info(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version="Something different",
            libver_str=cls.libver_str,
        )


class TestBag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_name = "savefile.h5"

    def setUp(self):
        H5Bag.save(42, self.save_name)

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.save_name)

    def test_bag_info_check(self):
        H5Bag(self.save_name)

        with self.assertRaises(
            BagMismatchError, msg="We expect to fail hard when bag info mismatches"
        ):
            BagVariant(self.save_name)
