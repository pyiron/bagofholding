"""
Testing bag features.
"""

import contextlib
import os
import unittest

import numpy as np

from bagofholding.bag import BagMismatchError
from bagofholding.h5.bag import H5Bag, H5Info
from bagofholding.metadata import EnvironmentMismatch


class BagVariant(H5Bag):
    @classmethod
    def get_bag_info(cls) -> H5Info:
        return H5Info(
            qualname=cls.__qualname__,
            module=cls.__module__,
            version=always_42(),
            libver_str=cls.libver_str,
        )


def always_42(module_name: str = "not even used") -> str:
    return "42"


class TestBag(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.save_name = "savefile.h5"

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.save_name)

    def test_bag_info_check(self):
        H5Bag.save(42, self.save_name)
        H5Bag(self.save_name)

        with self.assertRaises(
            BagMismatchError, msg="We expect to fail hard when bag info mismatches"
        ):
            BagVariant(self.save_name)

    def test_version_checking(self):
        obj = np.polynomial.Polynomial([1, 2, 3])

        H5Bag.save(obj, self.save_name)
        bag = H5Bag(self.save_name)
        self.assertEqual(
            np.__version__,
            bag["object"].version,
            msg="Object version metadata should be automatically scraped",
        )
        with self.assertRaises(
            EnvironmentMismatch, msg="Fail hard when env mismatches"
        ):
            bag.load(version_scraping={"numpy": always_42})

        self.assertEqual(
            obj,
            bag.load(version_scraping={"not_numpy": always_42}),
            msg="Ignore scrapers for irrelevant modules",
        )
