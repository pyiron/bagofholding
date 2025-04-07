"""
Testing bag features.
"""

import contextlib
import os
import unittest

import numpy as np
from objects import CustomReduce, Parent, SomeData

import bagofholding.h5.content as c
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

    def test_cases(self):
        sub_union = str | bytes
        union_type = int | float | sub_union

        simple_items = [
            ("42", c.Str),
            (complex(4.0, 2.0), c.Complex),
            (True, c.Bool),
            (42, c.Long),
            (42.0, c.Float),
            # (bytes(42), c.Bytes),  # TODO: BUG!!! bytes containing \00 are broken
            (bytearray([42]), c.Bytearray),
        ]
        complex_items = [
            (np.linspace(0, 1, 3), c.Array),
        ]
        simple_groups_ex_reducible = [
            ({42: 42.0}, c.Dict),
            ({"forty-two": 42}, c.StrKeyDict),
            (union_type, c.Union),
            ((42,), c.Tuple),
            ([42.0], c.List),
            ({"42"}, c.Set),
            (frozenset({42}), c.FrozenSet),
        ]
        global_content = [
            (obj, c.Global)
            for obj in [
                int,  # type
                H5Bag,  # type
                all,  # built-in function -- types.BuiltinFunctionType
                np.array,  # built-in function array -- types.BuiltinFunctionType
                c.pack,  # function -- types.FunctionType
                np.all,  # function -- types.FunctionType
                H5Bag.read_bag_info,  # function -- types.FunctionType
            ]
        ]
        reducible_content = [
            (obj, c.Reducible)
            for obj in [
                CustomReduce(10, ["iterable_item_1", "iterable_item_2"]),
                # ^^ Custom __reduce__
                # TODO: also __reduce_ex__
                SomeData(),  # a dataclass
                Parent(),  # An object with an internally cyclic relationship
            ]
        ]

        for obj, content_type in (
            simple_items
            + complex_items
            + simple_groups_ex_reducible
            + global_content
            + reducible_content
        ):
            with self.subTest(str(obj)):
                H5Bag.save(obj, self.save_name)
                bag = H5Bag(self.save_name)
                content_name = bag.get_enriched_metadata("object")[0].split(".")[-1]
                self.assertEqual(content_type.__name__, content_name)
                reloaded = bag.load()
                self.assertTrue(np.all(obj == reloaded))
                os.remove(self.save_name)
