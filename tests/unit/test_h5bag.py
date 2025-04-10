import contextlib
import os
import unittest

import numpy as np
import objects
from objects import (
    DRAGON,
    CustomReduce,
    ExReducta,
    NestedParent,
    Parent,
    Recursing,
    SomeData,
)
from pyiron_snippets.dotdict import DotDict

import bagofholding.h5.content as c
from bagofholding.bag import BagMismatchError
from bagofholding.h5.bag import H5Bag, H5Info
from bagofholding.metadata import (
    EnvironmentMismatchError,
    ModuleForbiddenError,
    NoVersionError,
)


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
            EnvironmentMismatchError, msg="Fail hard when env mismatches"
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
            (bytes("some plain old bytes", encoding="utf8"), c.Bytes),
            (b"\x00", c.Bytes),  # h5py leverages the null character, so we need to
            # ensure we treat our bytes specially
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
                DRAGON,  # Singleton
            ]
        ]
        reducible_content = [
            (obj, c.Reducible)
            for obj in [
                CustomReduce(10, ["iter1", "iter2"]),  # Custom __reduce__
                ExReducta(1),  # __reduce_ex__ pickle API
                SomeData(),  # a dataclass
                Parent(),  # An object with an internally cyclic relationship
                DotDict({"forty-two": 42}),  # Inheriting from a built-in class
                NestedParent.NestedChild(),  # Requiring qualname
                Recursing(2),
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
                self.assertIs(type(obj), type(reloaded))
                self.assertTrue(np.all(obj == reloaded))
                os.remove(self.save_name)

    def test_versions_required(self):
        obj = objects.SomeData()

        H5Bag.save(obj, self.save_name, require_versions=False)
        reloaded = H5Bag(self.save_name).load()
        self.assertEqual(
            reloaded,
            obj,
            msg="The objects module is not versioned, but without requiring versions "
            "this is not supposed to matter",
        )

        with self.assertRaises(
            NoVersionError, msg="Fail hard when version is required but missing"
        ):
            H5Bag.save(obj, self.save_name, require_versions=True)

    def test_forbidden_modules(self):
        obj = objects.SomeData()

        H5Bag.save(obj, self.save_name, forbidden_modules=())
        reloaded = H5Bag(self.save_name).load()
        self.assertEqual(
            reloaded,
            obj,
            msg="The module is not forbidden, so saving should proceed fine.",
        )

        with self.assertRaises(
            ModuleForbiddenError, msg="Fail hard when module forbidden"
        ):
            H5Bag.save(
                obj, self.save_name, forbidden_modules=(obj.__module__.split(".")[0],)
            )
