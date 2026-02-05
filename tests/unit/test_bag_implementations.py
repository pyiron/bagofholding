import abc
import contextlib
import os
import tempfile
import unittest

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st
from pyiron_snippets.dotdict import DotDict
from static.objects import (
    DRAGON,
    CustomReduce,
    ExReducta,
    NestedParent,
    Parent,
    Recursing,
    SomeData,
    is_a_lambda,
)

import bagofholding.bag as bag
import bagofholding.content as c
import bagofholding.h5.bag
import bagofholding.h5.content as h5c
import bagofholding.h5.triebag
from bagofholding import (
    BagMismatchError,
    EnvironmentMismatchError,
    ModuleForbiddenError,
    NoVersionError,
    PickleProtocolError,
    StringNotImportableError,
)


def always_42(module_name: str = "not even used") -> str:
    return "42"


def get_modified_bag_info(cls: type[bag.Bag]) -> bag.BagInfo:
    return cls._bag_info_class()(
        qualname=cls.__qualname__,
        module=cls.__module__,
        version=always_42(),
    )


def numpy_array_strategy():
    return np_st.arrays(
        dtype=st.sampled_from(bagofholding.h5.dtypes.H5PY_DTYPE_WHITELIST),
        shape=np_st.array_shapes(),
    )


def leaf_strategy():
    return st.one_of(
        st.none(),
        st.text(alphabet=st.characters(blacklist_characters="\x00")),
        st.booleans(),
        st.integers(min_value=np.iinfo(np.int64).min, max_value=np.iinfo(np.int64).max),
        st.floats(allow_nan=False, allow_infinity=False),
        st.complex_numbers(allow_nan=False, allow_infinity=False),
        st.binary(),
        st.binary().map(bytearray),
        numpy_array_strategy(),
    )


class AbstractTestNamespace:

    class TestBagImplementation(unittest.TestCase, abc.ABC):
        """
        A generic bag test which should pass for all implementations of Bag.
        """

        @classmethod
        @abc.abstractmethod
        def bag_class(cls) -> type[bag.Bag]: ...

        @classmethod
        def setUpClass(cls):
            cls.save_name = "savefile"

        def tearDown(self):
            with contextlib.suppress(FileNotFoundError):
                os.remove(self.save_name)

        def test_bag_info_check(self):
            self.bag_class().save(42, self.save_name)
            self.bag_class()(self.save_name)

            with self.assertRaises(
                BagMismatchError, msg="We expect to fail hard when bag info mismatches"
            ):
                type(
                    "BagSubclass",
                    (self.bag_class(),),
                    {"get_bag_info": classmethod(get_modified_bag_info)},
                )(self.save_name)

        def test_version_checking(self):
            obj = np.polynomial.Polynomial([1, 2, 3])

            self.bag_class().save(obj, self.save_name)
            bag = self.bag_class()(self.save_name)
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
                (b"", c.Bytes),  # h5py can give size complaints about zero-length
                # bytes, so test our the special handling there
                (bytearray([42]), c.Bytearray),
            ]
            complex_items = [
                (np.linspace(0, 1, 3), h5c.Array),
                # (, h5c.Array)
            ]
            simple_groups_ex_reducible = [
                ({42: 42.0}, c.Dict),
                ({"forty-two": 42}, c.StrKeyDict),
                ({"forty/two": 42}, c.Dict),
                (union_type, c.Union),
                ((42,), c.Tuple),
                ([42.0], c.List),
                ({"42"}, c.Set),
                (frozenset({42}), c.FrozenSet),
                ({"0": bytearray(b"\x00"), "1": bytearray(b"")}, c.Dict),
                ({"0": 282574505116416}, c.Dict),  # Just a big int
                ({"Ă": None}, c.Dict),
                ({"/": None}, c.Dict),
                ({"0/": None}, c.Dict),
                ({"0/0": None}, c.Dict),
                ({"_0": None}, c.Dict),
                ({"\ud800": None}, c.Dict),
            ]
            global_content = [
                (obj, c.Global)
                for obj in [
                    int,  # type
                    self.bag_class(),  # type
                    all,  # built-in function -- types.BuiltinFunctionType
                    np.array,  # built-in function array -- types.BuiltinFunctionType
                    c.pack,  # function -- types.FunctionType
                    np.all,  # function -- types.FunctionType
                    self.bag_class()._unpack_bag_info,  # function -- types.FunctionType
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
                    # Arrays of str and bytes types get special treatment
                    np.array([""], dtype="<U1"),  # string
                    np.array([b""], dtype="|S1"),  # byte
                    np.array([b"a", b"abc"], dtype="|S3"),  # different lengths
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
                    self.bag_class().save(obj, self.save_name)
                    bag = self.bag_class()(self.save_name)
                    self.assertEqual(
                        content_type.__name__, bag["object"].content_type.split(".")[-1]
                    )
                    reloaded = bag.load()
                    self.assertIs(type(obj), type(reloaded))
                    self.assertTrue(
                        np.all(obj == reloaded),
                        msg=f"Mismatch between {obj} and reloaded {reloaded}",
                    )
                    os.remove(self.save_name)

        def test_versions_required(self):
            obj = SomeData()

            self.bag_class().save(obj, self.save_name, require_versions=False)
            reloaded = self.bag_class()(self.save_name).load()
            self.assertEqual(
                reloaded,
                obj,
                msg="The objects module is not versioned, but without requiring versions "
                "this is not supposed to matter",
            )

            with self.assertRaises(
                NoVersionError, msg="Fail hard when version is required but missing"
            ):
                self.bag_class().save(obj, self.save_name, require_versions=True)

        def test_forbidden_modules(self):
            obj = SomeData()

            self.bag_class().save(obj, self.save_name, forbidden_modules=())
            reloaded = self.bag_class()(self.save_name).load()
            self.assertEqual(
                reloaded,
                obj,
                msg="The module is not forbidden, so saving should proceed fine.",
            )

            with self.assertRaises(
                ModuleForbiddenError, msg="Fail hard when module forbidden"
            ):
                self.bag_class().save(
                    obj,
                    self.save_name,
                    forbidden_modules=(obj.__module__.split(".")[0],),
                )

        def test_list_paths(self):
            self.bag_class().save(Parent(), self.save_name)
            paths = self.bag_class()(self.save_name).list_paths()
            self.assertSetEqual(
                {
                    "object",
                    "object/args",
                    "object/args/i0",
                    "object/constructor",
                    "object/item_iterator",
                    "object/kv_iterator",
                    "object/state",
                    "object/state/child",
                    "object/state/child/args",
                    "object/state/child/args/i0",
                    "object/state/child/constructor",
                    "object/state/child/item_iterator",
                    "object/state/child/kv_iterator",
                    "object/state/child/state",
                    "object/state/child/state/data",
                    "object/state/child/state/modified_data",
                    "object/state/child/state/name",
                    "object/state/child/state/parent",
                    "object/state/data",
                    "object/state/data/i0",
                    "object/state/data/i1",
                    "object/state/data/i2",
                    "object/state/name",
                },
                set(paths),
                msg=f"Got instead {paths}",
            )

        def test_subaccess(self):
            r = Recursing(2)
            self.bag_class().save(r, self.save_name)
            self.assertEqual(
                r.label,
                self.bag_class()(self.save_name).load("object/state/label"),
                msg="We allow loading only part of the object",
            )

        def test_bad_protocol(self):
            with self.assertRaises(
                PickleProtocolError, msg="We don't support out of band data transfers"
            ):
                self.bag_class().save(42, self.save_name, _pickle_protocol=5)

        def test_early_failure_for_lambda(self):
            with self.assertRaises(StringNotImportableError):
                self.bag_class().save(is_a_lambda, self.save_name)

        def test_early_failure_for_locals(self):
            def this_cannot_be_reimported(x):
                return x + 1

            with self.assertRaises(StringNotImportableError):
                self.bag_class().save(this_cannot_be_reimported, self.save_name)

        @settings(suppress_health_check=[HealthCheck.differing_executors])
        @given(
            data=st.recursive(
                leaf_strategy(),
                lambda children: st.dictionaries(
                    keys=st.text(
                        alphabet=st.characters(blacklist_characters="\x00."), min_size=1
                    ),
                    values=children,
                    min_size=1,
                ),
                max_leaves=10,
            )
        )
        def test_hypothesis(self, data):
            with tempfile.TemporaryDirectory() as tmpdir:
                path = os.path.join(tmpdir, "test.h5")
                self.bag_class().save(data, filepath=path)
                loaded_data = self.bag_class()(path).load()

            self.assert_equal_recursive(data, loaded_data)

        def assert_equal_recursive(self, a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                self.assertEqual(len(a), len(b))
                for key in a:
                    self.assertIn(key, b)
                    self.assert_equal_recursive(a[key], b[key])
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                try:
                    self.assertTrue(np.array_equal(a, b, equal_nan=True))
                # np.isnan may complain on some non numerica dtypes
                except TypeError:
                    self.assertTrue(np.array_equal(a, b))
            else:
                self.assertEqual(a, b)


class TestH5BagBagImplementation(AbstractTestNamespace.TestBagImplementation):
    @classmethod
    def bag_class(cls) -> type[bagofholding.h5.bag.H5Bag]:
        return bagofholding.h5.bag.H5Bag


class TestH5TrieBagBagImplementation(AbstractTestNamespace.TestBagImplementation):
    @classmethod
    def bag_class(cls) -> type[bagofholding.h5.triebag.TrieH5Bag]:
        return bagofholding.h5.triebag.TrieH5Bag
