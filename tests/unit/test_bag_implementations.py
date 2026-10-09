import abc
import contextlib
import dataclasses
import os
import pathlib
import tempfile
import types
import unittest
import warnings
from unittest import mock

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as np_st
from static import assertions
from static.objects import (
    DRAGON,
    STALE_CLASS,
    STALE_SENTINEL,
    Parent,
    Recursing,
    SomeData,
    build_cases,
    is_a_lambda,
    make_namedtuple_class,
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


def versioned_bag_class(base: type[bag.Bag], version: str | None) -> type[bag.Bag]:
    """A subclass reporting `base`'s exact bag info, except for the version."""
    info = dataclasses.replace(base.get_bag_info(), version=version)
    return type(
        base.__name__,
        (base,),
        {"get_bag_info": classmethod(lambda cls: info)},
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

        save_name: str

        @classmethod
        @abc.abstractmethod
        def bag_class(cls) -> type[bag.Bag]: ...

        @classmethod
        def setUpClass(cls):
            cls.save_name = "savefile"

        def tearDown(self):
            for path in (self.save_name, f"{self.save_name}.h5"):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)

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

        def _save_with_bag_version(self, version: str | None) -> None:
            versioned_bag_class(self.bag_class(), version).save(42, self.save_name)

        def _open_with_bag_version(self, version: str | None, **kwargs):
            return versioned_bag_class(self.bag_class(), version)(
                self.save_name, **kwargs
            )

        def test_bag_version_default_is_semantic_minor(self):
            self._save_with_bag_version("1.2.3")
            self._open_with_bag_version("1.2.4")
            self._open_with_bag_version("1.2.4.dev1+gabc")
            with self.assertRaises(BagMismatchError):
                self._open_with_bag_version("1.3.0")

        def test_bag_version_validator_keywords(self):
            self._save_with_bag_version("1.2.3")
            with self.assertRaises(BagMismatchError):
                self._open_with_bag_version("1.2.4", bag_version_validator="exact")
            self._open_with_bag_version("1.2.3", bag_version_validator="exact")
            self._open_with_bag_version("1.9.0", bag_version_validator="semantic-major")
            self._open_with_bag_version("9.9.9", bag_version_validator="none")

        def test_bag_version_validator_callable(self):
            self._save_with_bag_version("1.2.3")
            self._open_with_bag_version(
                "9.9.9", bag_version_validator=lambda current, stored: True
            )
            with self.assertRaises(BagMismatchError):
                self._open_with_bag_version(
                    "1.2.3", bag_version_validator=lambda current, stored: False
                )

        def test_bag_version_unparseable(self):
            self._save_with_bag_version("not-a-version")
            with self.assertRaises(BagMismatchError):
                self._open_with_bag_version("1.2.3")
            self._open_floored("not-a-version", None)

        def test_bag_version_missing(self):
            self._save_with_bag_version(None)
            self._open_floored(None, None)
            with self.assertRaises(BagMismatchError):
                self._open_with_bag_version("1.2.3")

        def _open_floored(self, version: str | None, floor: str | None, **kwargs):
            floored = type(
                self.bag_class().__name__,
                (versioned_bag_class(self.bag_class(), version),),
                {
                    "min_compatible_version": floor,
                    # Floors apply to bags saved by the declaring module
                    "__module__": self.bag_class().__module__,
                },
            )
            return floored(self.save_name, **kwargs)

        def test_bag_version_floor_ignores_other_modules(self):
            self._save_with_bag_version("1.1.9")
            elsewhere = type(
                self.bag_class().__name__,
                (versioned_bag_class(self.bag_class(), "1.2.3"),),
                {"min_compatible_version": "1.2.0", "__module__": "elsewhere"},
            )
            elsewhere(self.save_name, bag_version_validator="semantic-major")

        def test_bag_version_floor(self):
            self._save_with_bag_version("1.1.9")
            with self.assertRaisesRegex(BagMismatchError, "1.2.0"):
                self._open_floored(
                    "1.2.3", "1.2.0", bag_version_validator="semantic-major"
                )
            self._open_floored("1.2.3", "1.2.0", bag_version_validator="none")

            self._save_with_bag_version("1.2.0")
            self._open_floored("1.2.3", "1.2.0", bag_version_validator="semantic-major")

        def test_bag_version_floor_ignores_own_version(self):
            # E.g. an untagged install reporting a fallback version below the floor
            self._save_with_bag_version("1.1.9")
            self._open_floored("1.1.9", "1.2.0")

        def test_bag_version_floor_needs_parseable_version(self):
            def accept_all(current, stored):
                return True

            self._save_with_bag_version("not-a-version")
            with self.assertRaisesRegex(BagMismatchError, "1.2.0"):
                self._open_floored("1.2.3", "1.2.0", bag_version_validator=accept_all)

            self._save_with_bag_version(None)
            with self.assertRaisesRegex(BagMismatchError, "1.2.0"):
                self._open_floored("1.2.3", "1.2.0", bag_version_validator=accept_all)

        def test_bag_info_non_version_fields_always_checked(self):
            self.bag_class().save(42, self.save_name)
            with self.assertRaises(BagMismatchError):
                type(
                    "BagSubclass",
                    (self.bag_class(),),
                    {"get_bag_info": classmethod(get_modified_bag_info)},
                )(self.save_name, bag_version_validator="none")

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
            expected_content_types = {
                "str": c.Str,
                "complex": c.Complex,
                "bool": c.Bool,
                "int": c.Long,
                "float": c.Float,
                "bytes": c.Bytes,
                "bytes_null": c.Bytes,
                "bytes_empty": c.Bytes,
                "bytearray": c.Bytearray,
                "int_below_int64": c.Long,
                "int_above_uint64": c.Long,
                "ndarray_float": h5c.Array,
                "dict_int_key": c.Dict,
                "dict_str_key": c.StrKeyDict,
                "dict_slash_key": c.Dict,
                "dict_empty_key": c.Dict,
                "dict_str_subclass_key": c.Dict,
                "union": c.Union,
                "tuple": c.Tuple,
                "list": c.List,
                "set": c.Set,
                "frozenset": c.FrozenSet,
                "dict_bytearrays": c.StrKeyDict,
                "dict_big_int": c.StrKeyDict,
                "dict_nonascii_key": c.Dict,
                "dict_root_slash_key": c.Dict,
                "dict_trailing_slash_key": c.Dict,
                "dict_inner_slash_key": c.Dict,
                "dict_underscore_key": c.StrKeyDict,
                "dict_surrogate_key": c.Dict,
                "global_type": c.Global,
                "global_builtin_function": c.Global,
                "global_numpy_builtin_function": c.Global,
                "global_numpy_function": c.Global,
                "global_singleton": c.Global,
                "global_nonetype": c.Global,
                "global_ellipsis_type": c.Global,
                "global_notimplemented_type": c.Global,
                "global_ellipsis": c.Global,
                "global_notimplemented": c.Global,
                "custom_reduce": c.Reducible,
                "reduce_ex": c.Reducible,
                "dataclass": c.Reducible,
                "cyclic": c.Reducible,
                "dotdict": c.Reducible,
                "builtin_subclass": c.Reducible,
                "nested_class_instance": c.Reducible,
                "recursing": c.Reducible,
                "ndarray_str": c.Reducible,
                "ndarray_bytes": c.Reducible,
                "ndarray_bytes_ragged": c.Reducible,
                "cycle_list_self": c.Reducible,
                "cycle_list": c.Reducible,
                "cycle_dict": c.Reducible,
                "cycle_str_key_dict": c.Reducible,
                "cycle_set": c.Reducible,
                "cycle_bound_builtin_method": c.Reducible,
                "cycle_tuple": c.Reducible,
                "cycle_frozenset": c.Reducible,
                "cycle_constructor_args": c.Reducible,
            }
            cases = build_cases()
            self.assertEqual(
                set(cases),
                set(expected_content_types),
                msg="Every shared case needs an expected content type",
            )
            bag_internal_globals = [
                self.bag_class(),
                c.pack,
                self.bag_class()._unpack_bag_info,
            ]

            for name, obj, content_type in [
                (name, cases[name], ctype)
                for name, ctype in expected_content_types.items()
            ] + [(str(obj), obj, c.Global) for obj in bag_internal_globals]:
                with self.subTest(name):
                    self.bag_class().save(obj, self.save_name)
                    bag = self.bag_class()(self.save_name)
                    self.assertEqual(
                        content_type.__name__, bag["object"].content_type.split(".")[-1]
                    )
                    assertions.assert_roundtrip_equal(self, obj, bag.load())
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

        def test_interior_paths_roundtrip(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                for sub in ("group", "nested/group", "/leading/slash"):
                    path = f"{file_path}/{sub}"
                    obj = Parent()
                    self.bag_class().save(obj, path)
                    self.assertEqual(obj, self.bag_class()(path).load())

        def test_multiple_bags_per_file(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                payloads = {
                    "first": Parent(),
                    "deeply/nested/second": SomeData(),
                    "third": Recursing(2),
                }
                for sub, obj in payloads.items():
                    self.bag_class().save(obj, f"{file_path}/{sub}")
                for sub, obj in payloads.items():
                    self.assertEqual(
                        obj,
                        self.bag_class()(f"{file_path}/{sub}").load(),
                        msg="Each bag in the file should round-trip independently",
                    )

        def test_interior_path_metadata_isolation(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                obj_a = np.polynomial.Polynomial([1, 2, 3])
                obj_b = SomeData()

                self.bag_class().save(obj_a, f"{file_path}/poly")
                self.bag_class().save(
                    obj_b, f"{file_path}/data", require_versions=False
                )

                bag_a = self.bag_class()(f"{file_path}/poly")
                bag_b = self.bag_class()(f"{file_path}/data")

                self.assertEqual(
                    np.__version__,
                    bag_a["object"].version,
                    msg="Object metadata should be scoped to its own bag/group",
                )
                self.assertIsNone(
                    bag_b["object"].version,
                    msg="Metadata from another bag in the same file must not bleed across",
                )
                self.assertEqual(obj_a, bag_a.load())
                self.assertEqual(obj_b, bag_b.load())

        def test_interior_path_overwrite(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                self.bag_class().save(Parent(), f"{file_path}/spot")
                # A peer bag we want to preserve across overwrites
                self.bag_class().save(SomeData(), f"{file_path}/peer")

                # overwrite=False must refuse, leaving the existing bag intact
                with self.assertRaises(FileExistsError):
                    self.bag_class().save(
                        Recursing(2),
                        f"{file_path}/spot",
                        overwrite_existing=False,
                    )
                self.assertEqual(Parent(), self.bag_class()(f"{file_path}/spot").load())

                # overwrite=True replaces just the targeted group
                self.bag_class().save(
                    Recursing(2), f"{file_path}/spot", overwrite_existing=True
                )
                self.assertEqual(
                    Recursing(2),
                    self.bag_class()(f"{file_path}/spot").load(),
                )
                self.assertEqual(
                    SomeData(),
                    self.bag_class()(f"{file_path}/peer").load(),
                    msg="Overwriting one bag must not disturb its peers",
                )

        def test_save_above_existing_bag_overwrites_descendants(self):
            """Saving a new bag at an ancestor of an existing bag wipes the
            descendant when overwriting is allowed, and refuses otherwise."""
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "nested.h5")
                self.bag_class().save(Parent(), f"{file_path}/sub/inner")

                with self.assertRaises(
                    FileExistsError,
                    msg="overwrite_existing=False must refuse to clobber the ancestor",
                ):
                    self.bag_class().save(
                        Recursing(2),
                        f"{file_path}/sub",
                        overwrite_existing=False,
                    )
                self.assertEqual(
                    Parent(),
                    self.bag_class()(f"{file_path}/sub/inner").load(),
                    msg="The inner bag must survive a refused save",
                )

                self.bag_class().save(Recursing(2), f"{file_path}/sub")
                self.assertEqual(
                    Recursing(2),
                    self.bag_class()(f"{file_path}/sub").load(),
                )
                with self.assertRaises(
                    KeyError,
                    msg="The inner bag's group must be gone after overwriting its ancestor",
                ):
                    self.bag_class()(f"{file_path}/sub/inner").load()

        def test_save_below_existing_bag_is_rejected(self):
            """A bag cannot live inside another bag -- the outer bag's metadata
            would otherwise reach into the inner bag's storage."""
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "nested.h5")
                self.bag_class().save(Parent(), f"{file_path}/sub")

                for overwrite in (True, False):
                    with self.assertRaises(
                        FileExistsError,
                        msg=(
                            "Saving below an existing bag must be rejected "
                            f"regardless of overwrite_existing={overwrite}"
                        ),
                    ):
                        self.bag_class().save(
                            Recursing(2),
                            f"{file_path}/sub/inner",
                            overwrite_existing=overwrite,
                        )
                self.assertEqual(
                    Parent(),
                    self.bag_class()(f"{file_path}/sub").load(),
                    msg="The outer bag must be untouched by the rejected saves",
                )

        def test_interior_path_missing_group_for_read(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                self.bag_class().save(Parent(), f"{file_path}/present")
                with self.assertRaises(
                    KeyError,
                    msg="Reading from a missing interior group should raise",
                ):
                    self.bag_class()(f"{file_path}/absent").load()

        def test_top_level_open_on_multibag_file(self):
            """A file-root open on a file holding only interior bags must not
            mis-detect a bag where none lives.
            """
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                self.bag_class().save(Parent(), f"{file_path}/inside")

                # No bag at the file root: instantiation should succeed cleanly
                # without falsely tripping BagMismatchError on empty attrs.
                root_bag = self.bag_class()(file_path)
                self.assertFalse(
                    hasattr(root_bag, "bag_info"),
                    msg="No bag_info should be loaded when no bag lives at the path",
                )
                # But the interior bag still loads as expected.
                self.assertEqual(
                    Parent(),
                    self.bag_class()(f"{file_path}/inside").load(),
                )

        def test_path_parsing_properties(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "data.h5")
                self.bag_class().save(Parent(), file_path)

                top = self.bag_class()(file_path)
                self.assertFalse(top.is_subpath)
                self.assertEqual("/", top.h5_group_path)
                self.assertEqual(pathlib.Path(file_path), top.h5_file_path)

                interior = self.bag_class()(f"{file_path}/sub/group")
                self.assertTrue(interior.is_subpath)
                self.assertEqual("/sub/group", interior.h5_group_path)
                self.assertEqual(pathlib.Path(file_path), interior.h5_file_path)

        def test_hdf5_extension_is_recognized(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "store.hdf5")
                obj_a = Parent()
                obj_b = Recursing(2)
                self.bag_class().save(obj_a, f"{file_path}/a")
                self.bag_class().save(obj_b, f"{file_path}/b")
                self.assertEqual(obj_a, self.bag_class()(f"{file_path}/a").load())
                self.assertEqual(obj_b, self.bag_class()(f"{file_path}/b").load())

        def _count_file_opens(self, action):
            """Run ``action`` and return the number of times h5py.File was opened."""
            import h5py as _h5py

            real_File = _h5py.File
            counter = mock.MagicMock(side_effect=real_File)
            with mock.patch.object(_h5py, "File", counter):
                action()
            return counter.call_count

        def test_save_opens_file_once_for_new_top_level(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "fresh.h5")
                opens = self._count_file_opens(
                    lambda: self.bag_class().save(Parent(), file_path)
                )
                self.assertEqual(
                    1,
                    opens,
                    msg="Saving a new top-level bag should open the file exactly once",
                )

        def test_save_opens_file_once_for_overwrite_top_level(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "fresh.h5")
                self.bag_class().save(Parent(), file_path)
                opens = self._count_file_opens(
                    lambda: self.bag_class().save(Recursing(2), file_path)
                )
                self.assertEqual(
                    1,
                    opens,
                    msg="Overwriting a top-level bag should open the file exactly once",
                )

        def test_save_opens_file_once_for_new_interior(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                # Pre-create the file with a peer bag so the second save sees an existing file.
                self.bag_class().save(Parent(), f"{file_path}/peer")
                opens = self._count_file_opens(
                    lambda: self.bag_class().save(Recursing(2), f"{file_path}/fresh")
                )
                self.assertEqual(
                    1,
                    opens,
                    msg="Adding a new interior bag should open the file exactly once",
                )

        def test_save_opens_file_once_for_overwrite_interior(self):
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "multi.h5")
                self.bag_class().save(Parent(), f"{file_path}/spot")
                self.bag_class().save(Parent(), f"{file_path}/peer")
                opens = self._count_file_opens(
                    lambda: self.bag_class().save(Recursing(2), f"{file_path}/spot")
                )
                self.assertEqual(
                    1,
                    opens,
                    msg="Overwriting an interior bag should open the file exactly once",
                )

        def test_top_level_overwrite_existing_false(self):
            self.bag_class().save(Parent(), self.save_name)
            with self.assertRaises(FileExistsError):
                self.bag_class().save(
                    Recursing(2), self.save_name, overwrite_existing=False
                )

        def test_load_nonexistent_file(self):
            """Instantiating against a missing file is fine; loading then fails."""
            with tempfile.TemporaryDirectory() as tmpdir:
                missing = os.path.join(tmpdir, "missing.h5")
                bag_ = self.bag_class()(missing)
                self.assertFalse(
                    hasattr(bag_, "bag_info"),
                    msg="No bag_info should be loaded when the file is missing",
                )
                with self.assertRaises(
                    (FileNotFoundError, OSError),
                    msg="Loading from a missing file should raise",
                ):
                    bag_.load()

        def test_save_to_non_file_location(self):
            """Saving where the target path is a directory (not a file) fails cleanly."""
            with (
                tempfile.TemporaryDirectory() as tmpdir,
                self.assertRaises(
                    FileExistsError,
                    msg="Saving into a directory path should raise FileExistsError",
                ),
            ):
                self.bag_class().save(Parent(), tmpdir)

        def test_open_creates_missing_interior_group(self):
            """`open` in a write mode creates a missing interior group."""
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "data.h5")
                bag_ = self.bag_class()(f"{file_path}/fresh")
                try:
                    group = bag_.open("a")
                    self.assertEqual("/fresh", group.name)
                finally:
                    bag_.close()

        def test_custom_file_extension(self):
            CustomExtBag = type(
                "CustomExtBag",
                (self.bag_class(),),
                {"file_extensions": (".bag",)},
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "custom.bag")
                obj_a = Parent()
                obj_b = Recursing(2)
                CustomExtBag.save(obj_a, f"{file_path}/a")
                CustomExtBag.save(obj_b, f"{file_path}/b")
                self.assertEqual(obj_a, CustomExtBag(f"{file_path}/a").load())
                self.assertEqual(obj_b, CustomExtBag(f"{file_path}/b").load())

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

        def test_early_failure_for_unimportable_builtin_type(self):
            with self.assertRaises(StringNotImportableError):
                self.bag_class().save(types.FunctionType, self.save_name)

        def test_require_importable(self):
            FactoryMade = make_namedtuple_class()

            with self.subTest("Importable objects are fine"):
                self.bag_class().save([c.pack, np.all, DRAGON], self.save_name)

            for label, obj in [
                ("Global", FactoryMade),
                ("Reducible", FactoryMade(42)),
                ("Dict keys", {FactoryMade: 42}),
                ("Dict values", {42: FactoryMade}),
                ("StrKeyDict", {"forty-two": FactoryMade}),
                ("Union", int | FactoryMade),
                ("Indexable", [FactoryMade]),
            ]:
                with self.subTest(label):
                    with self.assertRaises(
                        StringNotImportableError, msg="Should be strict by default"
                    ):
                        self.bag_class().save(obj, self.save_name)
                    # E.g. for browse-only use, or if users will re-execute code to
                    # make the object importable before loading
                    with self.assertWarns(DeprecationWarning):
                        self.bag_class().save(
                            obj, self.save_name, require_importable=False
                        )

        def test_require_importable_identity(self):
            for label, obj in [
                ("Global", STALE_CLASS),
                ("Reducible", STALE_CLASS()),
                ("String reduction", STALE_SENTINEL),
            ]:
                with self.subTest(label):
                    with self.assertRaises(
                        StringNotImportableError,
                        msg="The import path leads to a different object",
                    ):
                        self.bag_class().save(obj, self.save_name)
                    with self.assertWarns(DeprecationWarning):
                        self.bag_class().save(
                            obj, self.save_name, require_importable=False
                        )

        def test_require_importable_deprecated(self):
            for value in [True, False]:
                with (
                    self.subTest(value),
                    self.assertWarns(
                        DeprecationWarning,
                        msg="Any explicit value will break when the kwarg is removed",
                    ),
                ):
                    self.bag_class().save(42, self.save_name, require_importable=value)

            with self.subTest("Default"), warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                self.bag_class().save(42, self.save_name)

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


class TestCompatibilityFloors(unittest.TestCase):
    def test_floors(self):
        self.assertIsNone(bagofholding.h5.bag.H5Bag.min_compatible_version)
        self.assertEqual(
            "0.1.9",
            bagofholding.h5.triebag.TrieH5Bag.min_compatible_version,
            msg="TrieH5Bag type codes were renumbered in 0.1.9",
        )


class TestH5BagBagImplementation(AbstractTestNamespace.TestBagImplementation):
    @classmethod
    def bag_class(cls) -> type[bagofholding.h5.bag.H5Bag]:
        return bagofholding.h5.bag.H5Bag


class TestH5TrieBagBagImplementation(AbstractTestNamespace.TestBagImplementation):
    @classmethod
    def bag_class(cls) -> type[bagofholding.h5.triebag.TrieH5Bag]:
        return bagofholding.h5.triebag.TrieH5Bag
