import abc
import contextlib
import dataclasses
import os
import unittest
from typing import Any, cast
from unittest import mock

import h5py

import bagofholding.bag as bag
import bagofholding.h5.bag
import bagofholding.h5.triebag
from bagofholding import BagMismatchError, EnvironmentMismatchError
from bagofholding.h5.bag import H5Info


class TestH5InfoEquality(unittest.TestCase):
    def test_h5info_equality_ignores_libver(self):
        a = H5Info(qualname="q", module="m", version="1", libver_str="latest")
        b = dataclasses.replace(a, libver_str="v110")
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))


class AbstractTestNamespace:

    class TestLibverLoad(unittest.TestCase, abc.ABC):
        save_name = "libver_savefile"

        @classmethod
        @abc.abstractmethod
        def bag_class(cls) -> type[bag.Bag]: ...

        def tearDown(self):
            for path in (self.save_name, f"{self.save_name}.h5"):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)

        def _save_with_stored_libver(
            self, code: str | None, **info_overrides: Any
        ) -> None:
            base = self.bag_class()
            info = dataclasses.replace(
                cast(H5Info, base.get_bag_info()), libver_str=code, **info_overrides
            )

            def _pack_bag_info(this):
                this._pack_fields(info, bag.PATH_DELIMITER)

            subclass: Any = type(
                base.__name__,
                (base,),
                {
                    "get_bag_info": classmethod(lambda cls: base.get_bag_info()),
                    "_pack_bag_info": _pack_bag_info,
                },
            )
            subclass.save(42, self.save_name)

        def test_stored_latest_loads(self):
            self._save_with_stored_libver("latest")
            self.assertEqual(42, self.bag_class()(self.save_name).load())

        def test_stored_lower_code_loads(self):
            self._save_with_stored_libver("v110")
            self.assertEqual(42, self.bag_class()(self.save_name).load())

        def test_stored_unknown_code_rejected(self):
            for code in ("v220", "v200"):
                with self.subTest(code=code):
                    self._save_with_stored_libver(code)
                    with self.assertRaisesRegex(
                        BagMismatchError, "update bagofholding"
                    ):
                        self.bag_class()(self.save_name)
                    os.remove(self.save_name)

        def test_libver_message_wins_over_newer_minor_version(self):
            self._save_with_stored_libver("v999", version="99.0.0")
            with self.assertRaisesRegex(BagMismatchError, "update bagofholding"):
                self.bag_class()(self.save_name)

        def test_other_bag_class_gets_generic_mismatch(self):
            self._save_with_stored_libver("v999", qualname="SomeOtherBag")
            with self.assertRaises(BagMismatchError) as ctx:
                self.bag_class()(self.save_name)
            self.assertNotIn("update bagofholding", str(ctx.exception))

        def test_stored_missing_code_rejected(self):
            self._save_with_stored_libver(None)
            with self.assertRaises(BagMismatchError):
                self.bag_class()(self.save_name)

        def test_none_validator_skips_libver_check(self):
            self._save_with_stored_libver("v220")
            loaded = self.bag_class()(self.save_name, bag_version_validator="none")
            self.assertEqual(42, loaded.load())

        def test_env_check_on_load(self):
            self.bag_class().save(42, self.save_name)
            with mock.patch.object(h5py.version, "hdf5_version_tuple", (1, 12, 0)):
                with self.assertRaises(EnvironmentMismatchError):
                    self.bag_class()(self.save_name)
                with self.assertRaises(EnvironmentMismatchError):
                    self.bag_class()(self.save_name, bag_version_validator="none")

        def test_unreadable_metadata_raises_clean_error(self):
            self.bag_class().save(42, self.save_name)
            for exc in (
                RuntimeError("bad version number for datatype message"),
                KeyError("nope"),
                OSError("nope"),
            ):
                with self.subTest(exc=type(exc).__name__):
                    with (
                        mock.patch.object(
                            self.bag_class(), "_unpack_bag_info", side_effect=exc
                        ),
                        self.assertRaisesRegex(
                            BagMismatchError, "newer bagofholding"
                        ) as ctx,
                    ):
                        self.bag_class()(self.save_name)
                    self.assertIsInstance(ctx.exception.__cause__, type(exc))

        def test_read_opens_pass_no_libver(self):
            self.bag_class().save(42, self.save_name)
            with mock.patch("h5py.File", wraps=h5py.File) as spy:
                self.assertEqual(42, self.bag_class()(self.save_name).load())
            read_calls = [
                c
                for c in spy.call_args_list
                if (c.args[1] if len(c.args) > 1 else c.kwargs.get("mode")) == "r"
            ]
            self.assertTrue(read_calls)
            for call in read_calls:
                self.assertNotIn("libver", call.kwargs)

    class TestLibverSave(unittest.TestCase, abc.ABC):
        save_name = "libver_savefile"

        @classmethod
        @abc.abstractmethod
        def bag_class(cls) -> type[bag.Bag]: ...

        def tearDown(self):
            for path in (self.save_name, f"{self.save_name}.h5"):
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)

        def _stored(self, path: str) -> str | None:
            return cast(H5Info, self.bag_class()(path).bag_info).libver_str

        def test_default_save_stores_max_libver(self):
            self.bag_class().save(42, self.save_name)
            self.assertEqual("v114", self._stored(self.save_name))

        def test_explicit_libver_stored_and_loads(self):
            self.bag_class().save(42, self.save_name, libver="v110")
            self.assertEqual("v110", self._stored(self.save_name))
            self.assertEqual(42, self.bag_class()(self.save_name).load())

        def test_invalid_libver_rejected_before_writing(self):
            for code in ("latest", "earliest", "v200", "v116"):
                with self.subTest(code=code):
                    with self.assertRaises(ValueError):
                        self.bag_class().save(42, self.save_name, libver=code)
                    self.assertFalse(os.path.exists(self.save_name))

        def test_unknown_option_is_type_error(self):
            with self.assertRaisesRegex(TypeError, "Unexpected save option"):
                self.bag_class().save(42, self.save_name, bogus=1)

        def test_env_check_on_save(self):
            with (
                mock.patch.object(h5py.version, "hdf5_version_tuple", (1, 12, 0)),
                self.assertRaises(EnvironmentMismatchError),
            ):
                self.bag_class().save(42, self.save_name, libver="v110")
            self.assertFalse(os.path.exists(self.save_name))

        def test_write_opens_use_chosen_libver(self):
            with mock.patch("h5py.File", wraps=h5py.File) as spy:
                self.bag_class().save(42, self.save_name, libver="v110")
            write_calls = [
                c
                for c in spy.call_args_list
                if (c.args[1] if len(c.args) > 1 else c.kwargs.get("mode")) != "r"
            ]
            self.assertTrue(write_calls)
            for call in write_calls:
                self.assertEqual(("v110", "v110"), call.kwargs.get("libver"))

        def test_subpath_save_keeps_own_libver(self):
            name = f"{self.save_name}.h5"
            self.bag_class().save(1, f"{name}/a", libver="v108")
            self.bag_class().save(2, f"{name}/b")
            self.assertEqual("v108", self._stored(f"{name}/a"))
            self.assertEqual("v114", self._stored(f"{name}/b"))
            self.assertEqual(1, self.bag_class()(f"{name}/a").load())
            self.assertEqual(2, self.bag_class()(f"{name}/b").load())


class TestH5BagLibverLoad(AbstractTestNamespace.TestLibverLoad):
    @classmethod
    def bag_class(cls):
        return bagofholding.h5.bag.H5Bag

    def test_h5bag_unpack_field_missing_vs_unreadable(self):
        self.bag_class().save(42, self.save_name)
        loaded = self.bag_class()(self.save_name)
        with loaded:
            self.assertIsNone(loaded._unpack_field("/", "no_such_attribute"))
            self.assertIsNone(loaded._unpack_field("/no_such_path", "qualname"))
            broken = mock.MagicMock()
            broken.attrs.__contains__.return_value = True
            broken.attrs.__getitem__.side_effect = KeyError("can't open")
            with (
                mock.patch.object(loaded, "_resolve", return_value=broken),
                self.assertRaises(KeyError),
            ):
                loaded._unpack_field("/", "qualname")


class TestTrieH5BagLibverLoad(AbstractTestNamespace.TestLibverLoad):
    @classmethod
    def bag_class(cls):
        return bagofholding.h5.triebag.TrieH5Bag

    def test_trieh5bag_unpack_field_missing_vs_unreadable(self):
        self.bag_class().save(42, self.save_name)
        loaded = self.bag_class()(self.save_name)
        with loaded:
            self.assertIsNone(loaded._unpack_field("/", "no_such_field"))
            self.assertIsNone(loaded._unpack_field("/no_such_path", "qualname"))
            with (
                mock.patch.object(
                    loaded,
                    "_read_trie",
                    return_value=(0, 0),
                ),
                mock.patch.object(
                    loaded,
                    "_read_pathlike",
                    side_effect=KeyError("Unable to synchronously open object"),
                ),
                self.assertRaises(KeyError),
            ):
                loaded._unpack_field("/", "qualname")


class TestH5BagLibverSave(AbstractTestNamespace.TestLibverSave):
    @classmethod
    def bag_class(cls):
        return bagofholding.h5.bag.H5Bag


class TestTrieH5BagLibverSave(AbstractTestNamespace.TestLibverSave):
    @classmethod
    def bag_class(cls):
        return bagofholding.h5.triebag.TrieH5Bag


if __name__ == "__main__":
    unittest.main()
