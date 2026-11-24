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

        def _save_with_stored_libver(self, code: str | None) -> None:
            base = self.bag_class()
            info = dataclasses.replace(
                cast(H5Info, base.get_bag_info()), libver_str=code
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


if __name__ == "__main__":
    unittest.main()
