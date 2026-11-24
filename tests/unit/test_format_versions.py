import unittest
from unittest import mock

from bagofholding.exceptions import BagMismatchError, EnvironmentMismatchError
from bagofholding.h5 import format_versions


class TestFormatVersions(unittest.TestCase):
    def test_allowed_libvers(self):
        self.assertEqual(
            format_versions.allowed_libvers("v114"), ("v108", "v110", "v112", "v114")
        )
        self.assertEqual(format_versions.allowed_libvers("v200")[-1], "v200")

    def test_resolve_save_libver(self):
        self.assertEqual(format_versions.resolve_save_libver(None, "v114"), "v114")
        self.assertEqual(format_versions.resolve_save_libver("v110", "v114"), "v110")
        for bad in ("latest", "earliest"):
            with self.assertRaisesRegex(ValueError, f"'{bad}'.*pin"):
                format_versions.resolve_save_libver(bad, "v114")
        for bad in ("v200", "v116", "114", ("v108", "v114")):
            with self.assertRaisesRegex(ValueError, "v108, v110, v112, v114"):
                format_versions.resolve_save_libver(bad, "v114")

    def test_check_stored_libver(self):
        for ok in ("latest", "v108", "v114"):
            format_versions.check_stored_libver(ok, "v114")  # no raise
        for bad in ("v200", "v220", None):
            with self.assertRaisesRegex(BagMismatchError, "update bagofholding"):
                format_versions.check_stored_libver(bad, "v114")

    def test_check_hdf5_supports(self):
        with mock.patch("h5py.version.hdf5_version_tuple", (1, 14, 6)):
            format_versions.check_hdf5_supports("v114")  # no raise
            with self.assertRaisesRegex(
                EnvironmentMismatchError, r"HDF5 >= 2\.0.*1\.14\.6.*update h5py"
            ):
                format_versions.check_hdf5_supports("v200")


class TestInvalidMaxLibver(unittest.TestCase):
    def test_allowed_libvers_rejects_unknown_code(self):
        with self.assertRaisesRegex(ValueError, "Invalid max_libver 'latest'.*v200"):
            format_versions.allowed_libvers("latest")

    def test_check_hdf5_supports_rejects_unknown_code(self):
        with self.assertRaisesRegex(ValueError, "Invalid max_libver 'v999'.*v200"):
            format_versions.check_hdf5_supports("v999")
