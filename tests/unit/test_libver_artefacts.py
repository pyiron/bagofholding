import unittest

import numpy as np
import static.libver
from static.libver import ARTEFACT_DIR, BAG_CLASSES, FUTURE_DIR

from bagofholding.exceptions import BagMismatchError
from bagofholding.h5 import format_versions


class TestLibverArtefacts(unittest.TestCase):
    def assert_same_value_and_type(self, got, expected):
        self.assertIs(type(got), type(expected))
        if isinstance(expected, dict):
            self.assertEqual(got.keys(), expected.keys())
            for key, value in expected.items():
                self.assert_same_value_and_type(got[key], value)
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(got), len(expected))
            for got_item, expected_item in zip(got, expected, strict=True):
                self.assert_same_value_and_type(got_item, expected_item)
        else:
            self.assertEqual(got, expected)

    def test_one_artefact_per_allowed_code(self):
        for name, cls in BAG_CLASSES.items():
            found = {p.stem.split(".")[0] for p in ARTEFACT_DIR.glob(f"*.{name}.h5")}
            self.assertEqual(
                found, set(format_versions.allowed_libvers(cls.max_libver)), name
            )

    def test_artefacts_load(self):
        expected = static.libver.build()
        paths = sorted(ARTEFACT_DIR.glob("*.h5"))
        self.assertTrue(paths)
        for path in paths:
            code, name = path.stem.split(".")
            with self.subTest(path.name):
                bag = BAG_CLASSES[name](path, bag_version_validator="none")
                self.assertEqual(bag.bag_info.libver_str, code)
                got = bag.load(version_validator="none")
                self.assertEqual(got.keys(), expected.keys())
                for key, value in expected.items():
                    with self.subTest(key):
                        if isinstance(value, np.ndarray):
                            self.assertEqual(got[key].dtype, value.dtype)
                            np.testing.assert_array_equal(got[key], value)
                        else:
                            self.assert_same_value_and_type(got[key], value)

    def test_one_future_artefact_per_class(self):
        self.assertEqual(
            {p.name for p in FUTURE_DIR.glob("*.h5")},
            {f"future.{name}.h5" for name in BAG_CLASSES},
        )

    def test_future_artefact_fails_cleanly(self):
        # Metadata is read and validated before any (possibly undecodable) data.
        for name, cls in BAG_CLASSES.items():
            with (
                self.subTest(name),
                self.assertRaisesRegex(BagMismatchError, "'v999'.*update bagofholding"),
            ):
                cls(FUTURE_DIR / f"future.{name}.h5")
