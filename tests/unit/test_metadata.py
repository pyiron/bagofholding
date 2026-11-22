import sys
import unittest

import numpy as np
import pyiron_snippets
from packaging import version as packaging_version

from bagofholding import EnvironmentMismatchError
from bagofholding.metadata import (
    Metadata,
    get_module,
    get_qualname,
    get_version,
    validate_version,
    versions_match,
)


def some_version_scraper(module_name: str) -> str | None:
    return "some_nonsemantic.version"


def _modify_numpy_version(index: int) -> str:
    v = packaging_version.Version(np.__version__)
    release = (v.major, v.minor, v.micro)
    return ".".join(
        "9999999999" if i == index else str(x) for i, x in enumerate(release)
    )


def numpy_unmodified(_: str) -> str:
    return str(np.__version__)


def numpy_modify_patch(_: str) -> str:
    return _modify_numpy_version(2)


def numpy_modify_minor(_: str) -> str:
    return _modify_numpy_version(1)


def numpy_modify_major(_: str) -> str:
    return _modify_numpy_version(0)


class TestMetadata(unittest.TestCase):
    def test_get_module(self):
        self.assertEqual("builtins", get_module(int), msg="Should work with types")
        self.assertEqual("builtins", get_module(5), msg="Should work with instances")

    def test_get_qualname(self):
        self.assertEqual("int", get_qualname(int), msg="Should work with types")
        self.assertEqual("int", get_qualname(5), msg="Should work with instances")

    def test_version_scraping(self):

        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.assertEqual(
            py_version,
            get_version(get_module(int), {}),
            msg="builtins should return the python version",
        )

        self.assertEqual(
            py_version,
            get_version(get_module(5), {"builtins": some_version_scraper}),
            msg="builtins should _always_ just return the python version",
        )

        self.assertIsNone(
            get_version("static", {}),
            msg="Modules without a version are expected to return None",
        )

        self.assertEqual(
            pyiron_snippets.__version__,
            get_version("pyiron_snippets", {}),
            msg="This is the fundamental behaviour of the default",
        )

        self.assertEqual(
            some_version_scraper("foo"),
            get_version("pyiron_snippets", {"pyiron_snippets": some_version_scraper}),
            msg="Users can override how versions are scraped for a particular module",
        )

        self.assertEqual(
            pyiron_snippets.__version__,
            get_version(
                "pyiron_snippets", {"not_pyiron_snippets": some_version_scraper}
            ),
            msg="Modules shouldn't care about other modules' overrides",
        )

        self.assertEqual(
            np.__version__,
            get_version("numpy.fft", {}),
            msg="The version of the root module should be accessed",
        )

        self.assertEqual(
            some_version_scraper("foo"),
            get_version("numpy.fft", {"numpy": some_version_scraper}),
            msg="The root module should be accessed to search the scaper map",
        )

    def test_validate_version(self):

        self.assertIsNone(
            validate_version(Metadata("SomeContentType")),
            msg="Empty metadata can't be invalid",
        )

        numpy_metadata = Metadata(
            "DummyContentType",
            module=np.__name__,
            version=str(np.__version__),
        )

        self.assertIsNone(validate_version(numpy_metadata))
        self.assertIsNone(validate_version(numpy_metadata, validator="exact"))
        self.assertIsNone(
            validate_version(
                numpy_metadata,
                validator="semantic-minor",
                version_scraping={np.__name__: numpy_modify_patch},
            )
        )
        self.assertIsNone(
            validate_version(
                numpy_metadata,
                validator="semantic-major",
                version_scraping={np.__name__: numpy_modify_minor},
            )
        )
        self.assertIsNone(
            validate_version(
                numpy_metadata,
                validator="none",
                version_scraping={np.__name__: numpy_modify_major},
            )
        )
        self.assertIsNone(
            validate_version(
                numpy_metadata, version_scraping={np.__name__: numpy_unmodified}
            )
        )

        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                numpy_metadata,
                version_scraping={np.__name__: numpy_modify_patch},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                numpy_metadata,
                validator="exact",
                version_scraping={np.__name__: numpy_modify_patch},
            )
        self.assertIsNone(
            validate_version(
                numpy_metadata,
                validator="semantic-patch",
                version_scraping={np.__name__: numpy_unmodified},
            )
        )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                numpy_metadata,
                validator="semantic-patch",
                version_scraping={np.__name__: numpy_modify_patch},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                numpy_metadata,
                validator="semantic-minor",
                version_scraping={np.__name__: numpy_modify_minor},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                numpy_metadata,
                validator="semantic-major",
                version_scraping={np.__name__: numpy_modify_major},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                numpy_metadata,
                validator="semantic-major",
                version_scraping={np.__name__: some_version_scraper},
            )

        non_semantic_metadata = Metadata(
            "DummyContentType",
            module=np.__name__,
            version=some_version_scraper(""),  # Force-override the version
        )
        self.assertIsNone(
            validate_version(
                non_semantic_metadata,
                version_scraping={np.__name__: some_version_scraper},
            ),
        )
        self.assertIsNone(
            validate_version(non_semantic_metadata, validator="none"),
        )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                non_semantic_metadata,
                version_scraping={np.__name__: numpy_modify_patch},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                non_semantic_metadata,
                validator="exact",
                version_scraping={np.__name__: numpy_modify_patch},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                non_semantic_metadata,
                validator="semantic-minor",
                version_scraping={np.__name__: numpy_modify_minor},
            )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(
                non_semantic_metadata,
                validator="semantic-major",
                version_scraping={np.__name__: numpy_modify_major},
            )

        self.assertIsNone(
            validate_version(non_semantic_metadata, validator=lambda _a, _b: True),
        )
        with self.assertRaises(EnvironmentMismatchError):
            validate_version(non_semantic_metadata, validator=lambda _a, _b: False)

        with self.assertRaises(ValueError):
            validate_version(non_semantic_metadata, validator="not-a-valid-keyword")


class TestVersionsMatch(unittest.TestCase):
    def test_exact(self):
        self.assertTrue(versions_match("0.1.3", "0.1.3", "exact"))
        self.assertFalse(
            versions_match("0.1.3.dev1", "0.1.3", "exact"),
            msg="exact must stay literal and not strip dev segments",
        )
        self.assertFalse(versions_match("1.2", "1.2.0", "exact"))

    def test_semantic_patch(self):
        self.assertTrue(versions_match("0.1.3.dev1", "0.1.3", "semantic-patch"))
        self.assertTrue(versions_match("0.1.3+local", "0.1.3rc1", "semantic-patch"))
        self.assertFalse(versions_match("0.1.4", "0.1.3", "semantic-patch"))

    def test_semantic_minor(self):
        self.assertTrue(
            versions_match("0.1.11.dev2+gd4bafdeb1", "0.1.0", "semantic-minor")
        )
        self.assertTrue(versions_match("2.3.0rc1", "2.3.1", "semantic-minor"))
        self.assertTrue(versions_match("1.2", "1.2.0", "semantic-minor"))
        self.assertFalse(versions_match("0.2.0", "0.1.0", "semantic-minor"))

    def test_semantic_major(self):
        self.assertTrue(versions_match("1.9.0", "1.0.0", "semantic-major"))
        self.assertTrue(versions_match("1", "1.5.2", "semantic-major"))
        self.assertFalse(versions_match("2.0.0", "1.0.0", "semantic-major"))

    def test_invalid_versions_fall_back_to_exact(self):
        for validator in ("semantic-patch", "semantic-minor", "semantic-major"):
            with self.subTest(validator):
                self.assertTrue(
                    versions_match("not.a.version", "not.a.version", validator)
                )
                self.assertFalse(versions_match("not.a.version", "1.0.0", validator))
                self.assertFalse(versions_match("1.0.0", "not.a.version", validator))

    def test_none(self):
        self.assertTrue(versions_match("1.0.0", "garbage", "none"))

    def test_callable_receives_current_then_stored(self):
        received: list[tuple[str, str]] = []

        def record(current: str, stored: str) -> bool:
            received.append((current, stored))
            return False

        self.assertFalse(versions_match("current", "stored", record))
        self.assertEqual([("current", "stored")], received)

    def test_unknown_keyword_raises(self):
        with self.assertRaises(ValueError):
            versions_match("1.0.0", "1.0.0", "not-a-valid-keyword")
