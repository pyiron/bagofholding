import sys
import unittest

import bidict
import numpy as np

from bagofholding.metadata import (
    EnvironmentMismatchError,
    Metadata,
    _decompose_semver,
    get_module,
    get_qualname,
    get_version,
    validate_version,
)


def some_version_scraper(module_name: str) -> str | None:
    return "some_nonsemantic.version"


def _modify_numpy_version(index: int) -> str:
    semver = _decompose_semver(np.__version__)
    if semver is None:
        raise ValueError("Expected semantic version for numpy.")
    return ".".join(
        "9999999999" if i == index else str(x) for i, x in enumerate(semver)
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
            bidict.__version__,
            get_version("bidict", {}),
            msg="This is the fundamental behaviour of the default",
        )

        self.assertEqual(
            some_version_scraper("foo"),
            get_version("bidict", {"bidict": some_version_scraper}),
            msg="Users can override how versions are scraped for a particular module",
        )

        self.assertEqual(
            bidict.__version__,
            get_version("bidict", {"not_bidict": some_version_scraper}),
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
            validate_version(Metadata()),
            msg="Empty metadata can't be invalid",
        )

        numpy_metadata = Metadata(
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
