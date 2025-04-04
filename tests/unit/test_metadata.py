import sys
import unittest

import bidict
import numpy

from bagofholding.metadata import _get_module, get_version


def some_version_scraper(module_name: str) -> str | None:
    return "some.version"


class TestMetadata(unittest.TestCase):
    def test_get_module(self):
        self.assertEqual("builtins", _get_module(int), msg="Should work with types")
        self.assertEqual("builtins", _get_module(5), msg="Should work with instances")

    def test_version_scraping(self):

        py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        self.assertEqual(
            py_version,
            get_version(_get_module(int), {}),
            msg="builtins should return the python version",
        )

        self.assertEqual(
            py_version,
            get_version(_get_module(5), {"builtins": some_version_scraper}),
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
            numpy.__version__,
            get_version("numpy.fft", {}),
            msg="The version of the root module should be accessed",
        )

        self.assertEqual(
            some_version_scraper("foo"),
            get_version("numpy.fft", {"numpy": some_version_scraper}),
            msg="The root module should be accessed to search the scaper map",
        )
