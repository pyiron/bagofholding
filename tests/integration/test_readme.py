import contextlib
import doctest
import os
import unittest


def remove_save_file(_):
    for filename in ["file.h5", "something.h5", "custom.h5"]:
        with contextlib.suppress(FileNotFoundError):
            os.remove(filename)


def load_tests(loader, tests, ignore):
    tests.addTests(
        doctest.DocFileSuite(
            "../../docs/README.md",
            optionflags=doctest.ELLIPSIS,
            tearDown=remove_save_file,
            globs={"__name__": "__main__"},
        )
    )

    return tests


class TestTriggerFromIDE(unittest.TestCase):
    """
    Just so we can instruct it to run unit tests here with a gui run command on the file
    """

    def test_void(self):
        pass


if __name__ == "__main__":
    unittest.main()
