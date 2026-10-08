import contextlib
import doctest
import os
import sys
import types
import unittest


def use_globs_as_main(test):
    """
    Register the doctest namespace as the actual `__main__` module, so objects defined
    in the README are importable just as they would be in a notebook or interpreter.
    """
    main = types.ModuleType("__main__")
    main.__dict__.update(test.globs)
    test.globs = main.__dict__
    test.original_main = sys.modules["__main__"]
    sys.modules["__main__"] = main


def tear_down(test):
    sys.modules["__main__"] = test.original_main
    remove_save_file(test)


def remove_save_file(_):
    for filename in ["file.h5", "something.h5", "custom.h5", "will_fail.h5"]:
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(os.getcwd(), filename))


def load_tests(loader, tests, ignore):
    tests.addTests(
        doctest.DocFileSuite(
            "../../docs/README.md",
            optionflags=doctest.ELLIPSIS,
            setUp=use_globs_as_main,
            tearDown=tear_down,
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
