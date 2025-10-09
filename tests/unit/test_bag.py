import contextlib
import io
import pickle
import sys
import unittest

from bagofholding.bag import Bag


class TestBag(unittest.TestCase):

    def test_pickle_helper(self):
        unpickleable = lambda x: x  # noqa E731

        with self.assertRaises((AttributeError, pickle.PicklingError)):
            Bag.pickle_check(unpickleable)

        msg = Bag.pickle_check(unpickleable, raise_exceptions=False)
        verb = "get" if sys.version_info >= (3, 12) else "pickle"
        self.assertEqual(
            f"Can't {verb} local object 'TestBag.test_pickle_helper.<locals>.<lambda>'",
            msg,
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Bag.pickle_check(unpickleable, raise_exceptions=False, print_message=True)
        self.assertIn(f"Can't {verb} local object", buf.getvalue())
