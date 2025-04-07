import contextlib
import io
import unittest

from bagofholding.bag import Bag


class TestBag(unittest.TestCase):

    def test_pickle_helper(self):
        unpickleable = lambda x: x  # noqa E731

        with self.assertRaises(AttributeError):
            Bag.pickle_check(unpickleable)

        msg = Bag.pickle_check(unpickleable, raise_exceptions=False)
        self.assertEqual(
            "Can't get local object 'TestBag.test_pickle_helper.<locals>.<lambda>'", msg
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Bag.pickle_check(unpickleable, raise_exceptions=False, print_message=True)
        self.assertIn("Can't get local object", buf.getvalue())
