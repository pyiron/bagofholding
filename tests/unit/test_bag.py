import contextlib
import io
import pickle
import string
import sys
import unittest

from bagofholding.bag import Bag


class TestBag(unittest.TestCase):

    def test_pickle_helper(self):
        unpickleable = lambda x: x  # noqa E731

        with self.assertRaises((AttributeError, pickle.PicklingError)):
            Bag.pickle_check(unpickleable)

        msg = Bag.pickle_check(unpickleable, raise_exceptions=False)
        msg_reference = string.Template("Can't $verb local object $obj")

        verb = (
            "get"
            if sys.version_info >= (3, 12) and sys.version_info < (3, 14)
            else "pickle"
        )
        obj_reference = "TestBag.test_pickle_helper.<locals>.<lambda>"
        obj = (
            f"'{obj_reference}'"
            if sys.version_info < (3, 14)
            else f"<function {obj_reference} at {id(unpickleable):#x}"
        )

        self.assertEqual(msg_reference.substitute(verb=verb, obj=obj), msg)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            Bag.pickle_check(unpickleable, raise_exceptions=False, print_message=True)
        self.assertIn(f"Can't {verb} local object", buf.getvalue())
