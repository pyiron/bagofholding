from typing import get_args
import unittest

import bagofholding.types as t


class TestBag(unittest.TestCase):
    def test_builtin_types_are_are_synchronized(self):
        for type_set, union_set, label in (
            (t.BuiltinItemType, t.BuiltinItemUnion, "items"),
            (t.BuiltinGroupType, t.BuiltinGroupUnion, "groups"),
        ):
            with self.subTest(label):
                self.assertEqual(
                    set(type_set.__constraints__),
                    set(get_args(union_set)),
                    msg="These are just different representations of the same thing. "
                    "Python and mypy don't currently play nicely enough to let us "
                    "construct one programmatically from the other, so guarantee here "
                    "that they still cover the same elements.",
                )
