import contextlib
import os
import pickle
import unittest

import numpy as np
from pyiron_snippets.dotdict import DotDict
from static.objects import CustomReduce, Parent, SomeData, build_workflow

from bagofholding import H5Bag


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.objects = (
            5,
            SomeData(),
            CustomReduce(10, ["iterable_item_1", "iterable_item_2"]),
            Parent(),
            Parent,
            all,
            complex(2, 3),
            np.linspace(1, 10, 10),
            np.random.random(size=(2, 2, 2)),
            DotDict({"a": 1, "b": 2, "c": 3}),
            np.array([Parent("p"), Parent("q")]),
            np.array([SomeData(), SomeData()]),
        )
        self.wf = build_workflow(3)
        self.bags = [H5Bag]
        self.fname = "obj.bag"

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.fname)

    def test_pickle(self):
        """
        Sanity check -- unpickleable objects are _allowed_ to fail bagging.
        """
        for obj in self.objects:
            with self.subTest(name=str(obj)):
                pickle.loads(pickle.dumps(obj))

    def test_save_load_cycle(self):
        for bag_class in self.bags:
            for obj in self.objects:
                with self.subTest(name=f"{bag_class.__name__} on {type(obj).__name__}"):
                    bag_class.save(obj, filepath=self.fname)
                    obj2 = bag_class(filepath=self.fname).load()
                    self.assertIs(obj2.__class__, obj.__class__)
                    if isinstance(obj, np.ndarray):
                        self.assertTrue(
                            np.all(obj == obj2), msg=f"Expected {obj} but got {obj2}"
                        )
                    else:
                        self.assertEqual(
                            obj2, obj, msg=f"Expected {obj} but got {obj2}"
                        )

    def test_workflow(self):
        for bag_class in self.bags:
            with self.subTest(name=f"{bag_class.__name__}"):
                bag_class.save(self.wf, filepath=self.fname)
                wf2 = bag_class(filepath=self.fname).load()
                self.assertDictEqual(
                    wf2.outputs.to_value_dict(), self.wf.outputs.to_value_dict()
                )
                self.assertTupleEqual(wf2.child_labels, self.wf.child_labels)

    def test_subaccess(self):
        for bag_class in self.bags:
            with self.subTest(name=str(bag_class)):
                obj = Parent()
                bag_class.save(obj, self.fname)
                obj2 = bag_class(self.fname).load(
                    "object/state/child/state/modified_data"
                )
                # The path is _extremely_ ugly, but it does the trick
                self.assertEqual(obj.child.modified_data, obj2)
                # Of course something like "object/state/child/state/modified_data" would be better
                # This just requires a "StringKeyDictionary(Group)"
                # Or even "object/child/modified_data"
                # But that would require being sneaky and implicit about the "state" access,
                # Which may not be an option since the state is not mandatorily a dictionary

    def test_list_paths(self):
        for bag_class in self.bags:
            with self.subTest(name=str(bag_class)):
                bag_class.save(Parent(), self.fname)
                self.assertSetEqual(
                    set(bag_class(self.fname).list_paths()),
                    {
                        "object",
                        "object/constructor",
                        "object/args",
                        "object/args/i0",
                        "object/args/i1",
                        "object/args/i2",
                        "object/state",
                        "object/state/name",
                        "object/state/data",
                        "object/state/data/i0",
                        "object/state/data/i1",
                        "object/state/data/i2",
                        "object/state/child",
                        "object/state/child/constructor",
                        "object/state/child/args",
                        "object/state/child/args/i0",
                        "object/state/child/args/i1",
                        "object/state/child/args/i2",
                        "object/state/child/state",
                        "object/state/child/state/name",
                        "object/state/child/state/parent",
                        "object/state/child/state/data",
                        "object/state/child/state/modified_data",
                    },
                    msg=f"Got instead {bag_class(self.fname).list_paths()}",
                )
