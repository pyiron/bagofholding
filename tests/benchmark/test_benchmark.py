import contextlib
import os
import pickle
import time
import unittest

from objects import Recursing

from bagofholding import H5Bag


class TestBenchmark(unittest.TestCase):
    """
    Presently, these can't actually fail -- they just print stuff for us to build
    intuition and start making comparisons
    """

    @classmethod
    def setUpClass(cls):
        cls.save_name = "savefile.h5"

    def tearDown(self):
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.save_name)

    def test_context_benefit(self):
        for depth, n_reps in zip(
                (2, 20, 200),
                (500, 50, 5),
        ):
            with self.subTest(f"depth={depth}, reps={n_reps}"):
                r = Recursing(depth)
                H5Bag.save(r, self.save_name)

                bag = H5Bag(self.save_name)

                t0 = time.time()
                with bag:
                    for _ in range(n_reps):
                        bag.load()
                dt_context = time.time() - t0

                t0 = time.time()
                for _ in range(n_reps):
                    bag.load()
                dt_direct = time.time() - t0

                self.assertLess(
                    dt_context,
                    dt_direct,
                    msg="Expected the with-context speed to be faster since the file "
                        "is not re-opened multiple times"
                )

                tolerable_overhead_ms = 10
                average_overhead_ms = 1000 * ((dt_direct - dt_context) / n_reps)
                self.assertLess(
                    average_overhead_ms,
                    tolerable_overhead_ms,
                    msg="Average file-opening overhead exceeded the (somewhat "
                        "arbitrary) threshold"
                )

    def test_timing(self):
        fname = "wf.pckl"
        n_pickle_repeats = 10

        for n in [3, 30]:
            wf = Recursing(n)
            with self.subTest(f"------ {len(wf)}-sized graph ------"):
                t0 = time.time()
                for _ in range(n_pickle_repeats):
                    with open(fname, "wb") as f:
                        pickle.dump(wf, f)
                pt_save = (time.time() - t0) / n_pickle_repeats

                psize = os.path.getsize(fname)

                t1 = time.time()
                for _ in range(n_pickle_repeats):
                    with open(fname, "rb") as f:
                        pickle.load(f)
                pt_load = (time.time() - t1) / n_pickle_repeats

                with contextlib.suppress(FileNotFoundError):
                    os.remove(fname)

                fname = "wf.h5"
                for bag_class in [H5Bag]:
                    t0 = time.time()
                    bag_class.save(wf, fname)
                    h5t_save = time.time() - t0

                    h5size = os.path.getsize(fname)

                    t1 = time.time()
                    b = bag_class(fname)
                    t2 = time.time()
                    paths = b.list_paths()
                    t3 = time.time()
                    metadata = b["object/state/child/"]
                    t4 = time.time()
                    partial = b.load("object/state/child/")
                    t5 = time.time()
                    b.load()
                    h5t_load = time.time() - t5
                    print(paths[:10])
                    print(metadata)
                    print(partial.label)
                    print(
                        "Instantiate, list, item access, partial",
                        t2 - t1,
                        t3 - t2,
                        t4 - t3,
                        t5 - t4,
                    )

                    with contextlib.suppress(FileNotFoundError):
                        os.remove(fname)

                    print("Pickle baseline size, save, load (mb, ms, ms)")
                    print(
                        round(psize / 1024, 2),
                        round(pt_save * 1000, 2),
                        round(pt_load * 1000, 2),
                    )
                    print(f"{bag_class} size, save, load (mb, ms, ms)")
                    print(
                        round(h5size / 1024, 2),
                        round(h5t_save * 1000, 2),
                        round(h5t_load * 1000, 2),
                    )
                    print("Ratios: size, save, load")
                    print(
                        round(h5size / psize, 2),
                        round(h5t_save / pt_save, 2),
                        round(h5t_load / pt_load, 2),
                    )
                    print("--------------------------")
