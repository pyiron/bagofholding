import contextlib
import os
import pickle
import time
import unittest

from static.objects import build_workflow

from bagofholding import H5Bag


class TestBenchmark(unittest.TestCase):
    """
    Presently, these can't actually fail -- they just print stuff for us to build
    intuition and start making comparisons
    """

    def test_timing(self):
        fname = "wf.pckl"

        for n in [3, 30]:
            wf = build_workflow(n)
            with self.subTest(f"------ {len(wf)}-sized graph ------"):
                t0 = time.time()
                with open(fname, "wb") as f:
                    pickle.dump(wf, f)
                pt_save = time.time() - t0

                psize = os.path.getsize(fname)

                t1 = time.time()
                with open(fname, "rb") as f:
                    pickle.load(f)
                pt_load = time.time() - t1

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
                    metadata = b["object/state/diff/"]
                    t4 = time.time()
                    partial = b.load("object/state/diff/")
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
