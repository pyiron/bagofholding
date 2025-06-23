import abc
import contextlib
import os
import pickle
import time
import unittest
from typing import ClassVar, Generic, TypeVar

from static.objects import Recursing

from bagofholding.bag import Bag
from bagofholding.h5.bag import H5Bag
from bagofholding.h5.triebag import TrieH5Bag


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
            strict=True,
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

                print(f"H5 with-context benchmark: depth={depth}, reps={n_reps}")
                self.assertLess(
                    dt_context,
                    dt_direct,
                    msg="Expected the with-context speed to be faster since the file "
                    "is not re-opened multiple times",
                )
                print("With context", dt_context, "<", dt_direct, "Direct acces")

                tolerable_overhead_ms = 100
                average_overhead_ms = 1000 * ((dt_direct - dt_context) / n_reps)
                self.assertLess(
                    average_overhead_ms,
                    tolerable_overhead_ms,
                    msg="Average file-opening overhead exceeded the (somewhat "
                    "arbitrary) threshold",
                )
                print("Average overhead", average_overhead_ms, "(ms)")

    def test_timing(self) -> None:
        fname = "benchmark_test"

        class Tester(abc.ABC):
            repeats: ClassVar[int] = 1

            @classmethod
            @abc.abstractmethod
            def _save(cls, obj: object, fname: str) -> None: ...

            @classmethod
            @abc.abstractmethod
            def _load(cls, fname: str) -> None: ...

            @classmethod
            def save(cls, obj: object, fname: str) -> int:
                for _ in range(cls.repeats):
                    cls._save(obj, fname)
                return cls.repeats

            @classmethod
            def load(cls, fname: str) -> int:
                for _ in range(cls.repeats):
                    cls._load(fname)
                return cls.repeats

        class WithPickle(Tester):
            repeats = 100

            @classmethod
            def _save(cls, obj, fname) -> None:
                with open(fname, "wb") as f:
                    pickle.dump(obj, f)

            @classmethod
            def _load(cls, fname) -> None:
                with open(fname, "rb") as f:
                    pickle.load(f)

        BagType = TypeVar("BagType", bound=Bag)

        class WithBag(Tester, Generic[BagType], abc.ABC):
            @classmethod
            @abc.abstractmethod
            def bag_class(cls) -> type[BagType]: ...

            @classmethod
            def _save(cls, obj: object, fname: str):
                cls.bag_class().save(obj, fname)

            @classmethod
            def _load(cls, fname: str):
                cls.bag_class()(fname).load()

        class WithH5Bag(WithBag[H5Bag]):
            @classmethod
            def bag_class(cls) -> type[H5Bag]:
                return H5Bag

        class WithTrieH5Bag(WithBag[TrieH5Bag]):
            @classmethod
            def bag_class(cls) -> type[TrieH5Bag]:
                return TrieH5Bag

        methods = [WithPickle, WithH5Bag, WithTrieH5Bag]
        method_names = [method.__name__ for method in methods]
        sizes = range(10, 200, 10)
        performance: dict[str, dict[str, list[float]]] = {
            k: {n: [] for n in method_names}
            for k in ["size (mb)", "save (ms)", "load (ms)"]
        }
        scales = {
            "size (mb)": 1.0 / 1024,
            "save (ms)": 1000,
            "load (ms)": 1000,
        }
        for n in sizes:
            obj = Recursing(n)
            for method in methods:

                t0 = time.time()
                scale = method.save(obj, fname)
                performance["save (ms)"][method.__name__].append(
                    (time.time() - t0) / scale
                )
                performance["size (mb)"][method.__name__].append(os.path.getsize(fname))
                t1 = time.time()
                scale = method.load(fname)
                performance["load (ms)"][method.__name__].append(
                    (time.time() - t1) / scale
                )

                with contextlib.suppress(FileNotFoundError):
                    os.remove(fname)

        for k, p in performance.items():
            print(k)
            print(f"size\t{'\t'.join(p.keys())}")
            for i, n in enumerate(sizes):
                print(
                    n,
                    "\t\t",
                    "\t\t".join(
                        [str(round(pp[i] * scales[k], 2)) for pp in p.values()]
                    ),
                )
