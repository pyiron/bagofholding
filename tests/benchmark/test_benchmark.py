import abc
import contextlib
import os
import pickle
import time
import unittest
from collections.abc import Callable
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy import optimize
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

                fudge_factor = 1.05
                # On the remote CI, the context does not always give a benefit but
                # can actually be _slower_. This is stochastic, and I assume it relates
                # to the load on the remote machine, which is outside our control
                # Instead of testing that it is faster, let's test that it is at least
                # not much slower.
                dt_reference = fudge_factor * dt_direct

                print(f"H5 with-context benchmark: depth={depth}, reps={n_reps}")
                self.assertLess(
                    dt_context,
                    dt_reference,
                    msg="Expected the with-context speed to be faster since the file "
                    "is not re-opened multiple times...or at least much not slower",
                )
                print(
                    f"With context {dt_context} < {dt_reference} = {fudge_factor} * "
                    f"{dt_direct} Direct access with a fudge factor"
                )

                tolerable_overhead_ms = 150
                average_overhead_ms = 1000 * ((dt_direct - dt_context) / n_reps)
                self.assertLess(
                    average_overhead_ms,
                    tolerable_overhead_ms,
                    msg="Average file-opening overhead exceeded the (somewhat "
                    "arbitrary) threshold",
                )
                print("Average overhead", average_overhead_ms, "(ms)")

    def test_timing(self) -> None:
        """
        This is just a naive brute-force test to ensure that the scaling behaviour of
        the established storage routines is doing what we expect.

        Determining the scaling for :class:`bagofholding.h5.bag.TrieH5Bag` is a little
        tricky, because as established in the trie scaling test, de/reconstruction of
        the trie should move from a best-case scenario of nearly O(N) for breadth-like
        tries, to a worst-case scenario of O(N^2) scaling of depth-like tries, and our
        test object is not _perfectly_ depth-like.
        """
        fname = "benchmark_test"

        class Tester(abc.ABC):
            repeats: ClassVar[int] = 2

            @classmethod
            @abc.abstractmethod
            def _save(cls, obj: object, fname: str) -> None: ...

            @classmethod
            @abc.abstractmethod
            def _load(cls, fname: str) -> None: ...

            @classmethod
            def save(cls, obj: object, fname: str) -> int:
                cls._save(obj, fname)
                return cls.repeats

            @classmethod
            def load(cls, fname: str) -> int:
                cls._load(fname)
                return cls.repeats

        class WithPickle(Tester):
            repeats = 2000

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
        sizes = np.arange(start=10, stop=221, step=10)
        performance: dict[str, dict[str, list[float]]] = {
            "size (mb)": {},
            "save (ms)": {},
            "load (ms)": {},
        }
        scales = {
            "size (mb)": 1.0 / 1024,
            "save (ms)": 1000,
            "load (ms)": 1000,
        }
        for method in methods:
            performance["size (mb)"][method.__name__] = []
            performance["save (ms)"][method.__name__] = []
            performance["load (ms)"][method.__name__] = []
        for n in sizes:
            obj = Recursing(n)
            for method in methods:
                performance["save (ms)"][method.__name__].append(0)
                performance["size (mb)"][method.__name__].append(0)
                performance["load (ms)"][method.__name__].append(0)

                for _ in range(method.repeats):
                    t0 = time.time()
                    scale = method.save(obj, fname)
                    performance["save (ms)"][method.__name__][-1] += (time.time() - t0) / scale
                    performance["size (mb)"][method.__name__][-1] = os.path.getsize(fname)
                    t1 = time.time()
                    scale = method.load(fname)
                    performance["load (ms)"][method.__name__][-1] += (time.time() - t1) / scale
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(fname)

        print("Raw scaling data")
        for k, p in performance.items():
            print(k)
            sep = "\t"  # python <3.12 compatibility -- no escaping inside f-strings
            print(f"size\t{sep.join(p.keys())}")
            for i, n in enumerate(sizes):
                print(
                    n,
                    "\t\t",
                    "\t\t".join(
                        [str(round(pp[i] * scales[k], 2)) for pp in p.values()]
                    ),
                )

        def linear(x, a, b):
            return a * x + b

        def quadratic(x, a, b, c):
            return a * x**2 + b * x + c

        def cubic(x, a, b, c, d):
            return a * x**3 + b * x**2 + c * x + d

        models: dict[str, Callable[..., NDArray[np.float64]]] = {
            "linear": linear,
            "quadratic": quadratic,
            "cubic": cubic,
        }

        fit_results: dict[str, dict[str, tuple[str, list[float]]]] = {}

        residual_improvement_to_accept_new_model = 0.2
        # Demand a 5x improvement in residuals to warrant a more complex model
        for k, data in performance.items():
            fit_results[k] = {}
            for name, raw_y in data.items():
                y = np.array(raw_y) * scales[k]
                best_fit = ("not a model", [0.0])
                best_res: np.floating[Any] | float = np.inf
                for model_name, model_func in models.items():
                    popt, _ = optimize.curve_fit(model_func, sizes, y, maxfev=10000)
                    residual = np.mean((y - model_func(sizes, *popt)) ** 2)
                    if residual < residual_improvement_to_accept_new_model * best_res:
                        best_res = residual
                        best_fit = (model_name, popt.tolist())
                fit_results[k][name] = best_fit

        # Check expected models and leading coefficients
        expected = {
            "size (mb)": {
                "WithPickle": ("quadratic", 4.0e-3),
                "WithH5Bag": ("quadratic", 4.3e-3),
                "WithTrieH5Bag": ("quadratic", 3.7e-3),
            },
            "save (ms)": {
                "WithPickle": ("linear", 2.0e-3),
                "WithH5Bag": ("quadratic", 6.9e-2),
                "WithTrieH5Bag": ("cubic", 5e-4),
            },
            "load (ms)": {
                "WithPickle": ("linear", 1.0e-3),
                "WithH5Bag": ("quadratic", 2.8e-2),
                "WithTrieH5Bag": ("cubic", 2.5e-4),
            },
        }

        max_leading_parameter_relative_error = 1./3.
        for metric, tools in expected.items():
            for tool, (expected_model, expected_param) in tools.items():
                with self.subTest(f"{metric} {tool}"):
                    actual_model, actual_params = fit_results[metric][tool]
                    self.assertEqual(
                        actual_model,
                        expected_model,
                        msg=f"Previous data has indicated that {tool} should scale "
                        f"{expected_model} with respect to {metric}, but got "
                        f"{actual_model}.",
                    )
                    with self.subTest(
                        f"{metric} {tool} {expected_model} leading paremeter"
                    ):
                        rel_err = abs(actual_params[0] - expected_param) / abs(
                            expected_param
                        )
                        self.assertLess(
                            rel_err,
                            max_leading_parameter_relative_error,
                            msg=f"Expected parameter {expected_param} got {actual_params[0]} -- relative error {rel_err}",
                        )

        print("Fit results:")
        for metric_name, tools_dict in fit_results.items():
            print(metric_name)
            for tool_name, (model_name, coefficients) in tools_dict.items():
                print(
                    f"  {tool_name}: {model_name}, params = {coefficients};"
                    f"expected {expected[metric_name][tool_name]}"
                )
