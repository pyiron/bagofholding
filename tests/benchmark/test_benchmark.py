import abc
import contextlib
import os
import pickle
import time
import unittest
from typing import ClassVar, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
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
        Using bayesian information criterion to assess model scaling, and then compare
        a noise-adjusted measure of the expected leading fit parameter against by-hand
        measure of that parameter from past runs.

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

        sizes = np.arange(start=10, stop=221, step=10)
        metrics = ["size (mb)", "save (ms)", "load (ms)"]
        tools = [WithPickle, WithH5Bag, WithTrieH5Bag]
        scales = {
            "size (mb)": 1.0 / 1024,
            "save (ms)": 1000,
            "load (ms)": 1000,
        }
        performance: dict[str, dict[str, list[float]]] = {
            metric: {tool.__name__: [] for tool in tools} for metric in metrics
        }
        for n in sizes:
            obj = Recursing(n)
            for tool in tools:
                performance["save (ms)"][tool.__name__].append(0)
                performance["size (mb)"][tool.__name__].append(0)
                performance["load (ms)"][tool.__name__].append(0)

                for _ in range(tool.repeats):
                    t0 = time.time()
                    scale = tool.save(obj, fname)
                    performance["save (ms)"][tool.__name__][-1] += (
                        time.time() - t0
                    ) / scale
                    performance["size (mb)"][tool.__name__][-1] = os.path.getsize(fname)
                    t1 = time.time()
                    scale = tool.load(fname)
                    performance["load (ms)"][tool.__name__][-1] += (
                        time.time() - t1
                    ) / scale
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(fname)

        print("Raw scaling data")
        for metric, p in performance.items():
            print(metric)
            sep = "\t"  # python <3.12 compatibility -- no escaping inside f-strings
            print(f"size\t{sep.join(p.keys())}")
            for i, n in enumerate(sizes):
                print(
                    n,
                    "\t\t",
                    "\t\t".join(
                        [str(round(pp[i] * scales[metric], 2)) for pp in p.values()]
                    ),
                )

        # Check expected models and leading coefficients
        expected = {
            "size (mb)": {
                "WithPickle": ("quadratic", 4.00e-3),
                "WithH5Bag": ("quadratic", 4.32e-3),
                "WithTrieH5Bag": ("quadratic", 3.70e-3),
            },
            "save (ms)": {
                "WithPickle": ("linear", 1.97e-3),
                "WithH5Bag": ("quadratic", 7.61e-2),
                "WithTrieH5Bag": ("cubic", 5.05e-4),
            },
            "load (ms)": {
                "WithPickle": ("linear", 1.01e-3),
                "WithH5Bag": ("quadratic", 2.77e-2),
                "WithTrieH5Bag": ("cubic", 2.30e-4),
            },
        }
        # Data from earlier human-supervised runs

        fit_results: dict[str, dict[str, list[float]]] = {
            metric: {} for metric in metrics
        }
        best_models: dict[str, dict[str, str]] = {metric: {} for metric in metrics}
        z_scores: dict[str, dict[str, float]] = {metric: {} for metric in metrics}
        name_map = ["scalar", "linear", "quadratic", "cubic", "quartic"]

        for metric, data in performance.items():
            for tool_name, raw_y in data.items():
                y = np.array(raw_y) * scales[metric]
                best_score = None
                for degree in (1, 2, 3, 4):
                    coeffs, residuals, cov, score = scored_least_squares(
                        sizes, y, degree
                    )

                    print(metric, tool_name, name_map[degree], "BIC Score =", score)
                    threshold = 10  # Demand pretty strong evidence for more complexity
                    if best_score is None or bic_improvement(
                        score, best_score, threshold
                    ):
                        best_score = score
                        best_models[metric][tool_name] = name_map[degree]
                    if name_map[degree] == expected[metric][tool_name][0]:
                        fit_results[metric][tool_name] = coeffs.tolist()
                        z_scores[metric][tool_name] = z_score(
                            coeffs, cov, expected[metric][tool_name][1]
                        )

        for metric, tool_expectation in expected.items():
            for tool_name, (expected_model, expected_param) in tool_expectation.items():
                if tool_name == "WithPickle":
                    # Pickle can be quite noisy, and is anyhow not what we implemented
                    # Leave it in the printouts, but don't fail because of it
                    continue

                with self.subTest(f"{metric} {tool_name} best model"):
                    actual_model = best_models[metric][tool_name]
                    self.assertEqual(
                        actual_model,
                        expected_model,
                        msg=f"Previous data has indicated that {tool_name} should "
                        f"scale {expected_model} with respect to {metric}, but got "
                        f"{actual_model}.",
                    )
                with self.subTest(
                    f"{metric} {tool_name} {expected_model} leading parameter z-score"
                ):
                    threshold = 3  # flag three-sigma results
                    self.assertLess(
                        z_scores[metric][tool_name],
                        threshold,
                        msg=f"Expected z-score < {threshold} but got "
                        f"{z_scores[metric][tool_name]} -- actual and expected "
                        f"parameters were {fit_results[metric][tool_name][0]} and "
                        f"{expected_param}, respectively.",
                    )

        print("Fit results:")
        for metric_name, parameters_dict in fit_results.items():
            print(metric_name)
            for tool_name, coefficients in parameters_dict.items():
                print(
                    f"  {tool_name}: {expected[metric_name][tool_name][0]}, params = "
                    f"{coefficients}; expected {expected[metric_name][tool_name][1]}"
                )


def scored_least_squares(x, y, degree: int):
    n = len(x)
    X = np.vander(x, N=degree + 1, increasing=False)
    coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)

    dof = max(n - degree - 1, 1)
    sigma2 = residuals[0] / dof if residuals.size > 0 else 0
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = sigma2 * XtX_inv

    rss = residuals[0] if residuals.size > 0 else np.sum((y - X @ coeffs) ** 2)
    score = bayesian_information_criterion(degree, n, rss)

    return coeffs, residuals, cov, score


def bayesian_information_criterion(degree: int, n: int, rss) -> float:
    return float((degree + 1) * np.log(n) + n * np.log(rss / n))


def bic_improvement(
    score: float, best_score: float, improvement_threshold: float = 0.0
) -> bool:
    return score < (best_score - improvement_threshold)


def z_score(
    coeffs: list[float], covariance: NDArray[np.float64], expected: float
) -> float:
    std_leading = np.sqrt(covariance[0, 0])
    return float(abs(coeffs[0] - expected) / std_leading if std_leading > 0 else np.inf)
