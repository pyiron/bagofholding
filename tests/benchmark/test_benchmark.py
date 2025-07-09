import abc
import contextlib
import os
import pickle
import platform
import subprocess
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

        cls._thread_env_vars = {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
        cls._prev_env = {}
        for k, v in cls._thread_env_vars.items():
            cls._prev_env[k] = os.environ.get(k)
            os.environ[k] = v

    @classmethod
    def tearDownClass(cls):
        for k, v in cls._prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

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

                t0 = time.perf_counter()
                with bag:
                    for _ in range(n_reps):
                        bag.load()
                dt_context = time.perf_counter() - t0

                t0 = time.perf_counter()
                for _ in range(n_reps):
                    bag.load()
                dt_direct = time.perf_counter() - t0

                with_context_fudge_factor = 1.02 if is_github() else 1
                dt_reference = dt_direct * with_context_fudge_factor

                print(f"H5 with-context benchmark: depth={depth}, reps={n_reps}")
                self.assertLess(
                    dt_context,
                    dt_reference,
                    msg="Expected the with-context speed to be faster since the file "
                    "is not re-opened multiple times...or at least much not slower -- "
                    "locally it's always faster, but sometimes on the remote CI it is "
                    "a hair slower and fails.",
                )
                print(
                    f"With context {dt_context} < {dt_reference} = "
                    f"{with_context_fudge_factor} * {dt_direct} direct access."
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
            repeats: ClassVar[int] = 1

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
            repeats = 1000

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

        sizes = np.arange(start=10, stop=221, step=20)
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
                    t0 = time.perf_counter()
                    scale = tool.save(obj, fname)
                    performance["save (ms)"][tool.__name__][-1] += (
                        time.perf_counter() - t0
                    ) / scale
                    performance["size (mb)"][tool.__name__][-1] = os.path.getsize(fname)
                    t1 = time.perf_counter()
                    scale = tool.load(fname)
                    performance["load (ms)"][tool.__name__][-1] += (
                        time.perf_counter() - t1
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
                "WithPickle": ("quadratic", 4.03e-3),
                "WithH5Bag": ("quadratic", 4.50e-3),
                "WithTrieH5Bag": ("linear", 3.67),
            },
            "save (ms)": {
                "WithPickle": ("linear", 1.99e-3),
                "WithH5Bag": ("quadratic", 6.88e-2),
                "WithTrieH5Bag": ("cubic", 5.38e-4),
            },
            "load (ms)": {
                "WithPickle": ("linear", 1.02e-3),
                "WithH5Bag": ("quadratic", 2.74e-2),
                "WithTrieH5Bag": ("cubic", 2.06e-4),
            },
        }
        # Data from earlier human-supervised runs
        bic_improvement_threshold = 25  # Demand very strong evidence for complexity

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
                    if best_score is None or bic_improvement(
                        score, best_score, bic_improvement_threshold
                    ):
                        best_score = score
                        best_models[metric][tool_name] = name_map[degree]
                    if name_map[degree] == expected[metric][tool_name][0]:
                        fit_results[metric][tool_name] = coeffs.tolist()
                        z_scores[metric][tool_name] = z_score(
                            coeffs, cov, expected[metric][tool_name][1]
                        )

        z_score_threshold_sigma = 3
        for metric, tool_expectation in expected.items():
            for tool_name, (expected_model, expected_param) in tool_expectation.items():

                with self.subTest(f"{metric} {tool_name} best model"):
                    if tool_name == "WithPickle":
                        # Pickle can be quite noisy, and is anyhow not what we implemented
                        # Leave it in the printouts, but don't fail because of it
                        continue

                    actual_model = best_models[metric][tool_name]
                    self.assertEqual(
                        actual_model,
                        expected_model,
                        msg=f"Previous data has indicated that {tool_name} should "
                        f"scale {expected_model} with respect to {metric}, but got "
                        f"{actual_model}.",
                    )

                stored_z_scores_are_reasonable = is_m3_pro()
                if stored_z_scores_are_reasonable:
                    # They were collected manually on my (@liamhuber) macbook
                    # Since I'm the one working on this, I want the tests to be failable
                    # when I run them locally, but overall it's sufficient to check the
                    # scaling behaviour
                    with self.subTest(
                        f"{metric} {tool_name} {expected_model} leading parameter z-score"
                    ):
                        self.assertLess(
                            z_scores[metric][tool_name],
                            z_score_threshold_sigma,
                            msg=f"Expected z-score < {z_score_threshold_sigma} but got "
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


def is_m3_pro():
    if platform.system() != "Darwin":
        return False
    try:
        output = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        )
        return "Apple M3 Pro" in output
    except Exception:
        return False


def is_github():
    return os.environ.get("GITHUB_ACTIONS") == "true"
