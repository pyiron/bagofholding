import dataclasses
import hashlib
import importlib
import inspect
import pathlib
import pkgutil
import unittest
from typing import Any

from packaging import version as packaging_version
from static import assertions, compat

import bagofholding
from bagofholding import bag

ARTEFACT_DIR = pathlib.Path(compat.__file__).parent / "artefacts"
BAG_CLASSES: dict[str, type[bag.Bag]] = {
    cls.__name__: cls for cls in (bagofholding.H5Bag, bagofholding.TrieH5Bag)
}


@dataclasses.dataclass(frozen=True)
class Artefact:
    path: pathlib.Path
    module_name: str
    bag_class: type[bag.Bag]

    def saved_version(self) -> packaging_version.Version:
        info = self.bag_class(self.path, bag_version_validator="none").bag_info
        return packaging_version.Version(str(info.version))

    def build(self) -> dict[str, Any]:
        cases: dict[str, Any] = importlib.import_module(
            f"static.compat.{self.module_name}"
        ).build()
        return cases


def discover_artefacts() -> list[Artefact]:
    artefacts = []
    for path in sorted(ARTEFACT_DIR.glob("*.h5")):
        module_name, class_name = path.stem.split(".")
        artefacts.append(Artefact(path, module_name, BAG_CLASSES[class_name]))
    return artefacts


def compat_module_names() -> list[str]:
    return sorted(
        info.name
        for info in pkgutil.iter_modules(compat.__path__)
        if info.name[0] == "v"
    )


def module_version(module_name: str) -> packaging_version.Version:
    return packaging_version.Version(module_name.removeprefix("v").replace("_", "."))


def in_series(
    saved: packaging_version.Version, current: packaging_version.Version
) -> bool:
    if current.major == 0:
        return (saved.major, saved.minor) == (current.major, current.minor)
    return saved.major == current.major


def next_series(version: packaging_version.Version) -> str:
    if version.major == 0:
        return f"0.{version.minor + 1}.0"
    return f"{version.major + 1}.0.0"


def expected_default(current: packaging_version.Version) -> str:
    return "semantic-minor" if current.major == 0 else "semantic-major"


def current_version() -> packaging_version.Version:
    return packaging_version.Version(bagofholding.__version__)


def missing_series_message(raw_version: str) -> str:
    return (
        f"bagofholding is now {raw_version} -- generate compat artefacts for this "
        f"series (see tests/compat/generate.py)"
    )


class TestSeriesHelpers(unittest.TestCase):
    def test_series_helpers(self):
        v = packaging_version.Version
        self.assertTrue(in_series(v("0.1.0"), v("0.1.15.dev3+gabc")))
        self.assertTrue(in_series(v("0.1.0"), v("0.1.dev1+gabc")))
        self.assertFalse(in_series(v("0.1.0"), v("0.2.0")))
        self.assertFalse(in_series(v("0.1.0"), v("0.0.0+unknown")))
        self.assertTrue(in_series(v("1.0.0"), v("1.7.2")))
        self.assertFalse(in_series(v("1.9.0"), v("2.0.0")))
        self.assertEqual("0.2.0", next_series(v("0.1.14")))
        self.assertEqual("2.0.0", next_series(v("1.4.0")))
        self.assertEqual("semantic-minor", expected_default(v("0.9.9")))
        self.assertEqual("semantic-major", expected_default(v("1.0.0")))
        self.assertIn("0.0.0+unknown", missing_series_message("0.0.0+unknown"))


class TestCompat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artefacts = discover_artefacts()
        cls.current = current_version()

    def test_artefacts_present(self):
        self.assertTrue(self.artefacts, msg=f"No artefacts found in {ARTEFACT_DIR}")

    def test_every_compat_module_has_artefacts(self):
        existing = {(a.module_name, a.bag_class) for a in self.artefacts}
        for module_name in compat_module_names():
            for bag_class in BAG_CLASSES.values():
                floor = bag_class.min_compatible_version
                if floor is not None and module_version(
                    module_name
                ) < packaging_version.Version(floor):
                    continue
                with self.subTest(module=module_name, bag_class=bag_class.__name__):
                    self.assertIn(
                        (module_name, bag_class),
                        existing,
                        msg="Generate it with tests/compat/generate.py",
                    )

    def test_current_series_has_artefacts(self):
        self.assertTrue(
            any(in_series(a.saved_version(), self.current) for a in self.artefacts),
            msg=missing_series_message(bagofholding.__version__),
        )

    def test_default_validator_matches_maturity(self):
        default = (
            inspect.signature(bag.Bag.__init__)
            .parameters["bag_version_validator"]
            .default
        )
        self.assertEqual(expected_default(self.current), default)

    def test_in_series_artefacts_load(self):
        for artefact in self.artefacts:
            if not in_series(artefact.saved_version(), self.current):
                continue
            with self.subTest(artefact.path.name):
                loaded = artefact.bag_class(artefact.path).load(
                    version_validator="none"
                )
                expected = artefact.build()
                self.assertEqual(set(expected), set(loaded))
                for key, obj in expected.items():
                    with self.subTest(key=key):
                        assertions.assert_roundtrip_equal(self, obj, loaded[key])

    def test_artefacts_rejected_out_of_series(self):
        for artefact in self.artefacts:
            with self.subTest(artefact.path.name):
                newer = next_series(artefact.saved_version())
                info = dataclasses.replace(
                    artefact.bag_class.get_bag_info(), version=newer
                )
                future_bag = type(
                    artefact.bag_class.__name__,
                    (artefact.bag_class,),
                    {
                        "get_bag_info": classmethod(lambda cls, info=info: info),
                        "__module__": artefact.bag_class.__module__,
                    },
                )
                with self.assertRaises(bagofholding.BagMismatchError):
                    future_bag(artefact.path)

    def test_artefacts_unmodified_by_reading(self):
        for artefact in self.artefacts:
            with self.subTest(artefact.path.name):
                before = hashlib.sha256(artefact.path.read_bytes()).hexdigest()
                artefact.bag_class(artefact.path, bag_version_validator="none").load(
                    version_validator="none"
                )
                after = hashlib.sha256(artefact.path.read_bytes()).hexdigest()
                self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
