"""
Generate load-compatibility artefacts for a frozen compat module.

Usage, from the repository root, in an environment where the *historical*
bagofholding version is installed (not this checkout):

    python tests/compat/generate.py v0_1_0

Recipe for a released version X.Y.Z (no git operations needed):

    uv venv /tmp/boh-X.Y.Z --python 3.12
    uv pip install --python /tmp/boh-X.Y.Z bagofholding==X.Y.Z
    # If a pinned dependency has no wheel for your platform (e.g. mpi4py), use
    # `--no-deps` and install the remaining pins (bidict, h5py, numpy, pygtrie,
    # pyiron_snippets) at their pinned versions where possible.
    /tmp/boh-X.Y.Z/bin/python tests/compat/generate.py vX_Y_Z [--bag-classes ...]

Only generate for bag classes whose `min_compatible_version` is at most X.Y.Z.

The compat module is imported from this checkout's `tests/` so that the
historical library pickles today's (frozen) definitions. Never run in CI.
"""

from __future__ import annotations

import argparse
import importlib
import pathlib
import sys

TESTS_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTEFACT_DIR = TESTS_DIR / "static" / "compat" / "artefacts"


def expected_version(module_name: str) -> str:
    return module_name.removeprefix("v").replace("_", ".")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate load-compatibility artefacts."
    )
    parser.add_argument("module", help="Compat module name, e.g. v0_1_0")
    parser.add_argument(
        "--bag-classes",
        nargs="+",
        default=["H5Bag", "TrieH5Bag"],
        help="Names of bagofholding bag classes to save with (default: all)",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(TESTS_DIR))
    import bagofholding

    if bagofholding.__version__ != expected_version(args.module):
        raise SystemExit(
            f"Installed bagofholding is {bagofholding.__version__}, but "
            f"{args.module} must be generated with {expected_version(args.module)}"
        )

    obj = importlib.import_module(f"static.compat.{args.module}").build()
    ARTEFACT_DIR.mkdir(exist_ok=True)
    for bag_class in (getattr(bagofholding, name) for name in args.bag_classes):
        path = ARTEFACT_DIR / f"{args.module}.{bag_class.__name__}.h5"
        bag_class.save(obj, path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
