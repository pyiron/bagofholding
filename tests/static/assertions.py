from __future__ import annotations

import unittest
from typing import Any

import numpy as np


def assert_roundtrip_equal(
    testcase: unittest.TestCase, obj: Any, reloaded: Any
) -> None:
    testcase.assertIs(type(obj), type(reloaded))
    testcase.assertTrue(
        np.all(obj == reloaded),
        msg=f"Mismatch between {obj} and reloaded {reloaded}",
    )
