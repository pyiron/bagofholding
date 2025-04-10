import os
import subprocess
import unittest


class TestNotebook(unittest.TestCase):
    def test_notebook_with_nbval(self):
        """
        nbval lets us compare the notebook execution against expected output.
        It's tightly integrated with pytest instead of unittest, so just invoke it
        directly.
        """
        print(__file__)
        result = subprocess.run(["pytest", "--nbval", os.path.join(os.path.dirname(__file__), "../../notebooks/tutorial.ipynb")])
        if result.returncode != 0:
            self.fail(f"nbval failed with exit code {result.returncode}")