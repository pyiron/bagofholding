import unittest

from bagofholding import exceptions, retrieve


class SomeClass: ...


class TestRetrieve(unittest.TestCase):
    def test_get_importable_string_from_string_reduction(self):
        obj = SomeClass()
        with self.assertRaises(exceptions.StringNotImportableError):
            retrieve.get_importable_string_from_string_reduction(
                "this_is_not_a_reduction", obj
            )
