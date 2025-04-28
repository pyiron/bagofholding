import abstract_bag_test

from bagofholding.h5.triebag import TrieH5Bag


class TestRestructuredH5Bag(abstract_bag_test.AbstractBagTest[TrieH5Bag]):
    @classmethod
    def bag_class(cls) -> type[TrieH5Bag]:
        return TrieH5Bag
