import abstract_bag_test

from bagofholding.h5.bag import H5Bag


class TestRestructuredH5Bag(abstract_bag_test.AbstractBagTest):
    @classmethod
    def bag_class(cls) -> type[H5Bag]:
        return H5Bag
