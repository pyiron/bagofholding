from . import _version

__version__ = _version.get_versions()["version"]

from bagofholding.h5.bag import H5Bag as H5Bag
