from . import _version

__version__ = _version.get_versions()["version"]

from bagofholding.bag import BagMismatchError as BagMismatchError
from bagofholding.h5.bag import H5Bag as H5Bag
from bagofholding.metadata import (
    EnvironmentMismatchError as EnvironmentMismatchError,
)
from bagofholding.metadata import (
    ModuleForbiddenError as ModuleForbiddenError,
)
from bagofholding.metadata import (
    NoVersionError as NoVersionError,
)
