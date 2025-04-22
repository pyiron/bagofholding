from . import _version

__version__ = _version.get_versions()["version"]

from bagofholding.exceptions import BagMismatchError as BagMismatchError
from bagofholding.exceptions import BagOfHoldingError as BagOfHoldingError
from bagofholding.exceptions import EnvironmentMismatchError as EnvironmentMismatchError
from bagofholding.exceptions import FileAlreadyOpenError as FileAlreadyOpenError
from bagofholding.exceptions import FileNotOpenError as FileNotOpenError
from bagofholding.exceptions import FilepathError as FilepathError
from bagofholding.exceptions import InvalidMetadataError as InvalidMetadataError
from bagofholding.exceptions import ModuleForbiddenError as ModuleForbiddenError
from bagofholding.exceptions import NotAGroupError as NotAGroupError
from bagofholding.exceptions import NoVersionError as NoVersionError
from bagofholding.exceptions import PickleProtocolError as PickleProtocolError
from bagofholding.exceptions import (
    StringReductionNotImportableError as StringReductionNotImportableError,
)
from bagofholding.h5.bag import H5Bag as H5Bag
from bagofholding.h5.restructured import RestructuredH5Bag as RestructuredH5Bag
