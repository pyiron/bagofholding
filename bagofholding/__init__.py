from . import _version

__version__ = _version.get_versions()["version"]

from bagofholding.exception import BagMismatchError as BagMismatchError
from bagofholding.exception import BagOfHoldingError as BagOfHoldingError
from bagofholding.exception import EnvironmentMismatchError as EnvironmentMismatchError
from bagofholding.exception import FileAlreadyOpenError as FileAlreadyOpenError
from bagofholding.exception import FileNotOpenError as FileNotOpenError
from bagofholding.exception import FilepathError as FilepathError
from bagofholding.exception import InvalidMetadataError as InvalidMetadataError
from bagofholding.exception import ModuleForbiddenError as ModuleForbiddenError
from bagofholding.exception import NotAGroupError as NotAGroupError
from bagofholding.exception import NoVersionError as NoVersionError
from bagofholding.exception import PickleProtocolError as PickleProtocolError
from bagofholding.exception import (
    StringReductionNotImportableError as StringReductionNotImportableError,
)
from bagofholding.h5.bag import H5Bag as H5Bag
