import importlib.metadata

try:
    # Installed package will find its version
    __version__ = importlib.metadata.version(__name__)
except importlib.metadata.PackageNotFoundError:
    # Repository clones will register an unknown version
    __version__ = "0.0.0+unknown"

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
    StringNotImportableError as StringNotImportableError,
)
from bagofholding.h5.bag import H5Bag as H5Bag
from bagofholding.h5.triebag import TrieH5Bag as TrieH5Bag
