from __future__ import annotations


class BagOfHoldingError(Exception):
    """A base class for raising bagofholding-related exceptions"""


class BagMismatchError(BagOfHoldingError, ValueError):
    pass


class EnvironmentMismatchError(BagOfHoldingError, ModuleNotFoundError):
    pass


class FileAlreadyOpenError(BagOfHoldingError):
    pass


class FileNotOpenError(BagOfHoldingError):
    pass


class FilepathError(BagOfHoldingError, FileExistsError):
    pass


class InvalidMetadataError(BagOfHoldingError, ValueError):
    pass


class ModuleForbiddenError(BagOfHoldingError, ValueError):
    pass


class NotAGroupError(BagOfHoldingError, TypeError):
    pass


class NoVersionError(BagOfHoldingError, ValueError):
    pass


class PickleProtocolError(BagOfHoldingError, ValueError):
    pass


class StringReductionNotImportableError(
    BagOfHoldingError
):  # , ModuleNotFoundError, AttributeError
    pass
