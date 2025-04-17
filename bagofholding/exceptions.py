from __future__ import annotations


class BagOfHoldingError(Exception):
    """A base class for raising bagofholding-related exceptions"""


class BagMismatchError(BagOfHoldingError, ValueError): ...


class EnvironmentMismatchError(BagOfHoldingError, ModuleNotFoundError): ...


class FileAlreadyOpenError(BagOfHoldingError): ...


class FileNotOpenError(BagOfHoldingError): ...


class FilepathError(BagOfHoldingError, FileExistsError): ...


class InvalidMetadataError(BagOfHoldingError, ValueError): ...


class ModuleForbiddenError(BagOfHoldingError, ValueError): ...


class NotAGroupError(BagOfHoldingError, TypeError): ...


class NoVersionError(BagOfHoldingError, ValueError): ...


class PickleProtocolError(BagOfHoldingError, ValueError): ...


class StringReductionNotImportableError(BagOfHoldingError): ...
