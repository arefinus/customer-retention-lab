"""Exception types. Every message must say what to do next."""

from __future__ import annotations


class RetentionLabError(Exception):
    """Base class for all package errors."""


class SchemaError(RetentionLabError):
    """A table does not match the expected schema."""


class MissingDataError(RetentionLabError):
    """A required input file is absent. CLI maps this to exit code 2."""


class CensoringError(RetentionLabError):
    """The outcome window for an observation date is not fully inside the panel."""


class SplitError(RetentionLabError):
    """The split cannot be formed as configured."""


class ConfigError(RetentionLabError):
    """Configuration values are inconsistent or out of range."""
