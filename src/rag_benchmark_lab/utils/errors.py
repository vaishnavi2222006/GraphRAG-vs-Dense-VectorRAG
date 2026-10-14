class ConfigurationError(ValueError):
    """Raised when user configuration is invalid."""


class ServiceUnavailableError(RuntimeError):
    """Raised when a backend service cannot be reached."""


class StructuredOutputError(ValueError):
    """Raised when model output cannot be parsed safely."""


class IngestionError(RuntimeError):
    """Raised when a document cannot be ingested."""

