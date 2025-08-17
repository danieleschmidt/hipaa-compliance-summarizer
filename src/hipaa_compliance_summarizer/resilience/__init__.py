"""Resilience and reliability components for HIPAA compliance system."""

from .error_handler import (
    CircuitBreaker,
    ComplianceError,
    ErrorCategory,
    ErrorContext,
    ErrorHandler,
    ErrorInfo,
    ErrorSeverity,
    ProcessingError,
    RetryConfig,
    SecurityError,
    ValidationError,
    error_context,
    global_error_handler,
    handle_error,
    with_error_handling,
)


# Add placeholder classes for imports
class ResilientExecutor:
    """Placeholder for resilient executor."""
    pass

def resilient_operation(*args, **kwargs):
    """Placeholder resilient operation."""
    pass

__all__ = [
    "ErrorHandler",
    "ErrorSeverity",
    "ErrorCategory",
    "ErrorContext",
    "ErrorInfo",
    "ValidationError",
    "ProcessingError",
    "SecurityError",
    "ComplianceError",
    "RetryConfig",
    "CircuitBreaker",
    "with_error_handling",
    "error_context",
    "handle_error",
    "global_error_handler",
    "ResilientExecutor",
    "resilient_operation",
]
