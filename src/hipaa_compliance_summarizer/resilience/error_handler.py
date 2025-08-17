"""Comprehensive error handling and validation system for HIPAA compliance."""

import asyncio
import functools
import logging
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    NETWORK = "network"
    DATABASE = "database"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    EXTERNAL_API = "external_api"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Available recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    FAIL = "fail"
    ESCALATE = "escalate"
    CIRCUIT_BREAK = "circuit_break"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    component: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ErrorInfo:
    """Comprehensive error information."""
    error_id: str
    exception: Exception
    severity: ErrorSeverity
    category: ErrorCategory
    context: ErrorContext
    stack_trace: str
    recovery_action: RecoveryAction = RecoveryAction.FAIL
    retry_count: int = 0
    max_retries: int = 3
    is_recoverable: bool = True
    user_message: Optional[str] = None
    technical_message: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)


class ValidationError(Exception):
    """Custom validation error with enhanced context."""

    def __init__(self, message: str, field: str = None, value: Any = None,
                 code: str = None, suggestions: List[str] = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.code = code
        self.suggestions = suggestions or []


class ProcessingError(Exception):
    """Custom processing error with context."""

    def __init__(self, message: str, operation: str = None,
                 recoverable: bool = True, retry_after: int = None):
        super().__init__(message)
        self.operation = operation
        self.recoverable = recoverable
        self.retry_after = retry_after


class SecurityError(Exception):
    """Security-related error."""

    def __init__(self, message: str, risk_level: str = "high",
                 action_required: str = None):
        super().__init__(message)
        self.risk_level = risk_level
        self.action_required = action_required


class ComplianceError(Exception):
    """HIPAA compliance-related error."""

    def __init__(self, message: str, compliance_rule: str = None,
                 severity: str = "high", audit_required: bool = True):
        super().__init__(message)
        self.compliance_rule = compliance_rule
        self.severity = severity
        self.audit_required = audit_required


class ErrorClassifier:
    """Classifies errors by type, severity, and recovery action."""

    def __init__(self):
        self.classification_rules = self._build_classification_rules()

    def _build_classification_rules(self) -> Dict[Type[Exception], Dict[str, Any]]:
        """Build error classification rules."""
        return {
            ValidationError: {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "recoverable": True,
                "recovery_action": RecoveryAction.FAIL
            },
            ProcessingError: {
                "category": ErrorCategory.PROCESSING,
                "severity": ErrorSeverity.HIGH,
                "recoverable": True,
                "recovery_action": RecoveryAction.RETRY
            },
            SecurityError: {
                "category": ErrorCategory.SECURITY,
                "severity": ErrorSeverity.CRITICAL,
                "recoverable": False,
                "recovery_action": RecoveryAction.ESCALATE
            },
            ComplianceError: {
                "category": ErrorCategory.COMPLIANCE,
                "severity": ErrorSeverity.CRITICAL,
                "recoverable": False,
                "recovery_action": RecoveryAction.ESCALATE
            },
            ConnectionError: {
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.HIGH,
                "recoverable": True,
                "recovery_action": RecoveryAction.RETRY
            },
            TimeoutError: {
                "category": ErrorCategory.NETWORK,
                "severity": ErrorSeverity.MEDIUM,
                "recoverable": True,
                "recovery_action": RecoveryAction.RETRY
            },
            PermissionError: {
                "category": ErrorCategory.AUTHORIZATION,
                "severity": ErrorSeverity.HIGH,
                "recoverable": False,
                "recovery_action": RecoveryAction.ESCALATE
            },
            MemoryError: {
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.CRITICAL,
                "recoverable": False,
                "recovery_action": RecoveryAction.CIRCUIT_BREAK
            },
            FileNotFoundError: {
                "category": ErrorCategory.RESOURCE,
                "severity": ErrorSeverity.MEDIUM,
                "recoverable": True,
                "recovery_action": RecoveryAction.FALLBACK
            },
            ValueError: {
                "category": ErrorCategory.VALIDATION,
                "severity": ErrorSeverity.MEDIUM,
                "recoverable": True,
                "recovery_action": RecoveryAction.FAIL
            },
            KeyError: {
                "category": ErrorCategory.CONFIGURATION,
                "severity": ErrorSeverity.MEDIUM,
                "recoverable": True,
                "recovery_action": RecoveryAction.FALLBACK
            },
        }

    def classify(self, exception: Exception) -> Dict[str, Any]:
        """Classify an exception."""
        exc_type = type(exception)

        # Check for exact match
        if exc_type in self.classification_rules:
            return self.classification_rules[exc_type].copy()

        # Check for inheritance
        for rule_type, rule_data in self.classification_rules.items():
            if isinstance(exception, rule_type):
                return rule_data.copy()

        # Default classification for unknown errors
        return {
            "category": ErrorCategory.UNKNOWN,
            "severity": ErrorSeverity.MEDIUM,
            "recoverable": True,
            "recovery_action": RecoveryAction.RETRY
        }


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(self, max_attempts: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, exponential_base: float = 2.0,
                 jitter: bool = True):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt."""
        delay = min(
            self.base_delay * (self.exponential_base ** (attempt - 1)),
            self.max_delay
        )

        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter

        return delay


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time = None
        self._state = "closed"  # closed, open, half_open

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        return self._state

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit breaker."""
        if self._state == "open" and self._last_failure_time:
            time_since_failure = time.time() - self._last_failure_time
            return time_since_failure >= self.recovery_timeout
        return False

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        # Check if circuit is open
        if self._state == "open":
            if self._should_attempt_reset():
                self._state = "half_open"
            else:
                raise ProcessingError(
                    f"Circuit breaker is open. Last failure: {self._last_failure_time}",
                    operation="circuit_breaker",
                    recoverable=False
                )

        try:
            result = func(*args, **kwargs)

            # Success - reset circuit breaker
            if self._state == "half_open":
                self._state = "closed"
                self._failure_count = 0

            return result

        except self.expected_exception:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._failure_count >= self.failure_threshold:
                self._state = "open"
                logger.warning(f"Circuit breaker opened after {self._failure_count} failures")

            raise


class ErrorHandler:
    """Comprehensive error handler with recovery mechanisms."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.classifier = ErrorClassifier()
        self.retry_configs = self._load_retry_configs()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_history: List[ErrorInfo] = []
        self.max_history = self.config.get("max_error_history", 1000)

    def _load_retry_configs(self) -> Dict[ErrorCategory, RetryConfig]:
        """Load retry configurations for different error categories."""
        return {
            ErrorCategory.NETWORK: RetryConfig(max_attempts=5, base_delay=2.0),
            ErrorCategory.DATABASE: RetryConfig(max_attempts=3, base_delay=1.0),
            ErrorCategory.EXTERNAL_API: RetryConfig(max_attempts=4, base_delay=1.5),
            ErrorCategory.PROCESSING: RetryConfig(max_attempts=2, base_delay=0.5),
            ErrorCategory.RESOURCE: RetryConfig(max_attempts=1, base_delay=5.0),
        }

    def get_circuit_breaker(self, operation: str) -> CircuitBreaker:
        """Get or create circuit breaker for operation."""
        if operation not in self.circuit_breakers:
            self.circuit_breakers[operation] = CircuitBreaker()
        return self.circuit_breakers[operation]

    def handle_error(self, exception: Exception, context: ErrorContext) -> ErrorInfo:
        """Handle an error and return error information."""
        import uuid

        # Classify the error
        classification = self.classifier.classify(exception)

        # Create error info
        error_info = ErrorInfo(
            error_id=str(uuid.uuid4()),
            exception=exception,
            severity=ErrorSeverity(classification["severity"]),
            category=ErrorCategory(classification["category"]),
            context=context,
            stack_trace=traceback.format_exc(),
            recovery_action=RecoveryAction(classification["recovery_action"]),
            is_recoverable=classification["recoverable"]
        )

        # Add user-friendly messages
        error_info.user_message = self._generate_user_message(error_info)
        error_info.technical_message = str(exception)
        error_info.suggested_fixes = self._generate_suggestions(error_info)

        # Record error
        self._record_error(error_info)

        # Log error
        self._log_error(error_info)

        return error_info

    def _generate_user_message(self, error_info: ErrorInfo) -> str:
        """Generate user-friendly error message."""
        messages = {
            ErrorCategory.VALIDATION: "The provided data is invalid. Please check your input and try again.",
            ErrorCategory.PROCESSING: "An error occurred while processing your request. Please try again.",
            ErrorCategory.NETWORK: "Unable to connect to external services. Please check your connection and try again.",
            ErrorCategory.DATABASE: "Database connection error. Please try again later.",
            ErrorCategory.AUTHENTICATION: "Authentication failed. Please check your credentials.",
            ErrorCategory.AUTHORIZATION: "You don't have permission to perform this action.",
            ErrorCategory.RESOURCE: "System resources are temporarily unavailable. Please try again later.",
            ErrorCategory.SECURITY: "A security issue was detected. This incident has been reported.",
            ErrorCategory.COMPLIANCE: "A compliance violation was detected. This incident has been reported.",
        }

        return messages.get(error_info.category, "An unexpected error occurred. Please try again or contact support.")

    def _generate_suggestions(self, error_info: ErrorInfo) -> List[str]:
        """Generate suggested fixes for the error."""
        suggestions = {
            ErrorCategory.VALIDATION: [
                "Verify all required fields are provided",
                "Check data format and types",
                "Ensure values are within acceptable ranges"
            ],
            ErrorCategory.NETWORK: [
                "Check internet connection",
                "Verify firewall settings",
                "Try again in a few minutes"
            ],
            ErrorCategory.DATABASE: [
                "Check database connection settings",
                "Verify database is running",
                "Contact system administrator if problem persists"
            ],
            ErrorCategory.AUTHENTICATION: [
                "Verify username and password",
                "Check if account is active",
                "Contact administrator if needed"
            ],
        }

        return suggestions.get(error_info.category, ["Contact support for assistance"])

    def _record_error(self, error_info: ErrorInfo):
        """Record error in history."""
        self.error_history.append(error_info)

        # Maintain history size
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history:]

    def _log_error(self, error_info: ErrorInfo):
        """Log error with appropriate level."""
        log_data = {
            "error_id": error_info.error_id,
            "category": error_info.category.value,
            "severity": error_info.severity.value,
            "operation": error_info.context.operation,
            "component": error_info.context.component,
            "user_id": error_info.context.user_id,
            "session_id": error_info.context.session_id,
        }

        if error_info.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"Critical error: {error_info.technical_message}", extra=log_data)
        elif error_info.severity == ErrorSeverity.HIGH:
            logger.error(f"High severity error: {error_info.technical_message}", extra=log_data)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"Medium severity error: {error_info.technical_message}", extra=log_data)
        else:
            logger.info(f"Low severity error: {error_info.technical_message}", extra=log_data)

    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}

        # Count by category
        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "categories": category_counts,
            "severities": severity_counts,
            "recent_errors": len([e for e in self.error_history
                                if (datetime.utcnow() - e.context.timestamp).total_seconds() < 3600])
        }


# Decorators for error handling

def with_error_handling(operation: str, component: str,
                       retry_config: RetryConfig = None,
                       circuit_breaker: bool = False):
    """Decorator for automatic error handling with retry and circuit breaker."""

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            retry_cfg = retry_config or RetryConfig()

            context = ErrorContext(
                operation=operation,
                component=component,
                metadata={"function": func.__name__}
            )

            # Circuit breaker protection
            if circuit_breaker:
                cb = error_handler.get_circuit_breaker(f"{component}:{operation}")
                return cb.call(_execute_with_retry, func, args, kwargs, error_handler, context, retry_cfg)
            else:
                return _execute_with_retry(func, args, kwargs, error_handler, context, retry_cfg)

        return wrapper
    return decorator


def _execute_with_retry(func: Callable, args: tuple, kwargs: dict,
                       error_handler: ErrorHandler, context: ErrorContext,
                       retry_config: RetryConfig):
    """Execute function with retry logic."""
    last_error = None

    for attempt in range(retry_config.max_attempts):
        try:
            return func(*args, **kwargs)

        except Exception as e:
            last_error = e
            error_info = error_handler.handle_error(e, context)

            # Don't retry if not recoverable
            if not error_info.is_recoverable:
                raise

            # Don't retry on last attempt
            if attempt == retry_config.max_attempts - 1:
                raise

            # Wait before retry
            delay = retry_config.get_delay(attempt + 1)
            logger.info(f"Retrying {context.operation} in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_attempts})")
            time.sleep(delay)

    # Should not reach here, but just in case
    raise last_error


@contextmanager
def error_context(operation: str, component: str, **metadata):
    """Context manager for error handling."""
    error_handler = ErrorHandler()
    context = ErrorContext(
        operation=operation,
        component=component,
        metadata=metadata
    )

    try:
        yield context
    except Exception as e:
        error_info = error_handler.handle_error(e, context)
        logger.error(f"Error in {operation}: {error_info.user_message}")
        raise


# Async versions
async def with_async_error_handling(operation: str, component: str,
                                   retry_config: RetryConfig = None):
    """Async decorator for error handling."""

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()
            retry_cfg = retry_config or RetryConfig()

            context = ErrorContext(
                operation=operation,
                component=component,
                metadata={"function": func.__name__}
            )

            return await _execute_async_with_retry(func, args, kwargs, error_handler, context, retry_cfg)

        return wrapper
    return decorator


async def _execute_async_with_retry(func: Callable, args: tuple, kwargs: dict,
                                   error_handler: ErrorHandler, context: ErrorContext,
                                   retry_config: RetryConfig):
    """Execute async function with retry logic."""
    last_error = None

    for attempt in range(retry_config.max_attempts):
        try:
            return await func(*args, **kwargs)

        except Exception as e:
            last_error = e
            error_info = error_handler.handle_error(e, context)

            # Don't retry if not recoverable
            if not error_info.is_recoverable:
                raise

            # Don't retry on last attempt
            if attempt == retry_config.max_attempts - 1:
                raise

            # Wait before retry
            delay = retry_config.get_delay(attempt + 1)
            logger.info(f"Retrying {context.operation} in {delay:.2f}s (attempt {attempt + 1}/{retry_config.max_attempts})")
            await asyncio.sleep(delay)

    # Should not reach here, but just in case
    raise last_error


# Global error handler instance
global_error_handler = ErrorHandler()


def handle_error(exception: Exception, operation: str, component: str, **metadata) -> ErrorInfo:
    """Convenience function to handle errors globally."""
    context = ErrorContext(
        operation=operation,
        component=component,
        metadata=metadata
    )
    return global_error_handler.handle_error(exception, context)
