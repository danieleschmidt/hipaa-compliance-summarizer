"""Advanced error handling framework for HIPAA compliance system."""

import logging
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories."""
    VALIDATION = "validation"
    PROCESSING = "processing"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    INTEGRATION = "integration"
    SYSTEM = "system"


@dataclass
class ErrorContext:
    """Error context information."""

    category: ErrorCategory
    severity: ErrorSeverity
    timestamp: datetime
    source: str
    operation: str
    user_id: Optional[str] = None
    document_id: Optional[str] = None
    request_id: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None


class HIPAAError(Exception):
    """Base exception for HIPAA compliance system."""

    def __init__(
        self,
        message: str,
        error_code: str,
        context: ErrorContext,
        original_exception: Optional[Exception] = None
    ):
        """Initialize HIPAA error.
        
        Args:
            message: Error message
            error_code: Unique error code
            context: Error context
            original_exception: Original exception if this is a wrapper
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context
        self.original_exception = original_exception

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "category": self.context.category.value,
            "severity": self.context.severity.value,
            "timestamp": self.context.timestamp.isoformat(),
            "source": self.context.source,
            "operation": self.context.operation,
            "user_id": self.context.user_id,
            "document_id": self.context.document_id,
            "request_id": self.context.request_id,
            "additional_context": self.context.additional_context,
            "traceback": traceback.format_exception(
                type(self), self, self.__traceback__
            ) if self.__traceback__ else None,
            "original_exception": str(self.original_exception) if self.original_exception else None
        }


class ValidationError(HIPAAError):
    """Validation error."""
    pass


class ProcessingError(HIPAAError):
    """Processing error."""
    pass


class SecurityError(HIPAAError):
    """Security-related error."""
    pass


class ComplianceError(HIPAAError):
    """Compliance violation error."""
    pass


class IntegrationError(HIPAAError):
    """External integration error."""
    pass


class SystemError(HIPAAError):
    """System-level error."""
    pass


class ErrorHandler:
    """Advanced error handling with recovery strategies."""

    def __init__(self):
        """Initialize error handler."""
        self.error_history: List[HIPAAError] = []
        self.retry_strategies: Dict[str, callable] = {}
        self.error_callbacks: Dict[ErrorCategory, List[callable]] = {}

    def register_retry_strategy(self, error_code: str, strategy: callable):
        """Register retry strategy for specific error code.
        
        Args:
            error_code: Error code to handle
            strategy: Retry strategy function
        """
        self.retry_strategies[error_code] = strategy
        logger.info(f"Registered retry strategy for error code: {error_code}")

    def register_error_callback(self, category: ErrorCategory, callback: callable):
        """Register callback for error categories.
        
        Args:
            category: Error category
            callback: Callback function
        """
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)
        logger.info(f"Registered error callback for category: {category}")

    def handle_error(
        self,
        error: Exception,
        context: ErrorContext,
        error_code: Optional[str] = None,
        auto_retry: bool = True
    ) -> Optional[Any]:
        """Handle error with recovery strategies.
        
        Args:
            error: Original exception
            context: Error context
            error_code: Optional error code
            auto_retry: Whether to attempt automatic retry
            
        Returns:
            Result of recovery strategy if successful, None otherwise
        """
        # Convert to HIPAA error if needed
        if not isinstance(error, HIPAAError):
            hipaa_error = self._convert_to_hipaa_error(error, context, error_code)
        else:
            hipaa_error = error

        # Add to error history
        self.error_history.append(hipaa_error)

        # Log error
        logger.error(
            f"Error handled: {hipaa_error.error_code}",
            extra={
                "error_data": hipaa_error.to_dict(),
                "structured": True
            }
        )

        # Execute callbacks
        self._execute_callbacks(hipaa_error)

        # Attempt recovery if enabled
        if auto_retry and hipaa_error.error_code in self.retry_strategies:
            try:
                recovery_result = self.retry_strategies[hipaa_error.error_code](hipaa_error)
                logger.info(f"Error recovery successful for: {hipaa_error.error_code}")
                return recovery_result
            except Exception as recovery_error:
                logger.error(f"Error recovery failed: {recovery_error}")

        return None

    def _convert_to_hipaa_error(
        self,
        error: Exception,
        context: ErrorContext,
        error_code: Optional[str] = None
    ) -> HIPAAError:
        """Convert standard exception to HIPAA error."""
        if error_code is None:
            error_code = f"{context.category}_{type(error).__name__}".upper()

        # Map to specific HIPAA error types
        error_mapping = {
            ErrorCategory.VALIDATION: ValidationError,
            ErrorCategory.PROCESSING: ProcessingError,
            ErrorCategory.SECURITY: SecurityError,
            ErrorCategory.COMPLIANCE: ComplianceError,
            ErrorCategory.INTEGRATION: IntegrationError,
            ErrorCategory.SYSTEM: SystemError,
        }

        error_class = error_mapping.get(context.category, HIPAAError)

        return error_class(
            message=str(error),
            error_code=error_code,
            context=context,
            original_exception=error
        )

    def _execute_callbacks(self, error: HIPAAError):
        """Execute registered callbacks for error category."""
        callbacks = self.error_callbacks.get(error.context.category, [])
        for callback in callbacks:
            try:
                callback(error)
            except Exception as callback_error:
                logger.error(f"Error callback failed: {callback_error}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        if not self.error_history:
            return {"total_errors": 0}

        category_counts = {}
        severity_counts = {}

        for error in self.error_history:
            category = error.context.category.value
            severity = error.context.severity.value

            category_counts[category] = category_counts.get(category, 0) + 1
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        return {
            "total_errors": len(self.error_history),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "recent_errors": [
                error.to_dict() for error in self.error_history[-10:]
            ]
        }


class CircuitBreaker:
    """Circuit breaker for failing operations."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        expected_exception: Type[Exception] = Exception
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Time to wait before attempting reset
            expected_exception: Exception type to trigger circuit breaker
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.expected_exception = expected_exception

        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if self._should_attempt_reset():
                self.state = "HALF_OPEN"
            else:
                raise SystemError(
                    "Circuit breaker is OPEN",
                    "CIRCUIT_BREAKER_OPEN",
                    ErrorContext(
                        category=ErrorCategory.SYSTEM,
                        severity=ErrorSeverity.HIGH,
                        timestamp=datetime.utcnow(),
                        source="circuit_breaker",
                        operation=func.__name__
                    )
                )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return (datetime.utcnow() - self.last_failure_time).total_seconds() >= self.timeout_seconds

    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        self.state = "CLOSED"

    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def setup_error_handling() -> ErrorHandler:
    """Setup global error handling configuration."""
    error_handler = ErrorHandler()

    # Register default retry strategies
    def retry_network_error(error: HIPAAError):
        """Retry strategy for network errors."""
        import random
        import time

        # Exponential backoff with jitter
        for attempt in range(3):
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            time.sleep(wait_time)
            logger.info(f"Retrying network operation, attempt {attempt + 1}")
            # Return None to indicate retry failed
        return None

    def retry_processing_error(error: HIPAAError):
        """Retry strategy for processing errors."""
        # Simple retry for processing errors
        logger.info("Retrying processing operation")
        return None

    error_handler.register_retry_strategy("INTEGRATION_CONNECTIONERROR", retry_network_error)
    error_handler.register_retry_strategy("PROCESSING_TIMEOUT", retry_processing_error)

    # Register default callbacks
    def security_alert_callback(error: HIPAAError):
        """Send security alerts."""
        if error.context.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.critical(f"SECURITY ALERT: {error.message}", extra={"alert": True})

    def compliance_audit_callback(error: HIPAAError):
        """Log compliance violations for audit."""
        logger.warning(f"COMPLIANCE EVENT: {error.message}", extra={"audit": True})

    error_handler.register_error_callback(ErrorCategory.SECURITY, security_alert_callback)
    error_handler.register_error_callback(ErrorCategory.COMPLIANCE, compliance_audit_callback)

    return error_handler


# Global error handler instance
global_error_handler = setup_error_handling()


def handle_errors(
    category: ErrorCategory,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    source: str = "unknown",
    operation: str = "unknown"
):
    """Decorator for automatic error handling.
    
    Args:
        category: Error category
        severity: Error severity
        source: Error source
        operation: Operation name
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = ErrorContext(
                    category=category,
                    severity=severity,
                    timestamp=datetime.utcnow(),
                    source=source,
                    operation=operation
                )

                result = global_error_handler.handle_error(e, context)
                if result is not None:
                    return result
                raise

        return wrapper
    return decorator
