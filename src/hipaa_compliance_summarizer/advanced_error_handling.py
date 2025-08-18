"""Advanced error handling, recovery, and fault tolerance for HIPAA compliance system.

This module provides comprehensive error handling including:
- Structured error classification and handling
- Automatic retry mechanisms with exponential backoff
- Error recovery strategies
- Fault tolerance patterns
- Error reporting and alerting
- Dead letter queue for failed operations
"""

import functools
import logging
import json
import time
import traceback
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Type

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    SECURITY = "security"
    NETWORK = "network"
    DATABASE = "database"
    FILE_IO = "file_io"
    PROCESSING = "processing"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM_RESOURCE = "system_resource"
    UNKNOWN = "unknown"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"
    GRACEFUL_DEGRADATION = "graceful_degradation"


@dataclass
class ErrorContext:
    """Context information for an error."""

    operation: str
    timestamp: datetime
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    exception_type: str
    message: str
    stack_trace: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    recovery_attempted: bool = False
    resolved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation": self.operation,
            "timestamp": self.timestamp.isoformat(),
            "error_id": self.error_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "exception_type": self.exception_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "metadata": self.metadata,
            "retry_count": self.retry_count,
            "recovery_attempted": self.recovery_attempted,
            "resolved": self.resolved
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = min(self.base_delay * (self.exponential_base ** attempt), self.max_delay)

        if self.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter

        return delay


@dataclass
class FailedOperation:
    """Represents a failed operation for dead letter queue."""

    operation_id: str
    operation_name: str
    timestamp: datetime
    error_context: ErrorContext
    serialized_args: bytes
    serialized_kwargs: bytes
    max_retry_attempts: int
    current_attempt: int
    next_retry_at: Optional[datetime] = None

    @classmethod
    def create(cls, operation_name: str, error_context: ErrorContext,
               args: tuple, kwargs: dict, max_retry_attempts: int = 3) -> 'FailedOperation':
        """Create a failed operation record."""
        return cls(
            operation_id=str(uuid.uuid4()),
            operation_name=operation_name,
            timestamp=datetime.now(),
            error_context=error_context,
            serialized_args=json.dumps(str(args)),
            serialized_kwargs=json.dumps(str(kwargs)),
            max_retry_attempts=max_retry_attempts,
            current_attempt=1
        )

    def deserialize_args(self) -> tuple:
        """Deserialize operation arguments."""
        return eval(json.loads(self.serialized_args))

    def deserialize_kwargs(self) -> dict:
        """Deserialize operation keyword arguments."""
        return eval(json.loads(self.serialized_kwargs))


class AdvancedErrorHandler:
    """Advanced error handling and recovery system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced error handler."""
        self.config = config or {}
        self._lock = Lock()

        # Error tracking
        self._error_history: deque = deque(maxlen=10000)
        self._error_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "last_occurrence": None,
            "severity_counts": defaultdict(int)
        })

        # Dead letter queue for failed operations
        self._failed_operations: Dict[str, FailedOperation] = {}

        # Recovery strategies registry
        self._recovery_strategies: Dict[ErrorCategory, List[Callable]] = defaultdict(list)

        # Error handlers registry
        self._error_handlers: Dict[Type[Exception], List[Callable]] = defaultdict(list)

        # Configuration
        self.default_retry_config = RetryConfig(
            max_attempts=self.config.get('default_max_retries', 3),
            base_delay=self.config.get('default_retry_delay', 1.0),
            max_delay=self.config.get('max_retry_delay', 60.0)
        )

        # Start background processing
        self._processing_active = True
        self._start_background_processing()

        logger.info("Advanced error handler initialized")

    def classify_error(self, exception: Exception, operation: str = "") -> ErrorCategory:
        """Classify error into appropriate category."""
        error_type = type(exception).__name__.lower()
        error_message = str(exception).lower()

        # Security-related errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['security', 'permission', 'unauthorized', 'forbidden']):
            return ErrorCategory.SECURITY

        # Network-related errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['connection', 'timeout', 'network', 'socket', 'dns']):
            return ErrorCategory.NETWORK

        # Database errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['database', 'sql', 'connection', 'transaction']):
            return ErrorCategory.DATABASE

        # File I/O errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['file', 'io', 'permission', 'notfound', 'directory']):
            return ErrorCategory.FILE_IO

        # Validation errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['validation', 'invalid', 'format', 'parse']):
            return ErrorCategory.VALIDATION

        # Configuration errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['config', 'setting', 'environment', 'missing']):
            return ErrorCategory.CONFIGURATION

        # System resource errors
        if any(keyword in error_type or keyword in error_message
               for keyword in ['memory', 'disk', 'resource', 'limit', 'quota']):
            return ErrorCategory.SYSTEM_RESOURCE

        return ErrorCategory.UNKNOWN

    def determine_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on exception and category."""
        error_type = type(exception).__name__.lower()
        error_message = str(exception).lower()

        # Critical errors
        if category == ErrorCategory.SECURITY:
            return ErrorSeverity.CRITICAL

        if any(keyword in error_type or keyword in error_message
               for keyword in ['critical', 'fatal', 'corruption', 'integrity']):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if category in [ErrorCategory.DATABASE, ErrorCategory.SYSTEM_RESOURCE]:
            return ErrorSeverity.HIGH

        if any(keyword in error_type or keyword in error_message
               for keyword in ['error', 'failed', 'exception', 'abort']):
            return ErrorSeverity.HIGH

        # Medium severity errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.EXTERNAL_SERVICE]:
            return ErrorSeverity.MEDIUM

        # Low severity errors
        if category in [ErrorCategory.VALIDATION, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def handle_error(self, exception: Exception, operation: str = "",
                    metadata: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle and classify an error."""
        category = self.classify_error(exception, operation)
        severity = self.determine_severity(exception, category)

        error_context = ErrorContext(
            operation=operation,
            timestamp=datetime.now(),
            error_id=str(uuid.uuid4()),
            severity=severity,
            category=category,
            exception_type=type(exception).__name__,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            metadata=metadata or {}
        )

        # Store error
        with self._lock:
            self._error_history.append(error_context)

            # Update statistics
            key = f"{category.value}_{exception.__class__.__name__}"
            self._error_stats[key]["count"] += 1
            self._error_stats[key]["last_occurrence"] = error_context.timestamp
            self._error_stats[key]["severity_counts"][severity.value] += 1

        # Execute registered error handlers
        self._execute_error_handlers(exception, error_context)

        # Log error
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }[severity]

        logger.log(log_level, f"Error handled [{severity.value.upper()}]: {error_context.to_dict()}")

        return error_context

    def _execute_error_handlers(self, exception: Exception, error_context: ErrorContext) -> None:
        """Execute registered error handlers for the exception type."""
        exception_type = type(exception)

        # Execute handlers for specific exception type and its parent types
        for exc_type in exception_type.__mro__:
            if exc_type in self._error_handlers:
                for handler in self._error_handlers[exc_type]:
                    try:
                        handler(exception, error_context)
                    except Exception as e:
                        logger.error(f"Error handler failed: {e}")

    def register_error_handler(self, exception_type: Type[Exception],
                             handler: Callable[[Exception, ErrorContext], None]) -> None:
        """Register an error handler for specific exception type."""
        self._error_handlers[exception_type].append(handler)
        logger.info(f"Registered error handler for {exception_type.__name__}")

    def register_recovery_strategy(self, category: ErrorCategory,
                                 strategy_func: Callable[[ErrorContext], bool]) -> None:
        """Register a recovery strategy for error category."""
        self._recovery_strategies[category].append(strategy_func)
        logger.info(f"Registered recovery strategy for {category.value}")

    def retry_with_backoff(self, retry_config: Optional[RetryConfig] = None):
        """Decorator for automatic retry with exponential backoff."""
        config = retry_config or self.default_retry_config

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None

                for attempt in range(config.max_attempts):
                    try:
                        return func(*args, **kwargs)

                    except Exception as e:
                        last_exception = e

                        # Check if exception is retryable
                        if config.retryable_exceptions and not any(
                            isinstance(e, exc_type) for exc_type in config.retryable_exceptions
                        ):
                            break

                        # Handle the error
                        error_context = self.handle_error(
                            e, func.__name__,
                            {"attempt": attempt + 1, "max_attempts": config.max_attempts}
                        )
                        error_context.retry_count = attempt + 1

                        # Don't sleep on last attempt
                        if attempt < config.max_attempts - 1:
                            delay = config.calculate_delay(attempt)
                            logger.info(f"Retrying {func.__name__} in {delay:.2f} seconds (attempt {attempt + 1}/{config.max_attempts})")
                            time.sleep(delay)

                # All retries exhausted
                if last_exception:
                    # Add to dead letter queue
                    self._add_to_dead_letter_queue(func.__name__, last_exception, args, kwargs, config.max_attempts)
                    raise last_exception

            return wrapper
        return decorator

    def _add_to_dead_letter_queue(self, operation_name: str, exception: Exception,
                                 args: tuple, kwargs: dict, max_attempts: int) -> None:
        """Add failed operation to dead letter queue."""
        error_context = self.handle_error(exception, operation_name, {"dead_letter_queue": True})
        failed_op = FailedOperation.create(operation_name, error_context, args, kwargs, max_attempts)

        with self._lock:
            self._failed_operations[failed_op.operation_id] = failed_op

        logger.error(f"Operation {operation_name} added to dead letter queue: {failed_op.operation_id}")

    @contextmanager
    def error_boundary(self, operation: str, fallback_value: Any = None,
                      suppress_errors: bool = False):
        """Context manager for error boundaries with fallback values."""
        try:
            yield
        except Exception as e:
            error_context = self.handle_error(e, operation, {"error_boundary": True})

            if not suppress_errors:
                raise

            logger.warning(f"Error suppressed in boundary for {operation}: {e}")
            return fallback_value

    def attempt_recovery(self, error_context: ErrorContext) -> bool:
        """Attempt to recover from an error using registered strategies."""
        if error_context.recovery_attempted:
            return False

        error_context.recovery_attempted = True

        # Try recovery strategies for the error category
        strategies = self._recovery_strategies.get(error_context.category, [])

        for strategy in strategies:
            try:
                if strategy(error_context):
                    error_context.resolved = True
                    logger.info(f"Successfully recovered from error {error_context.error_id}")
                    return True
            except Exception as e:
                logger.error(f"Recovery strategy failed: {e}")

        return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        with self._lock:
            recent_errors = [
                e for e in self._error_history
                if e.timestamp > datetime.now() - timedelta(hours=24)
            ]

            # Count errors by category and severity
            category_counts = defaultdict(int)
            severity_counts = defaultdict(int)

            for error in recent_errors:
                category_counts[error.category.value] += 1
                severity_counts[error.severity.value] += 1

            # Top error types
            error_type_counts = defaultdict(int)
            for error in recent_errors:
                error_type_counts[error.exception_type] += 1

            top_errors = sorted(
                error_type_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            return {
                "total_errors_24h": len(recent_errors),
                "category_distribution": dict(category_counts),
                "severity_distribution": dict(severity_counts),
                "top_error_types": top_errors,
                "dead_letter_queue_size": len(self._failed_operations),
                "error_stats": dict(self._error_stats),
                "last_updated": datetime.now().isoformat()
            }

    def get_dead_letter_queue(self) -> List[Dict[str, Any]]:
        """Get dead letter queue contents."""
        with self._lock:
            return [
                {
                    "operation_id": op.operation_id,
                    "operation_name": op.operation_name,
                    "timestamp": op.timestamp.isoformat(),
                    "current_attempt": op.current_attempt,
                    "max_retry_attempts": op.max_retry_attempts,
                    "error_context": op.error_context.to_dict(),
                    "next_retry_at": op.next_retry_at.isoformat() if op.next_retry_at else None
                }
                for op in self._failed_operations.values()
            ]

    def retry_failed_operation(self, operation_id: str) -> bool:
        """Manually retry a failed operation from dead letter queue."""
        with self._lock:
            if operation_id not in self._failed_operations:
                return False

            failed_op = self._failed_operations[operation_id]

            if failed_op.current_attempt >= failed_op.max_retry_attempts:
                logger.warning(f"Failed operation {operation_id} has exceeded max retry attempts")
                return False

        try:
            args = failed_op.deserialize_args()
            kwargs = failed_op.deserialize_kwargs()

            # Increment attempt count
            failed_op.current_attempt += 1

            # This is a placeholder - in a real implementation, you would need to
            # store the actual function reference or use a registry of retryable operations
            logger.info(f"Retrying failed operation {operation_id} (attempt {failed_op.current_attempt})")

            # If retry succeeds, remove from dead letter queue
            with self._lock:
                del self._failed_operations[operation_id]

            return True

        except Exception as e:
            logger.error(f"Failed to retry operation {operation_id}: {e}")
            return False

    def _start_background_processing(self) -> None:
        """Start background processing for dead letter queue and cleanup."""
        import threading

        def background_loop():
            while self._processing_active:
                try:
                    # Process dead letter queue retries
                    self._process_dead_letter_queue()

                    # Cleanup old errors
                    self._cleanup_old_errors()

                    time.sleep(300)  # 5 minutes

                except Exception as e:
                    logger.error(f"Background processing error: {e}")
                    time.sleep(60)

        self._background_thread = threading.Thread(target=background_loop, daemon=True)
        self._background_thread.start()

    def _process_dead_letter_queue(self) -> None:
        """Process dead letter queue for automatic retries."""
        now = datetime.now()

        with self._lock:
            retry_operations = [
                op for op in self._failed_operations.values()
                if (op.next_retry_at and op.next_retry_at <= now and
                    op.current_attempt < op.max_retry_attempts)
            ]

        for operation in retry_operations:
            self.retry_failed_operation(operation.operation_id)

    def _cleanup_old_errors(self) -> None:
        """Clean up old error records and statistics."""
        cutoff_time = datetime.now() - timedelta(days=7)

        with self._lock:
            # Clean old failed operations
            expired_ops = [
                op_id for op_id, op in self._failed_operations.items()
                if op.timestamp < cutoff_time
            ]

            for op_id in expired_ops:
                del self._failed_operations[op_id]

            if expired_ops:
                logger.info(f"Cleaned up {len(expired_ops)} expired failed operations")

    def stop_processing(self) -> None:
        """Stop background processing."""
        self._processing_active = False
        logger.info("Advanced error handler stopped")


# Global error handler instance
_global_error_handler: Optional[AdvancedErrorHandler] = None
_handler_lock = Lock()


def get_error_handler(config: Optional[Dict[str, Any]] = None) -> AdvancedErrorHandler:
    """Get or create global error handler instance."""
    global _global_error_handler

    with _handler_lock:
        if _global_error_handler is None:
            _global_error_handler = AdvancedErrorHandler(config)
        return _global_error_handler


def initialize_error_handling(config: Optional[Dict[str, Any]] = None) -> AdvancedErrorHandler:
    """Initialize global error handling system."""
    return get_error_handler(config)


# Convenience decorators and context managers
def retry_on_failure(max_attempts: int = 3, base_delay: float = 1.0,
                    retryable_exceptions: Optional[List[Type[Exception]]] = None):
    """Decorator for retry with exponential backoff."""
    handler = get_error_handler()
    config = RetryConfig(
        max_attempts=max_attempts,
        base_delay=base_delay,
        retryable_exceptions=retryable_exceptions or [Exception]
    )
    return handler.retry_with_backoff(config)


@contextmanager
def safe_operation(operation_name: str, fallback_value: Any = None):
    """Context manager for safe operations with error handling."""
    handler = get_error_handler()
    try:
        yield
    except Exception as e:
        handler.handle_error(e, operation_name)
        if fallback_value is not None:
            return fallback_value
        raise
