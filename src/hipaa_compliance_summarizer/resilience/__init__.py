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


# Enhanced resilience implementations
import asyncio
import functools
import logging
import time

logger = logging.getLogger(__name__)

class ResilientExecutor:
    """Production-ready resilient executor with circuit breakers and retries."""
    
    def __init__(self, max_retries: int = 3, circuit_failure_threshold: int = 5):
        self.max_retries = max_retries
        self.circuit_failure_threshold = circuit_failure_threshold
        self.failure_count = 0
        self.circuit_open = False
        self.last_failure_time = None
    
    async def execute(self, operation: callable, *args, **kwargs):
        """Execute operation with resilience patterns."""
        if self.circuit_open:
            if time.time() - self.last_failure_time > 60:  # 60 second recovery
                self.circuit_open = False
                self.failure_count = 0
            else:
                raise Exception("Circuit breaker is open")
        
        for attempt in range(self.max_retries):
            try:
                result = await operation(*args, **kwargs) if asyncio.iscoroutinefunction(operation) else operation(*args, **kwargs)
                self.failure_count = 0  # Reset on success
                return result
            except Exception as e:
                self.failure_count += 1
                if self.failure_count >= self.circuit_failure_threshold:
                    self.circuit_open = True
                    self.last_failure_time = time.time()
                
                if attempt == self.max_retries - 1:
                    raise e
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

def resilient_operation(max_retries: int = 3, fallback_value=None):
    """Decorator for resilient operations with automatic retry and fallback."""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            executor = ResilientExecutor(max_retries=max_retries)
            try:
                return await executor.execute(func, *args, **kwargs)
            except Exception as e:
                logger.error(f"Operation failed after {max_retries} retries: {e}")
                if fallback_value is not None:
                    return fallback_value
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"Operation failed after {max_retries} retries: {e}")
                        if fallback_value is not None:
                            return fallback_value
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

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
