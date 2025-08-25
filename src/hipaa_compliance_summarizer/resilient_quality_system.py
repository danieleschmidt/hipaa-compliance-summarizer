"""
Resilient Quality System - Advanced Fault Tolerance & Recovery

This module implements advanced resilience patterns for the quality orchestration
system, including circuit breakers, bulkheads, timeouts, retries, and fallback
mechanisms to ensure robust operation under adverse conditions.
"""

import asyncio
import functools
import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

T = TypeVar('T')


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class RetryStrategy(Enum):
    """Retry strategies."""
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI = "fibonacci"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_recovery_time: float = 30.0
    monitor_requests: int = 10


@dataclass
class RetryConfig:
    """Configuration for retry mechanism."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    max_delay: float = 60.0
    backoff_factor: float = 2.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    retryable_exceptions: tuple = (Exception,)


@dataclass
class BulkheadConfig:
    """Configuration for bulkhead isolation."""
    max_concurrent_calls: int = 10
    queue_size: int = 100
    timeout: float = 30.0


@dataclass
class FallbackConfig:
    """Configuration for fallback mechanism."""
    enabled: bool = True
    fallback_value: Any = None
    fallback_function: Optional[Callable] = None


class ResilientQualityError(Exception):
    """Base exception for resilient quality system."""
    pass


class CircuitBreakerOpenError(ResilientQualityError):
    """Exception raised when circuit breaker is open."""
    pass


class BulkheadFullError(ResilientQualityError):
    """Exception raised when bulkhead is at capacity."""
    pass


class RetryExhaustedError(ResilientQualityError):
    """Exception raised when retry attempts are exhausted."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    Prevents cascading failures by monitoring operation failures
    and temporarily stopping calls when failure rate is too high.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.request_count = 0
        self.success_count = 0
        self.logger = logging.getLogger(f"{__name__}.CircuitBreaker")
        
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for circuit breaker."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self._execute(func, *args, **kwargs)
        return wrapper
    
    async def _execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Retry after {self.config.recovery_timeout}s"
                )
        
        try:
            result = await self._call_function(func, *args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    async def _call_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure >= timedelta(seconds=self.config.recovery_timeout)
    
    def _on_success(self):
        """Handle successful operation."""
        self.success_count += 1
        self.request_count += 1
        
        if self.state == CircuitState.HALF_OPEN:
            self.failure_count = 0
            self.state = CircuitState.CLOSED
            self.logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        self.failure_count += 1
        self.request_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(
                f"Circuit breaker opened due to {self.failure_count} failures. "
                f"Last error: {type(exception).__name__}: {str(exception)}"
            )
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state information."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "request_count": self.request_count,
            "failure_rate": self.failure_count / max(self.request_count, 1),
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
        }


class RetryMechanism:
    """
    Advanced retry mechanism with multiple backoff strategies.
    
    Provides configurable retry logic with different backoff strategies
    to handle transient failures gracefully.
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RetryMechanism")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for retry mechanism."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self._execute_with_retry(func, *args, **kwargs)
        return wrapper
    
    async def _execute_with_retry(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await self._call_function(func, *args, **kwargs)
                if attempt > 1:
                    self.logger.info(f"Function succeeded on attempt {attempt}")
                return result
                
            except self.config.retryable_exceptions as e:
                last_exception = e
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Attempt {attempt} failed: {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        f"All {self.config.max_attempts} attempts failed. "
                        f"Final error: {type(e).__name__}: {str(e)}"
                    )
            
            except Exception as e:
                # Non-retryable exception
                self.logger.error(f"Non-retryable exception: {type(e).__name__}: {str(e)}")
                raise
        
        raise RetryExhaustedError(
            f"Failed after {self.config.max_attempts} attempts. "
            f"Last error: {type(last_exception).__name__}: {str(last_exception)}"
        ) from last_exception
    
    async def _call_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt based on strategy."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            return self.config.initial_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.initial_delay * (self.config.backoff_factor ** (attempt - 1))
            return min(delay, self.config.max_delay)
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.initial_delay * attempt
            return min(delay, self.config.max_delay)
        
        elif self.config.strategy == RetryStrategy.FIBONACCI:
            fib_delay = self._fibonacci_delay(attempt) * self.config.initial_delay
            return min(fib_delay, self.config.max_delay)
        
        else:
            return self.config.initial_delay
    
    def _fibonacci_delay(self, n: int) -> int:
        """Calculate Fibonacci number for delay."""
        if n <= 2:
            return 1
        a, b = 1, 1
        for _ in range(3, n + 1):
            a, b = b, a + b
        return b


class Bulkhead:
    """
    Bulkhead pattern implementation for resource isolation.
    
    Prevents resource exhaustion by limiting concurrent operations
    and isolating failures to specific resource pools.
    """
    
    def __init__(self, config: BulkheadConfig):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.max_concurrent_calls)
        self.active_calls = 0
        self.total_calls = 0
        self.rejected_calls = 0
        self.logger = logging.getLogger(f"{__name__}.Bulkhead")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for bulkhead isolation."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self._execute_with_bulkhead(func, *args, **kwargs)
        return wrapper
    
    async def _execute_with_bulkhead(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with bulkhead protection."""
        self.total_calls += 1
        
        try:
            # Try to acquire semaphore with timeout
            acquired = False
            try:
                await asyncio.wait_for(
                    self.semaphore.acquire(),
                    timeout=self.config.timeout
                )
                acquired = True
            except asyncio.TimeoutError:
                self.rejected_calls += 1
                raise BulkheadFullError(
                    f"Bulkhead capacity exceeded. "
                    f"Active calls: {self.active_calls}, Max: {self.config.max_concurrent_calls}"
                )
            
            self.active_calls += 1
            try:
                result = await self._call_function(func, *args, **kwargs)
                return result
            finally:
                self.active_calls -= 1
                if acquired:
                    self.semaphore.release()
                
        except Exception as e:
            self.logger.error(f"Bulkhead execution failed: {type(e).__name__}: {str(e)}")
            raise
    
    async def _call_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get bulkhead metrics."""
        return {
            "active_calls": self.active_calls,
            "total_calls": self.total_calls,
            "rejected_calls": self.rejected_calls,
            "utilization": self.active_calls / self.config.max_concurrent_calls,
            "rejection_rate": self.rejected_calls / max(self.total_calls, 1),
        }


class FallbackMechanism:
    """
    Fallback mechanism for graceful degradation.
    
    Provides fallback values or alternative functions when
    primary operations fail, ensuring system continues to operate.
    """
    
    def __init__(self, config: FallbackConfig):
        self.config = config
        self.fallback_count = 0
        self.logger = logging.getLogger(f"{__name__}.FallbackMechanism")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator for fallback mechanism."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self._execute_with_fallback(func, *args, **kwargs)
        return wrapper
    
    async def _execute_with_fallback(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with fallback protection."""
        if not self.config.enabled:
            return await self._call_function(func, *args, **kwargs)
        
        try:
            return await self._call_function(func, *args, **kwargs)
        except Exception as e:
            self.fallback_count += 1
            self.logger.warning(
                f"Primary function failed: {type(e).__name__}: {str(e)}. "
                f"Using fallback mechanism."
            )
            
            if self.config.fallback_function:
                try:
                    return await self._call_function(self.config.fallback_function, *args, **kwargs)
                except Exception as fallback_error:
                    self.logger.error(f"Fallback function also failed: {fallback_error}")
                    if self.config.fallback_value is not None:
                        return self.config.fallback_value
                    raise fallback_error
            elif self.config.fallback_value is not None:
                return self.config.fallback_value
            else:
                # Re-raise original exception if no fallback configured
                raise
    
    async def _call_function(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Call the function, handling both sync and async."""
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_metrics(self) -> Dict[str, int]:
        """Get fallback metrics."""
        return {"fallback_count": self.fallback_count}


@dataclass
class HealthCheck:
    """Health check definition."""
    name: str
    check_function: Callable[[], bool]
    timeout: float = 5.0
    critical: bool = True
    interval: float = 30.0


class HealthMonitor:
    """
    Health monitoring system for quality components.
    
    Monitors system health and provides early warning of issues
    before they cause system failures.
    """
    
    def __init__(self):
        self.health_checks: Dict[str, HealthCheck] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.monitoring_active = False
        self.logger = logging.getLogger(f"{__name__}.HealthMonitor")
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.health_status[health_check.name] = {
            "status": "unknown",
            "last_check": None,
            "consecutive_failures": 0,
        }
    
    async def start_monitoring(self):
        """Start health monitoring."""
        self.monitoring_active = True
        self.logger.info("Health monitoring started")
        
        # Start monitoring tasks
        tasks = []
        for check_name, health_check in self.health_checks.items():
            task = asyncio.create_task(self._monitor_health_check(check_name, health_check))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        self.logger.info("Health monitoring stopped")
    
    async def _monitor_health_check(self, check_name: str, health_check: HealthCheck):
        """Monitor a specific health check."""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Execute health check with timeout
                is_healthy = await asyncio.wait_for(
                    self._execute_health_check(health_check.check_function),
                    timeout=health_check.timeout
                )
                
                execution_time = time.time() - start_time
                
                # Update health status
                self.health_status[check_name].update({
                    "status": "healthy" if is_healthy else "unhealthy",
                    "last_check": datetime.now().isoformat(),
                    "execution_time": execution_time,
                    "consecutive_failures": 0 if is_healthy else self.health_status[check_name]["consecutive_failures"] + 1,
                })
                
                if not is_healthy:
                    self.logger.warning(f"Health check '{check_name}' failed")
                
            except asyncio.TimeoutError:
                self.logger.error(f"Health check '{check_name}' timed out")
                self.health_status[check_name].update({
                    "status": "timeout",
                    "last_check": datetime.now().isoformat(),
                    "consecutive_failures": self.health_status[check_name]["consecutive_failures"] + 1,
                })
            
            except Exception as e:
                self.logger.error(f"Health check '{check_name}' error: {e}")
                self.health_status[check_name].update({
                    "status": "error",
                    "last_check": datetime.now().isoformat(),
                    "error": str(e),
                    "consecutive_failures": self.health_status[check_name]["consecutive_failures"] + 1,
                })
            
            # Wait for next check
            await asyncio.sleep(health_check.interval)
    
    async def _execute_health_check(self, check_function: Callable[[], bool]) -> bool:
        """Execute a health check function."""
        if asyncio.iscoroutinefunction(check_function):
            return await check_function()
        else:
            return check_function()
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        total_checks = len(self.health_status)
        healthy_checks = sum(1 for status in self.health_status.values() if status["status"] == "healthy")
        critical_failed = any(
            status["status"] != "healthy" and self.health_checks[name].critical
            for name, status in self.health_status.items()
        )
        
        overall_status = "healthy" if not critical_failed and healthy_checks == total_checks else "degraded"
        if critical_failed:
            overall_status = "unhealthy"
        
        return {
            "overall_status": overall_status,
            "healthy_checks": healthy_checks,
            "total_checks": total_checks,
            "health_ratio": healthy_checks / max(total_checks, 1),
            "individual_status": self.health_status,
        }


class ResilientQualitySystem:
    """
    Comprehensive resilient quality system combining all resilience patterns.
    
    Provides a unified interface for applying multiple resilience patterns
    to quality operations, ensuring robust and fault-tolerant execution.
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_mechanisms: Dict[str, RetryMechanism] = {}
        self.bulkheads: Dict[str, Bulkhead] = {}
        self.fallback_mechanisms: Dict[str, FallbackMechanism] = {}
        self.health_monitor = HealthMonitor()
        self.logger = logging.getLogger(f"{__name__}.ResilientQualitySystem")
    
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Create and register a circuit breaker."""
        circuit_breaker = CircuitBreaker(config)
        self.circuit_breakers[name] = circuit_breaker
        return circuit_breaker
    
    def create_retry_mechanism(self, name: str, config: RetryConfig) -> RetryMechanism:
        """Create and register a retry mechanism."""
        retry_mechanism = RetryMechanism(config)
        self.retry_mechanisms[name] = retry_mechanism
        return retry_mechanism
    
    def create_bulkhead(self, name: str, config: BulkheadConfig) -> Bulkhead:
        """Create and register a bulkhead."""
        bulkhead = Bulkhead(config)
        self.bulkheads[name] = bulkhead
        return bulkhead
    
    def create_fallback_mechanism(self, name: str, config: FallbackConfig) -> FallbackMechanism:
        """Create and register a fallback mechanism."""
        fallback_mechanism = FallbackMechanism(config)
        self.fallback_mechanisms[name] = fallback_mechanism
        return fallback_mechanism
    
    def resilient_operation(
        self,
        operation_name: str,
        circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        bulkhead_config: Optional[BulkheadConfig] = None,
        fallback_config: Optional[FallbackConfig] = None,
    ) -> Callable:
        """
        Decorator that applies multiple resilience patterns to an operation.
        
        Args:
            operation_name: Name of the operation for tracking
            circuit_breaker_config: Circuit breaker configuration
            retry_config: Retry mechanism configuration  
            bulkhead_config: Bulkhead configuration
            fallback_config: Fallback mechanism configuration
            
        Returns:
            Decorated function with resilience patterns applied
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            decorated_func = func
            
            # Apply resilience patterns in order: fallback -> retry -> circuit breaker -> bulkhead
            if fallback_config:
                fallback = self.create_fallback_mechanism(f"{operation_name}_fallback", fallback_config)
                decorated_func = fallback(decorated_func)
            
            if retry_config:
                retry = self.create_retry_mechanism(f"{operation_name}_retry", retry_config)
                decorated_func = retry(decorated_func)
            
            if circuit_breaker_config:
                circuit_breaker = self.create_circuit_breaker(f"{operation_name}_circuit_breaker", circuit_breaker_config)
                decorated_func = circuit_breaker(decorated_func)
            
            if bulkhead_config:
                bulkhead = self.create_bulkhead(f"{operation_name}_bulkhead", bulkhead_config)
                decorated_func = bulkhead(decorated_func)
            
            return decorated_func
        
        return decorator
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system resilience metrics."""
        return {
            "circuit_breakers": {
                name: cb.get_state() for name, cb in self.circuit_breakers.items()
            },
            "bulkheads": {
                name: bulkhead.get_metrics() for name, bulkhead in self.bulkheads.items()
            },
            "fallbacks": {
                name: fb.get_metrics() for name, fb in self.fallback_mechanisms.items()
            },
            "health": self.health_monitor.get_overall_health(),
        }
    
    async def start_health_monitoring(self):
        """Start system health monitoring."""
        # Register default health checks
        self._register_default_health_checks()
        await self.health_monitor.start_monitoring()
    
    async def stop_health_monitoring(self):
        """Stop system health monitoring."""
        await self.health_monitor.stop_monitoring()
    
    def _register_default_health_checks(self):
        """Register default health checks."""
        # Circuit breaker health check
        def check_circuit_breakers():
            open_breakers = [name for name, cb in self.circuit_breakers.items() 
                           if cb.state == CircuitState.OPEN]
            return len(open_breakers) == 0
        
        self.health_monitor.register_health_check(
            HealthCheck(
                name="circuit_breakers",
                check_function=check_circuit_breakers,
                critical=True,
                interval=30.0
            )
        )
        
        # Bulkhead health check
        def check_bulkheads():
            overloaded = [name for name, bulkhead in self.bulkheads.items()
                         if bulkhead.get_metrics()["utilization"] > 0.9]
            return len(overloaded) == 0
        
        self.health_monitor.register_health_check(
            HealthCheck(
                name="bulkheads",
                check_function=check_bulkheads,
                critical=False,
                interval=15.0
            )
        )


# Global resilient quality system instance
resilient_quality_system = ResilientQualitySystem()


def resilient_quality_gate(
    operation_name: str,
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
    retry_config: Optional[RetryConfig] = None,
    bulkhead_config: Optional[BulkheadConfig] = None,
    fallback_config: Optional[FallbackConfig] = None,
) -> Callable:
    """
    Convenience decorator for applying resilience patterns to quality gate operations.
    
    Usage:
        @resilient_quality_gate(
            "syntax_check",
            circuit_breaker_config=CircuitBreakerConfig(failure_threshold=3),
            retry_config=RetryConfig(max_attempts=3),
            fallback_config=FallbackConfig(fallback_value={"status": "unknown"})
        )
        async def check_syntax():
            # Your quality gate implementation
            pass
    """
    return resilient_quality_system.resilient_operation(
        operation_name=operation_name,
        circuit_breaker_config=circuit_breaker_config,
        retry_config=retry_config,
        bulkhead_config=bulkhead_config,
        fallback_config=fallback_config,
    )