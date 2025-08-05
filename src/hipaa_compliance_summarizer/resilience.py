"""Resilience and recovery mechanisms for HIPAA compliance system."""

import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import threading


logger = logging.getLogger(__name__)


class RetryPolicy(str, Enum):
    """Retry policy types."""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    FIXED_DELAY = "fixed_delay"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE = "immediate"


@dataclass
class RetryConfig:
    """Retry configuration."""
    
    max_attempts: int = 3
    policy: RetryPolicy = RetryPolicy.EXPONENTIAL_BACKOFF
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    jitter: bool = True
    backoff_multiplier: float = 2.0


class ResilientExecutor:
    """Execute operations with resilience patterns."""
    
    def __init__(self):
        """Initialize resilient executor."""
        self.retry_configs: Dict[str, RetryConfig] = {}
        self.execution_history: List[Dict[str, Any]] = []
        
    def register_retry_config(self, operation_name: str, config: RetryConfig):
        """Register retry configuration for operation.
        
        Args:
            operation_name: Name of the operation
            config: Retry configuration
        """
        self.retry_configs[operation_name] = config
        logger.info(f"Registered retry config for operation: {operation_name}")
        
    def execute_with_retry(
        self,
        operation: Callable,
        operation_name: str,
        *args, **kwargs
    ) -> Any:
        """Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            operation_name: Name of the operation
            *args: Arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        config = self.retry_configs.get(operation_name, RetryConfig())
        
        last_exception = None
        execution_record = {
            "operation_name": operation_name,
            "start_time": datetime.utcnow(),
            "attempts": [],
            "success": False,
            "total_duration_seconds": 0.0
        }
        
        start_time = time.time()
        
        for attempt in range(config.max_attempts):
            attempt_start = time.time()
            
            try:
                result = operation(*args, **kwargs)
                
                # Record successful attempt
                execution_record["attempts"].append({
                    "attempt_number": attempt + 1,
                    "success": True,
                    "duration_seconds": time.time() - attempt_start,
                    "error": None
                })
                execution_record["success"] = True
                execution_record["total_duration_seconds"] = time.time() - start_time
                
                self.execution_history.append(execution_record)
                
                logger.info(
                    f"Operation '{operation_name}' succeeded on attempt {attempt + 1}"
                )
                return result
                
            except Exception as e:
                last_exception = e
                attempt_duration = time.time() - attempt_start
                
                # Record failed attempt
                execution_record["attempts"].append({
                    "attempt_number": attempt + 1,
                    "success": False,
                    "duration_seconds": attempt_duration,
                    "error": str(e)
                })
                
                logger.warning(
                    f"Operation '{operation_name}' failed on attempt {attempt + 1}: {e}"
                )
                
                # Don't sleep after the last attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(config, attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    
        # All attempts failed
        execution_record["total_duration_seconds"] = time.time() - start_time
        self.execution_history.append(execution_record)
        
        logger.error(
            f"Operation '{operation_name}' failed after {config.max_attempts} attempts"
        )
        raise last_exception
        
    async def execute_with_retry_async(
        self,
        operation: Callable,
        operation_name: str,
        *args, **kwargs
    ) -> Any:
        """Execute async operation with retry logic."""
        config = self.retry_configs.get(operation_name, RetryConfig())
        
        last_exception = None
        execution_record = {
            "operation_name": operation_name,
            "start_time": datetime.utcnow(),
            "attempts": [],
            "success": False,
            "total_duration_seconds": 0.0
        }
        
        start_time = time.time()
        
        for attempt in range(config.max_attempts):
            attempt_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = operation(*args, **kwargs)
                
                # Record successful attempt
                execution_record["attempts"].append({
                    "attempt_number": attempt + 1,
                    "success": True,
                    "duration_seconds": time.time() - attempt_start,
                    "error": None
                })
                execution_record["success"] = True
                execution_record["total_duration_seconds"] = time.time() - start_time
                
                self.execution_history.append(execution_record)
                
                logger.info(
                    f"Async operation '{operation_name}' succeeded on attempt {attempt + 1}"
                )
                return result
                
            except Exception as e:
                last_exception = e
                attempt_duration = time.time() - attempt_start
                
                # Record failed attempt
                execution_record["attempts"].append({
                    "attempt_number": attempt + 1,
                    "success": False,
                    "duration_seconds": attempt_duration,
                    "error": str(e)
                })
                
                logger.warning(
                    f"Async operation '{operation_name}' failed on attempt {attempt + 1}: {e}"
                )
                
                # Don't sleep after the last attempt
                if attempt < config.max_attempts - 1:
                    delay = self._calculate_delay(config, attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
                    
        # All attempts failed
        execution_record["total_duration_seconds"] = time.time() - start_time
        self.execution_history.append(execution_record)
        
        logger.error(
            f"Async operation '{operation_name}' failed after {config.max_attempts} attempts"
        )
        raise last_exception
        
    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if config.policy == RetryPolicy.IMMEDIATE:
            delay = 0.0
        elif config.policy == RetryPolicy.FIXED_DELAY:
            delay = config.base_delay_seconds
        elif config.policy == RetryPolicy.LINEAR_BACKOFF:
            delay = config.base_delay_seconds * (attempt + 1)
        elif config.policy == RetryPolicy.EXPONENTIAL_BACKOFF:
            delay = config.base_delay_seconds * (config.backoff_multiplier ** attempt)
        else:
            delay = config.base_delay_seconds
            
        # Apply maximum delay limit
        delay = min(delay, config.max_delay_seconds)
        
        # Add jitter if enabled
        if config.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0.0, delay)
        
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {"total_executions": 0}
            
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record["success"])
        
        operation_stats = {}
        for record in self.execution_history:
            op_name = record["operation_name"]
            if op_name not in operation_stats:
                operation_stats[op_name] = {
                    "total": 0,
                    "successful": 0,
                    "average_attempts": 0.0,
                    "average_duration": 0.0
                }
                
            stats = operation_stats[op_name]
            stats["total"] += 1
            if record["success"]:
                stats["successful"] += 1
            stats["average_attempts"] += len(record["attempts"])
            stats["average_duration"] += record["total_duration_seconds"]
            
        # Calculate averages
        for stats in operation_stats.values():
            if stats["total"] > 0:
                stats["average_attempts"] /= stats["total"]
                stats["average_duration"] /= stats["total"]
                stats["success_rate"] = stats["successful"] / stats["total"]
                
        return {
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "overall_success_rate": successful_executions / total_executions if total_executions > 0 else 0.0,
            "by_operation": operation_stats,
            "recent_executions": self.execution_history[-10:]
        }


class HealthMonitor:
    """Monitor system health and trigger recovery actions."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.health_checks: Dict[str, Callable] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.recovery_actions: Dict[str, Callable] = {}
        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval_seconds = 30
        
    def register_health_check(self, name: str, check_func: Callable):
        """Register health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns health status
        """
        self.health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")
        
    def register_recovery_action(self, name: str, action_func: Callable):
        """Register recovery action for unhealthy components.
        
        Args:
            name: Name of the component
            action_func: Recovery action function
        """
        self.recovery_actions[name] = action_func
        logger.info(f"Registered recovery action for: {name}")
        
    def start_monitoring(self, interval_seconds: int = 30):
        """Start continuous health monitoring.
        
        Args:
            interval_seconds: Monitoring interval
        """
        self.check_interval_seconds = interval_seconds
        self.monitoring_active = True
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Health monitoring started with {interval_seconds}s interval")
        
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._perform_health_checks()
                time.sleep(self.check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(5)  # Short delay before retrying
                
    def _perform_health_checks(self):
        """Perform all registered health checks."""
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                health_data = check_func()
                duration = time.time() - start_time
                
                self.health_status[name] = {
                    "healthy": health_data.get("healthy", True),
                    "message": health_data.get("message", "OK"),
                    "details": health_data.get("details", {}),
                    "check_duration_seconds": duration,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                # Trigger recovery if unhealthy
                if not health_data.get("healthy", True) and name in self.recovery_actions:
                    logger.warning(f"Component '{name}' is unhealthy, triggering recovery")
                    try:
                        self.recovery_actions[name]()
                        logger.info(f"Recovery action completed for: {name}")
                    except Exception as recovery_error:
                        logger.error(f"Recovery action failed for '{name}': {recovery_error}")
                        
            except Exception as e:
                self.health_status[name] = {
                    "healthy": False,
                    "message": f"Health check failed: {str(e)}",
                    "details": {"error": str(e)},
                    "check_duration_seconds": 0.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
                logger.error(f"Health check failed for '{name}': {e}")
                
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        overall_healthy = all(
            status.get("healthy", False) for status in self.health_status.values()
        )
        
        return {
            "overall_healthy": overall_healthy,
            "components": self.health_status,
            "last_check": datetime.utcnow().isoformat()
        }


class GracefulShutdown:
    """Handle graceful shutdown of system components."""
    
    def __init__(self):
        """Initialize graceful shutdown handler."""
        self.shutdown_hooks: List[Callable] = []
        self.shutdown_timeout_seconds = 30
        self.shutdown_initiated = False
        
    def register_shutdown_hook(self, hook_func: Callable):
        """Register shutdown hook.
        
        Args:
            hook_func: Function to call during shutdown
        """
        self.shutdown_hooks.append(hook_func)
        logger.info(f"Registered shutdown hook: {hook_func.__name__}")
        
    def initiate_shutdown(self):
        """Initiate graceful shutdown process."""
        if self.shutdown_initiated:
            logger.warning("Shutdown already initiated")
            return
            
        self.shutdown_initiated = True
        logger.info("Initiating graceful shutdown...")
        
        start_time = time.time()
        
        for i, hook in enumerate(self.shutdown_hooks):
            try:
                hook_start = time.time()
                hook()
                hook_duration = time.time() - hook_start
                
                logger.info(f"Shutdown hook {i+1}/{len(self.shutdown_hooks)} completed in {hook_duration:.2f}s")
                
                # Check timeout
                if time.time() - start_time > self.shutdown_timeout_seconds:
                    logger.warning("Shutdown timeout reached, forcing exit")
                    break
                    
            except Exception as e:
                logger.error(f"Shutdown hook {i+1} failed: {e}")
                
        total_duration = time.time() - start_time
        logger.info(f"Graceful shutdown completed in {total_duration:.2f}s")


# Global instances
resilient_executor = ResilientExecutor()
health_monitor = HealthMonitor()
graceful_shutdown = GracefulShutdown()


def resilient_operation(
    operation_name: str,
    retry_config: Optional[RetryConfig] = None
):
    """Decorator for resilient operations.
    
    Args:
        operation_name: Name of the operation
        retry_config: Optional retry configuration
    """
    def decorator(func):
        if retry_config:
            resilient_executor.register_retry_config(operation_name, retry_config)
            
        def wrapper(*args, **kwargs):
            return resilient_executor.execute_with_retry(func, operation_name, *args, **kwargs)
            
        return wrapper
    return decorator


def setup_default_resilience():
    """Setup default resilience configurations."""
    # Default retry configurations
    resilient_executor.register_retry_config(
        "phi_detection",
        RetryConfig(max_attempts=3, policy=RetryPolicy.EXPONENTIAL_BACKOFF)
    )
    
    resilient_executor.register_retry_config(
        "document_processing",
        RetryConfig(max_attempts=2, policy=RetryPolicy.FIXED_DELAY, base_delay_seconds=2.0)
    )
    
    resilient_executor.register_retry_config(
        "compliance_check",
        RetryConfig(max_attempts=3, policy=RetryPolicy.LINEAR_BACKOFF)
    )
    
    # Default health checks
    def system_resources_check():
        """Check system resources."""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            healthy = cpu_percent < 90 and memory.percent < 90
            
            return {
                "healthy": healthy,
                "message": "System resources OK" if healthy else "System resources under stress",
                "details": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3)
                }
            }
        except ImportError:
            return {"healthy": True, "message": "psutil not available"}
        except Exception as e:
            return {"healthy": False, "message": f"Check failed: {e}"}
            
    health_monitor.register_health_check("system_resources", system_resources_check)
    
    logger.info("Default resilience configurations setup completed")


# Initialize default resilience on import
setup_default_resilience()