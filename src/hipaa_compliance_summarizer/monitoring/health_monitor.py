"""Comprehensive health monitoring system for HIPAA compliance."""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import threading

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    UNKNOWN = "unknown"


class ComponentType(str, Enum):
    """Types of system components to monitor."""
    DATABASE = "database"
    REDIS = "redis"
    EXTERNAL_API = "external_api"
    FILE_SYSTEM = "file_system"
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    NETWORK = "network"
    APPLICATION = "application"
    ML_MODEL = "ml_model"
    QUEUE = "queue"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    component: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    response_time_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    disk_io: Dict[str, int] = field(default_factory=dict)
    active_connections: int = 0
    uptime_seconds: float = 0.0


class HealthCheck(ABC):
    """Abstract base class for health checks."""
    
    def __init__(self, name: str, component_type: ComponentType, 
                 timeout: float = 30.0, critical: bool = True):
        self.name = name
        self.component_type = component_type
        self.timeout = timeout
        self.critical = critical
        self.last_result: Optional[HealthCheckResult] = None
    
    @abstractmethod
    async def check(self) -> HealthCheckResult:
        """Perform the health check."""
        pass
    
    async def run_check(self) -> HealthCheckResult:
        """Run the health check with timeout and error handling."""
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(self.check(), timeout=self.timeout)
            result.response_time_ms = (time.time() - start_time) * 1000
            self.last_result = result
            return result
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {self.timeout}s",
                response_time_ms=(time.time() - start_time) * 1000,
                error="timeout"
            )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"Health check failed: {str(e)}",
                response_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )


class DatabaseHealthCheck(HealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, db_manager, name: str = "database"):
        super().__init__(name, ComponentType.DATABASE)
        self.db_manager = db_manager
    
    async def check(self) -> HealthCheckResult:
        """Check database connectivity."""
        try:
            if self.db_manager and self.db_manager.health_check():
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message="Database connection is healthy",
                    details={"connection_pool": "active"}
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.DOWN,
                    message="Database connection failed"
                )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"Database check failed: {str(e)}",
                error=str(e)
            )


class RedisHealthCheck(HealthCheck):
    """Health check for Redis connectivity."""
    
    def __init__(self, redis_url: str, name: str = "redis"):
        super().__init__(name, ComponentType.REDIS)
        self.redis_url = redis_url
        self.redis_client = None
    
    async def check(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        try:
            if not REDIS_AVAILABLE:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.WARNING,
                    message="Redis library not available",
                    error="redis_not_installed"
                )
            
            if not self.redis_client:
                self.redis_client = redis.from_url(self.redis_url)
            
            # Test connection with ping
            response = self.redis_client.ping()
            
            if response:
                info = self.redis_client.info()
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.HEALTHY,
                    message="Redis connection is healthy",
                    details={
                        "connected_clients": info.get("connected_clients", 0),
                        "used_memory": info.get("used_memory_human", "N/A"),
                        "uptime": info.get("uptime_in_seconds", 0)
                    }
                )
            else:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.DOWN,
                    message="Redis ping failed"
                )
                
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"Redis check failed: {str(e)}",
                error=str(e)
            )


class FileSystemHealthCheck(HealthCheck):
    """Health check for file system access."""
    
    def __init__(self, path: str, min_free_space_gb: float = 1.0, name: str = "filesystem"):
        super().__init__(name, ComponentType.FILE_SYSTEM)
        self.path = path
        self.min_free_space_gb = min_free_space_gb
    
    async def check(self) -> HealthCheckResult:
        """Check file system health."""
        try:
            if PSUTIL_AVAILABLE:
                disk_usage = psutil.disk_usage(self.path)
                free_gb = disk_usage.free / (1024**3)
                total_gb = disk_usage.total / (1024**3)
                used_percent = (disk_usage.used / disk_usage.total) * 100
                
                if free_gb < self.min_free_space_gb:
                    status = HealthStatus.CRITICAL
                    message = f"Low disk space: {free_gb:.2f}GB free"
                elif used_percent > 90:
                    status = HealthStatus.WARNING
                    message = f"High disk usage: {used_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"File system healthy: {free_gb:.2f}GB free"
                
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=status,
                    message=message,
                    details={
                        "path": self.path,
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "used_percent": round(used_percent, 1)
                    }
                )
            else:
                # Basic check without psutil
                import os
                if os.path.exists(self.path) and os.access(self.path, os.R_OK | os.W_OK):
                    return HealthCheckResult(
                        component=self.name,
                        component_type=self.component_type,
                        status=HealthStatus.HEALTHY,
                        message="File system accessible",
                        details={"path": self.path}
                    )
                else:
                    return HealthCheckResult(
                        component=self.name,
                        component_type=self.component_type,
                        status=HealthStatus.DOWN,
                        message=f"File system not accessible: {self.path}"
                    )
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"File system check failed: {str(e)}",
                error=str(e)
            )


class SystemResourceHealthCheck(HealthCheck):
    """Health check for system resources (CPU, Memory)."""
    
    def __init__(self, cpu_threshold: float = 90.0, memory_threshold: float = 90.0,
                 name: str = "system_resources"):
        super().__init__(name, ComponentType.MEMORY)
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
    
    async def check(self) -> HealthCheckResult:
        """Check system resource usage."""
        try:
            if not PSUTIL_AVAILABLE:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.WARNING,
                    message="System monitoring not available (psutil not installed)"
                )
            
            # Get current resource usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Determine status
            if cpu_percent > self.cpu_threshold or memory_percent > self.memory_threshold:
                status = HealthStatus.CRITICAL
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            elif cpu_percent > self.cpu_threshold * 0.8 or memory_percent > self.memory_threshold * 0.8:
                status = HealthStatus.WARNING
                message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                details={
                    "cpu_percent": round(cpu_percent, 1),
                    "memory_percent": round(memory_percent, 1),
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_total_gb": round(memory.total / (1024**3), 2)
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"System resource check failed: {str(e)}",
                error=str(e)
            )


class MLModelHealthCheck(HealthCheck):
    """Health check for ML model availability."""
    
    def __init__(self, ml_manager, name: str = "ml_models"):
        super().__init__(name, ComponentType.ML_MODEL)
        self.ml_manager = ml_manager
    
    async def check(self) -> HealthCheckResult:
        """Check ML model health."""
        try:
            if not self.ml_manager:
                return HealthCheckResult(
                    component=self.name,
                    component_type=self.component_type,
                    status=HealthStatus.WARNING,
                    message="ML manager not initialized"
                )
            
            # Get model status
            model_status = self.ml_manager.get_model_status()
            loaded_models = sum(1 for status in model_status.values() if status.get("loaded", False))
            total_models = len(model_status)
            
            if loaded_models == 0:
                status = HealthStatus.CRITICAL
                message = "No ML models loaded"
            elif loaded_models < total_models:
                status = HealthStatus.WARNING
                message = f"Only {loaded_models}/{total_models} ML models loaded"
            else:
                status = HealthStatus.HEALTHY
                message = f"All {total_models} ML models loaded"
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=status,
                message=message,
                details={
                    "loaded_models": loaded_models,
                    "total_models": total_models,
                    "model_status": model_status
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"ML model check failed: {str(e)}",
                error=str(e)
            )


class ApplicationHealthCheck(HealthCheck):
    """General application health check."""
    
    def __init__(self, app_version: str = "1.0.0", name: str = "application"):
        super().__init__(name, ComponentType.APPLICATION)
        self.app_version = app_version
        self.start_time = time.time()
    
    async def check(self) -> HealthCheckResult:
        """Check application health."""
        try:
            uptime = time.time() - self.start_time
            
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.HEALTHY,
                message="Application is running",
                details={
                    "version": self.app_version,
                    "uptime_seconds": round(uptime, 2),
                    "uptime_human": str(timedelta(seconds=int(uptime))),
                    "start_time": datetime.fromtimestamp(self.start_time).isoformat()
                }
            )
            
        except Exception as e:
            return HealthCheckResult(
                component=self.name,
                component_type=self.component_type,
                status=HealthStatus.DOWN,
                message=f"Application check failed: {str(e)}",
                error=str(e)
            )


class HealthMonitor:
    """Central health monitoring system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.health_checks: List[HealthCheck] = []
        self.last_results: Dict[str, HealthCheckResult] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.max_metrics_history = self.config.get("max_metrics_history", 1440)  # 24 hours at 1 minute intervals
        self.monitoring_active = False
        self.monitor_task = None
        self.alert_callbacks: List[Callable] = []
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to monitor."""
        self.health_checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    def add_alert_callback(self, callback: Callable[[HealthCheckResult], None]):
        """Add callback for health check alerts."""
        self.alert_callbacks.append(callback)
    
    async def check_all(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks."""
        results = {}
        
        # Run all health checks concurrently
        tasks = [check.run_check() for check in self.health_checks]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for check, result in zip(self.health_checks, check_results):
            if isinstance(result, Exception):
                # Create error result for failed checks
                result = HealthCheckResult(
                    component=check.name,
                    component_type=check.component_type,
                    status=HealthStatus.DOWN,
                    message=f"Health check execution failed: {str(result)}",
                    error=str(result)
                )
            
            results[check.name] = result
            self.last_results[check.name] = result
            
            # Trigger alerts for critical issues
            if result.status in [HealthStatus.CRITICAL, HealthStatus.DOWN]:
                await self._trigger_alerts(result)
        
        return results
    
    async def _trigger_alerts(self, result: HealthCheckResult):
        """Trigger alerts for critical health issues."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(result)
                else:
                    callback(result)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        metrics = SystemMetrics()
        
        if PSUTIL_AVAILABLE:
            try:
                metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
                
                memory = psutil.virtual_memory()
                metrics.memory_percent = memory.percent
                
                disk = psutil.disk_usage('/')
                metrics.disk_percent = (disk.used / disk.total) * 100
                
                network = psutil.net_io_counters()
                metrics.network_io = {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                }
                
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    metrics.disk_io = {
                        "read_bytes": disk_io.read_bytes,
                        "write_bytes": disk_io.write_bytes,
                        "read_count": disk_io.read_count,
                        "write_count": disk_io.write_count
                    }
                
                metrics.active_connections = len(psutil.net_connections())
                metrics.uptime_seconds = time.time() - psutil.boot_time()
                
            except Exception as e:
                logger.warning(f"Failed to collect some system metrics: {e}")
        
        return metrics
    
    def record_metrics(self):
        """Record current system metrics."""
        metrics = self.collect_system_metrics()
        self.system_metrics.append(metrics)
        
        # Maintain metrics history size
        if len(self.system_metrics) > self.max_metrics_history:
            self.system_metrics = self.system_metrics[-self.max_metrics_history:]
    
    async def start_monitoring(self, interval: float = 60.0):
        """Start continuous health monitoring."""
        self.monitoring_active = True
        logger.info(f"Starting health monitoring with {interval}s interval")
        
        while self.monitoring_active:
            try:
                # Run health checks
                await self.check_all()
                
                # Collect metrics
                self.record_metrics()
                
                # Wait for next interval
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(min(interval, 30))  # Don't wait too long on error
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self.monitoring_active = False
        logger.info("Stopping health monitoring")
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        if HealthStatus.DOWN in statuses:
            return HealthStatus.DOWN
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_status = self.get_overall_status()
        
        status_counts = {}
        for result in self.last_results.values():
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Get recent metrics
        recent_metrics = self.system_metrics[-1] if self.system_metrics else None
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "last_check": result.timestamp.isoformat(),
                    "response_time_ms": result.response_time_ms
                }
                for name, result in self.last_results.items()
            },
            "status_summary": status_counts,
            "system_metrics": asdict(recent_metrics) if recent_metrics else None,
            "total_checks": len(self.health_checks)
        }
    
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get system metrics history for the last N minutes."""
        if not self.system_metrics:
            return []
        
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        return [
            metrics for metrics in self.system_metrics
            if metrics.timestamp >= cutoff_time
        ]


# Convenience function to create health monitor with common checks
def create_health_monitor(
    db_manager=None,
    redis_url: str = None,
    ml_manager=None,
    file_paths: List[str] = None,
    app_version: str = "1.0.0"
) -> HealthMonitor:
    """Create health monitor with standard checks."""
    
    monitor = HealthMonitor()
    
    # Add application check
    monitor.add_health_check(ApplicationHealthCheck(app_version))
    
    # Add system resource check
    monitor.add_health_check(SystemResourceHealthCheck())
    
    # Add database check if provided
    if db_manager:
        monitor.add_health_check(DatabaseHealthCheck(db_manager))
    
    # Add Redis check if provided
    if redis_url:
        monitor.add_health_check(RedisHealthCheck(redis_url))
    
    # Add ML model check if provided
    if ml_manager:
        monitor.add_health_check(MLModelHealthCheck(ml_manager))
    
    # Add file system checks if provided
    if file_paths:
        for i, path in enumerate(file_paths):
            monitor.add_health_check(FileSystemHealthCheck(path, name=f"filesystem_{i}"))
    
    return monitor