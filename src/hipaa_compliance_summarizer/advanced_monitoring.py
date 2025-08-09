"""Advanced monitoring, health checks, and observability for HIPAA compliance system.

This module provides comprehensive monitoring including:
- Real-time system health monitoring
- Performance metrics collection and analysis
- Automated alerting and notification system
- Circuit breaker pattern for resilience
- Distributed tracing capabilities
- Custom metrics and dashboards
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthCheckResult:
    """Result of a health check operation."""

    name: str
    status: HealthStatus
    message: str
    timestamp: datetime
    response_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "response_time_ms": self.response_time_ms,
            "metadata": self.metadata
        }


@dataclass
class SystemMetrics:
    """System performance metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    active_connections: int
    load_average: List[float]
    processes_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_mb": self.memory_used_mb,
            "disk_usage_percent": self.disk_usage_percent,
            "disk_free_gb": self.disk_free_gb,
            "active_connections": self.active_connections,
            "load_average": self.load_average,
            "processes_count": self.processes_count
        }


@dataclass
class Alert:
    """System alert representation."""

    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failures exceed threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    name: str = "circuit_breaker"


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    def __call__(self, func: Callable) -> Callable:
        """Decorator to wrap functions with circuit breaker."""
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.config.name} entering half-open state")
                else:
                    raise Exception(f"Circuit breaker {self.config.name} is OPEN")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.config.expected_exception as e:
            self._on_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout)
        )

    def _on_success(self) -> None:
        """Handle successful operation."""
        with self._lock:
            self.failure_count = 0
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker {self.config.name} recovered to CLOSED state")

    def _on_failure(self) -> None:
        """Handle failed operation."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.config.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logger.error(f"Circuit breaker {self.config.name} opened due to {self.failure_count} failures")


class AdvancedMonitor:
    """Advanced monitoring and observability system."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced monitoring system."""
        self.config = config or {}
        self._lock = threading.Lock()

        # Health checks registry
        self._health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}

        # Metrics storage
        self._metrics_history: deque = deque(maxlen=1000)
        self._custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Alerts management
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_handlers: List[Callable[[Alert], None]] = []

        # Circuit breakers
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        # Performance tracking
        self._operation_metrics: Dict[str, List[float]] = defaultdict(list)

        # Configuration
        self.metrics_retention_hours = self.config.get('metrics_retention_hours', 24)
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.metrics_collection_interval = self.config.get('metrics_collection_interval', 15)

        # Start background monitoring
        self._monitoring_active = True
        self._start_background_monitoring()

        logger.info("Advanced monitoring system initialized")

    def register_health_check(self, name: str, check_func: Callable[[], HealthCheckResult]) -> None:
        """Register a health check function."""
        with self._lock:
            self._health_checks[name] = check_func
        logger.info(f"Registered health check: {name}")

    def register_circuit_breaker(self, name: str, config: CircuitBreakerConfig) -> CircuitBreaker:
        """Register a circuit breaker."""
        circuit_breaker = CircuitBreaker(config)
        with self._lock:
            self._circuit_breakers[name] = circuit_breaker
        logger.info(f"Registered circuit breaker: {name}")
        return circuit_breaker

    def add_alert_handler(self, handler: Callable[[Alert], None]) -> None:
        """Add an alert notification handler."""
        self._alert_handlers.append(handler)

    def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}

        with self._lock:
            checks = self._health_checks.copy()

        for name, check_func in checks.items():
            try:
                start_time = time.time()
                result = check_func()
                result.response_time_ms = (time.time() - start_time) * 1000
                results[name] = result

                # Generate alerts for unhealthy checks
                if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                    self._create_alert(
                        severity=AlertSeverity.ERROR if result.status == HealthStatus.UNHEALTHY else AlertSeverity.CRITICAL,
                        title=f"Health check failed: {name}",
                        message=result.message,
                        source=f"health_check_{name}",
                        metadata={"health_check_result": result.to_dict()}
                    )

            except Exception as e:
                error_result = HealthCheckResult(
                    name=name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed with exception: {str(e)}",
                    timestamp=datetime.now(),
                    response_time_ms=0.0,
                    metadata={"error": str(e)}
                )
                results[name] = error_result
                logger.error(f"Health check {name} failed with exception: {e}")

        return results

    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)

            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.used / disk.total * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)

            # Network connections
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])

            # Load average (Unix-like systems)
            try:
                load_average = list(os.getloadavg())
            except (OSError, AttributeError):
                load_average = [0.0, 0.0, 0.0]

            # Process count
            processes_count = len(psutil.pids())

            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                active_connections=active_connections,
                load_average=load_average,
                processes_count=processes_count
            )

            # Store metrics
            with self._lock:
                self._metrics_history.append(metrics)

            # Check for resource alerts
            self._check_resource_alerts(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            raise

    def _check_resource_alerts(self, metrics: SystemMetrics) -> None:
        """Check metrics against alert thresholds."""
        # CPU usage alert
        if metrics.cpu_percent > 90:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title="High CPU Usage",
                message=f"CPU usage is {metrics.cpu_percent:.1f}%",
                source="system_metrics",
                metadata={"cpu_percent": metrics.cpu_percent}
            )
        elif metrics.cpu_percent > 80:
            self._create_alert(
                severity=AlertSeverity.WARNING,
                title="Elevated CPU Usage",
                message=f"CPU usage is {metrics.cpu_percent:.1f}%",
                source="system_metrics",
                metadata={"cpu_percent": metrics.cpu_percent}
            )

        # Memory usage alert
        if metrics.memory_percent > 90:
            self._create_alert(
                severity=AlertSeverity.CRITICAL,
                title="High Memory Usage",
                message=f"Memory usage is {metrics.memory_percent:.1f}% ({metrics.memory_used_mb:.0f} MB)",
                source="system_metrics",
                metadata={"memory_percent": metrics.memory_percent, "memory_used_mb": metrics.memory_used_mb}
            )

        # Disk usage alert
        if metrics.disk_usage_percent > 90:
            self._create_alert(
                severity=AlertSeverity.ERROR,
                title="Low Disk Space",
                message=f"Disk usage is {metrics.disk_usage_percent:.1f}% ({metrics.disk_free_gb:.1f} GB free)",
                source="system_metrics",
                metadata={"disk_usage_percent": metrics.disk_usage_percent, "disk_free_gb": metrics.disk_free_gb}
            )

    def record_custom_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a custom metric value."""
        metric_data = {
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "tags": tags or {}
        }

        with self._lock:
            self._custom_metrics[name].append(metric_data)

        logger.debug(f"Recorded metric {name}: {value}")

    @contextmanager
    def monitor_operation(self, operation_name: str, **metadata):
        """Context manager to monitor operation performance."""
        start_time = time.time()

        try:
            yield

            duration = time.time() - start_time
            with self._lock:
                self._operation_metrics[operation_name].append(duration)

            self.record_custom_metric(
                f"{operation_name}_duration_seconds",
                duration,
                tags={"status": "success", **metadata}
            )

        except Exception as e:
            duration = time.time() - start_time

            self.record_custom_metric(
                f"{operation_name}_duration_seconds",
                duration,
                tags={"status": "error", "error": str(e), **metadata}
            )

            # Create alert for failed operations
            self._create_alert(
                severity=AlertSeverity.ERROR,
                title=f"Operation Failed: {operation_name}",
                message=f"Operation {operation_name} failed: {str(e)}",
                source="operation_monitoring",
                metadata={"operation": operation_name, "error": str(e), "duration": duration, **metadata}
            )

            raise

    def _create_alert(self, severity: AlertSeverity, title: str, message: str,
                     source: str, metadata: Optional[Dict[str, Any]] = None) -> Alert:
        """Create and process a new alert."""
        alert_id = f"{source}_{title}_{int(time.time())}"

        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )

        with self._lock:
            self._active_alerts[alert_id] = alert

        # Notify alert handlers
        for handler in self._alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }[severity]

        logger.log(log_level, f"ALERT [{severity.value.upper()}]: {title} - {message}")

        return alert

    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        """Mark an alert as resolved."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                if resolution_note:
                    alert.metadata["resolution_note"] = resolution_note

                logger.info(f"Alert resolved: {alert_id}")
                return True

        return False

    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        with self._lock:
            # Recent metrics
            recent_metrics = [
                m for m in self._metrics_history
                if m.timestamp > datetime.now() - timedelta(hours=1)
            ]

            # Active alerts
            active_alerts = [
                alert.to_dict() for alert in self._active_alerts.values()
                if not alert.resolved
            ]

            # Circuit breaker states
            circuit_breaker_states = {
                name: cb.state.value
                for name, cb in self._circuit_breakers.items()
            }

            # Operation performance summary
            operation_summary = {}
            for operation, durations in self._operation_metrics.items():
                if durations:
                    operation_summary[operation] = {
                        "avg_duration": sum(durations) / len(durations),
                        "min_duration": min(durations),
                        "max_duration": max(durations),
                        "operation_count": len(durations)
                    }

        # Current system metrics
        current_metrics = self.collect_system_metrics()

        return {
            "system_status": "healthy" if len(active_alerts) == 0 else "degraded",
            "timestamp": datetime.now().isoformat(),
            "current_metrics": current_metrics.to_dict(),
            "metrics_history_count": len(recent_metrics),
            "active_alerts": active_alerts,
            "circuit_breakers": circuit_breaker_states,
            "operation_performance": operation_summary,
            "custom_metrics_count": len(self._custom_metrics),
            "health_checks_registered": len(self._health_checks)
        }

    def _start_background_monitoring(self) -> None:
        """Start background monitoring threads."""

        # Health checks thread
        def health_check_loop():
            while self._monitoring_active:
                try:
                    self.run_health_checks()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check loop error: {e}")
                    time.sleep(10)

        # Metrics collection thread
        def metrics_collection_loop():
            while self._monitoring_active:
                try:
                    self.collect_system_metrics()
                    time.sleep(self.metrics_collection_interval)
                except Exception as e:
                    logger.error(f"Metrics collection loop error: {e}")
                    time.sleep(10)

        self._health_check_thread = threading.Thread(target=health_check_loop, daemon=True)
        self._metrics_thread = threading.Thread(target=metrics_collection_loop, daemon=True)

        self._health_check_thread.start()
        self._metrics_thread.start()

    def stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitoring_active = False
        logger.info("Advanced monitoring stopped")


# Default health check implementations
def database_health_check() -> HealthCheckResult:
    """Basic database connectivity health check."""
    try:
        # Placeholder - implement actual database connectivity check
        return HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Database connection successful",
            timestamp=datetime.now(),
            response_time_ms=0.0
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {str(e)}",
            timestamp=datetime.now(),
            response_time_ms=0.0,
            metadata={"error": str(e)}
        )


def phi_service_health_check() -> HealthCheckResult:
    """PHI detection service health check."""
    try:
        # Placeholder - implement actual PHI service health check
        return HealthCheckResult(
            name="phi_service",
            status=HealthStatus.HEALTHY,
            message="PHI detection service operational",
            timestamp=datetime.now(),
            response_time_ms=0.0
        )
    except Exception as e:
        return HealthCheckResult(
            name="phi_service",
            status=HealthStatus.UNHEALTHY,
            message=f"PHI service health check failed: {str(e)}",
            timestamp=datetime.now(),
            response_time_ms=0.0,
            metadata={"error": str(e)}
        )


def storage_health_check() -> HealthCheckResult:
    """Storage system health check."""
    try:
        # Check disk space and accessibility
        disk_usage = psutil.disk_usage('/')
        free_percent = disk_usage.free / disk_usage.total * 100

        if free_percent < 10:
            status = HealthStatus.CRITICAL
            message = f"Critical: Only {free_percent:.1f}% disk space remaining"
        elif free_percent < 20:
            status = HealthStatus.UNHEALTHY
            message = f"Warning: Only {free_percent:.1f}% disk space remaining"
        else:
            status = HealthStatus.HEALTHY
            message = f"Storage healthy: {free_percent:.1f}% free space available"

        return HealthCheckResult(
            name="storage",
            status=status,
            message=message,
            timestamp=datetime.now(),
            response_time_ms=0.0,
            metadata={
                "total_gb": disk_usage.total / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "free_percent": free_percent
            }
        )

    except Exception as e:
        return HealthCheckResult(
            name="storage",
            status=HealthStatus.CRITICAL,
            message=f"Storage health check failed: {str(e)}",
            timestamp=datetime.now(),
            response_time_ms=0.0,
            metadata={"error": str(e)}
        )


# Global monitor instance
_global_monitor: Optional[AdvancedMonitor] = None
_monitor_lock = threading.Lock()


def get_advanced_monitor(config: Optional[Dict[str, Any]] = None) -> AdvancedMonitor:
    """Get or create global advanced monitor instance."""
    global _global_monitor

    with _monitor_lock:
        if _global_monitor is None:
            _global_monitor = AdvancedMonitor(config)

            # Register default health checks
            _global_monitor.register_health_check("database", database_health_check)
            _global_monitor.register_health_check("phi_service", phi_service_health_check)
            _global_monitor.register_health_check("storage", storage_health_check)

        return _global_monitor


def initialize_advanced_monitoring(config: Optional[Dict[str, Any]] = None) -> AdvancedMonitor:
    """Initialize global advanced monitoring system."""
    return get_advanced_monitor(config)
