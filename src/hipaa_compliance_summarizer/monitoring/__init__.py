"""Monitoring and observability for HIPAA compliance system."""

from .alerts import AlertManager, AlertRule
from .health_checks import HealthCheckManager, HealthCheckResult, HealthStatus
from .metrics import ComplianceMetrics, MetricsCollector, PerformanceMonitor
from .tracing import Tracer

# Compatibility aliases for legacy imports
HealthChecker = HealthCheckManager
SystemHealth = HealthStatus

# Add placeholder for missing imports
class MonitoringDashboard:
    """Placeholder monitoring dashboard."""
    pass

class MetricType:
    """Placeholder metric type."""
    pass

class ProcessingMetrics:
    """Placeholder processing metrics."""
    pass

__all__ = [
    "MetricsCollector",
    "PerformanceMonitor",
    "ComplianceMetrics",
    "HealthCheckManager",
    "HealthStatus",
    "HealthCheckResult",
    "HealthChecker",  # Compatibility alias
    "SystemHealth",   # Compatibility alias
    "AlertManager",
    "AlertRule",
    "Tracer",
    "MonitoringDashboard",
    "MetricType",
    "ProcessingMetrics",
]
