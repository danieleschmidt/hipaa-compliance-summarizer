"""Monitoring and observability for HIPAA compliance system."""

from .alerts import AlertManager, AlertRule
from .health_checks import HealthCheckManager, HealthCheckResult, HealthStatus
from .metrics import ComplianceMetrics, MetricsCollector, PerformanceMonitor
from .tracing import Tracer

# Compatibility aliases for legacy imports
HealthChecker = HealthCheckManager
SystemHealth = HealthStatus

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
]
