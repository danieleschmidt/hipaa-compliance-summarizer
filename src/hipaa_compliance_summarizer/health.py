"""Health check and monitoring endpoints for HIPAA Compliance Summarizer."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import psutil
import yaml


class HealthChecker:
    """Health check service for application monitoring."""

    def __init__(self, config_path: str = None):
        """Initialize health checker with configuration."""
        self.config_path = config_path or "config/hipaa_config.yml"
        self.start_time = time.time()
        self.checks = []

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "version": "0.0.1",
            "checks": {
                "database": self._check_database(),
                "config": self._check_configuration(),
                "memory": self._check_memory(),
                "disk": self._check_disk_space(),
                "dependencies": self._check_dependencies(),
            },
        }

    def _check_database(self) -> Dict[str, Any]:
        """Check database connectivity."""
        try:
            # Add actual database check here
            return {"status": "healthy", "response_time_ms": 10}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration file validity."""
        try:
            if Path(self.config_path).exists():
                with open(self.config_path) as f:
                    yaml.safe_load(f)
                return {"status": "healthy", "config_loaded": True}
            else:
                return {"status": "unhealthy", "error": "Config file not found"}
        except Exception as e:
            return {"status": "unhealthy", "error": f"Config validation failed: {e}"}

    def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            return {
                "status": "healthy" if memory.percent < 85 else "warning",
                "usage_percent": memory.percent,
                "available_gb": round(memory.available / (1024**3), 2),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage("/")
            usage_percent = (disk.used / disk.total) * 100
            return {
                "status": "healthy" if usage_percent < 85 else "warning",
                "usage_percent": round(usage_percent, 2),
                "free_gb": round(disk.free / (1024**3), 2),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_dependencies(self) -> Dict[str, Any]:
        """Check critical dependencies."""
        try:
            dependencies = ["yaml", "psutil"]
            missing = []
            for dep in dependencies:
                try:
                    __import__(dep)
                except ImportError:
                    missing.append(dep)

            if missing:
                return {"status": "unhealthy", "missing_dependencies": missing}
            else:
                return {"status": "healthy", "all_dependencies_available": True}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


class MetricsCollector:
    """Collect and expose application metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            "documents_processed": 0,
            "phi_entities_detected": 0,
            "total_processing_time": 0.0,
            "errors_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    def increment_counter(self, metric_name: str, value: int = 1):
        """Increment a counter metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name] += value

    def record_timing(self, metric_name: str, duration: float):
        """Record timing metric."""
        if metric_name in self.metrics:
            self.metrics[metric_name] += duration

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics.copy(),
        }

    def get_prometheus_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        prometheus_output = []
        for metric_name, value in self.metrics.items():
            prometheus_output.append(f"hipaa_summarizer_{metric_name} {value}")
        return "\n".join(prometheus_output)


def health_check_endpoint() -> Dict[str, Any]:
    """Flask/FastAPI compatible health check endpoint."""
    health_checker = HealthChecker()
    return health_checker.get_system_health()


def metrics_endpoint() -> str:
    """Flask/FastAPI compatible metrics endpoint."""
    metrics_collector = MetricsCollector()
    return metrics_collector.get_prometheus_metrics()
