"""Health check monitoring for HIPAA compliance system."""

import os
import time
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import requests
import psutil


logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health check status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    
    name: str
    status: HealthStatus
    response_time_ms: float
    message: str
    details: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "response_time_ms": self.response_time_ms,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }


class BaseHealthCheck:
    """Base class for health checks."""
    
    def __init__(self, name: str, timeout_seconds: float = 5.0):
        """Initialize health check.
        
        Args:
            name: Name of the health check
            timeout_seconds: Timeout for the health check
        """
        self.name = name
        self.timeout_seconds = timeout_seconds
    
    async def execute(self) -> HealthCheckResult:
        """Execute the health check."""
        start_time = time.time()
        
        try:
            # Implement timeout
            result = await asyncio.wait_for(
                self._check_health(),
                timeout=self.timeout_seconds
            )
            
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                name=self.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                response_time_ms=response_time,
                message=result.get("message", ""),
                details=result.get("details", {}),
                timestamp=datetime.utcnow()
            )
            
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check timed out after {self.timeout_seconds}s",
                details={"timeout": True},
                timestamp=datetime.utcnow()
            )
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                name=self.name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=response_time,
                message=f"Health check failed: {str(e)}",
                details={"error": str(e)},
                timestamp=datetime.utcnow()
            )
    
    async def _check_health(self) -> Dict[str, Any]:
        """Override this method to implement specific health check logic."""
        raise NotImplementedError


class DatabaseHealthCheck(BaseHealthCheck):
    """Health check for database connectivity."""
    
    def __init__(self, database_manager):
        """Initialize database health check.
        
        Args:
            database_manager: Database manager instance
        """
        super().__init__("database", timeout_seconds=3.0)
        self.database_manager = database_manager
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check database health."""
        try:
            # Simple query to test connectivity
            connection = self.database_manager.get_connection()
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if result:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": "Database connection successful",
                    "details": {"query_result": result[0]}
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": "Database query returned no result",
                    "details": {}
                }
                
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"Database connection failed: {str(e)}",
                "details": {"error": str(e)}
            }


class SystemResourcesHealthCheck(BaseHealthCheck):
    """Health check for system resources (CPU, memory, disk)."""
    
    def __init__(self):
        """Initialize system resources health check."""
        super().__init__("system_resources", timeout_seconds=2.0)
        
        # Thresholds
        self.cpu_warning_threshold = float(os.getenv("HEALTH_CPU_WARNING", "80.0"))
        self.cpu_critical_threshold = float(os.getenv("HEALTH_CPU_CRITICAL", "95.0"))
        self.memory_warning_threshold = float(os.getenv("HEALTH_MEMORY_WARNING", "80.0"))
        self.memory_critical_threshold = float(os.getenv("HEALTH_MEMORY_CRITICAL", "95.0"))
        self.disk_warning_threshold = float(os.getenv("HEALTH_DISK_WARNING", "80.0"))
        self.disk_critical_threshold = float(os.getenv("HEALTH_DISK_CRITICAL", "95.0"))
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check system resource health."""
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        status = HealthStatus.HEALTHY
        issues = []
        
        # Check CPU
        if cpu_percent >= self.cpu_critical_threshold:
            status = HealthStatus.UNHEALTHY
            issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
        elif cpu_percent >= self.cpu_warning_threshold:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"CPU usage high: {cpu_percent:.1f}%")
        
        # Check memory
        if memory.percent >= self.memory_critical_threshold:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Memory usage critical: {memory.percent:.1f}%")
        elif memory.percent >= self.memory_warning_threshold:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Memory usage high: {memory.percent:.1f}%")
        
        # Check disk
        if disk.percent >= self.disk_critical_threshold:
            status = HealthStatus.UNHEALTHY
            issues.append(f"Disk usage critical: {disk.percent:.1f}%")
        elif disk.percent >= self.disk_warning_threshold:
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED
            issues.append(f"Disk usage high: {disk.percent:.1f}%")
        
        message = "System resources healthy" if not issues else "; ".join(issues)
        
        return {
            "status": status,
            "message": message,
            "details": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        }


class ExternalServiceHealthCheck(BaseHealthCheck):
    """Health check for external service dependencies."""
    
    def __init__(self, service_name: str, url: str, expected_status: int = 200):
        """Initialize external service health check.
        
        Args:
            service_name: Name of the external service
            url: Health check URL for the service
            expected_status: Expected HTTP status code
        """
        super().__init__(f"external_service_{service_name}", timeout_seconds=5.0)
        self.service_name = service_name
        self.url = url
        self.expected_status = expected_status
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check external service health."""
        try:
            response = requests.get(self.url, timeout=self.timeout_seconds)
            
            if response.status_code == self.expected_status:
                return {
                    "status": HealthStatus.HEALTHY,
                    "message": f"{self.service_name} service is healthy",
                    "details": {
                        "status_code": response.status_code,
                        "response_time_ms": response.elapsed.total_seconds() * 1000
                    }
                }
            else:
                return {
                    "status": HealthStatus.UNHEALTHY,
                    "message": f"{self.service_name} returned unexpected status: {response.status_code}",
                    "details": {
                        "status_code": response.status_code,
                        "expected_status": self.expected_status
                    }
                }
                
        except requests.RequestException as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"{self.service_name} service unreachable: {str(e)}",
                "details": {"error": str(e)}
            }


class PHIDetectionServiceHealthCheck(BaseHealthCheck):
    """Health check for PHI detection service."""
    
    def __init__(self, phi_detection_service):
        """Initialize PHI detection service health check.
        
        Args:
            phi_detection_service: PHI detection service instance
        """
        super().__init__("phi_detection_service", timeout_seconds=3.0)
        self.phi_service = phi_detection_service
    
    async def _check_health(self) -> Dict[str, Any]:
        """Check PHI detection service health."""
        try:
            # Test with a simple text
            test_text = "Test document for health check"
            result = self.phi_service.detect_phi_entities(
                text=test_text,
                detection_method="pattern",
                confidence_threshold=0.8
            )
            
            return {
                "status": HealthStatus.HEALTHY,
                "message": "PHI detection service is operational",
                "details": {
                    "test_entities_found": len(result.entities),
                    "detection_method": "pattern"
                }
            }
            
        except Exception as e:
            return {
                "status": HealthStatus.UNHEALTHY,
                "message": f"PHI detection service failed: {str(e)}",
                "details": {"error": str(e)}
            }


class HealthCheckManager:
    """Manager for coordinating health checks."""
    
    def __init__(self):
        """Initialize health check manager."""
        self.health_checks: List[BaseHealthCheck] = []
        self.check_interval_seconds = int(os.getenv("HEALTH_CHECK_INTERVAL", "30"))
        self.last_results: Dict[str, HealthCheckResult] = {}
        self._running = False
    
    def register_health_check(self, health_check: BaseHealthCheck):
        """Register a health check.
        
        Args:
            health_check: Health check instance to register
        """
        self.health_checks.append(health_check)
        logger.info(f"Registered health check: {health_check.name}")
    
    async def execute_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Execute all registered health checks."""
        results = {}
        
        # Execute all checks concurrently
        tasks = [check.execute() for check in self.health_checks]
        check_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(check_results):
            check_name = self.health_checks[i].name
            
            if isinstance(result, Exception):
                # Handle exceptions from health checks
                results[check_name] = HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    response_time_ms=0.0,
                    message=f"Health check failed with exception: {str(result)}",
                    details={"exception": str(result)},
                    timestamp=datetime.utcnow()
                )
            else:
                results[check_name] = result
        
        # Update last results
        self.last_results = results
        
        return results
    
    def get_overall_health(self, results: Dict[str, HealthCheckResult] = None) -> Dict[str, Any]:
        """Get overall system health status.
        
        Args:
            results: Optional health check results (uses last results if not provided)
            
        Returns:
            Overall health status summary
        """
        if results is None:
            results = self.last_results
        
        if not results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks executed",
                "timestamp": datetime.utcnow().isoformat(),
                "checks": {}
            }
        
        # Determine overall status
        statuses = [result.status for result in results.values()]
        
        if any(status == HealthStatus.UNHEALTHY for status in statuses):
            overall_status = HealthStatus.UNHEALTHY
            message = "One or more critical health checks failed"
        elif any(status == HealthStatus.DEGRADED for status in statuses):
            overall_status = HealthStatus.DEGRADED
            message = "System performance is degraded"
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            overall_status = HealthStatus.HEALTHY
            message = "All systems operational"
        else:
            overall_status = HealthStatus.UNKNOWN
            message = "Health status unclear"
        
        # Calculate summary statistics
        response_times = [r.response_time_ms for r in results.values()]
        
        return {
            "status": overall_status.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_checks": len(results),
                "healthy": sum(1 for r in results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in results.values() if r.status == HealthStatus.UNHEALTHY),
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0
            },
            "checks": {name: result.to_dict() for name, result in results.items()}
        }
    
    async def start_continuous_monitoring(self):
        """Start continuous health monitoring."""
        self._running = True
        logger.info(f"Starting health check monitoring (interval: {self.check_interval_seconds}s)")
        
        while self._running:
            try:
                await self.execute_all_checks()
                await asyncio.sleep(self.check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in health check monitoring: {e}")
                await asyncio.sleep(5)  # Short delay before retrying
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        self._running = False
        logger.info("Health check monitoring stopped")
    
    def setup_default_checks(self, database_manager=None, phi_service=None):
        """Setup default health checks.
        
        Args:
            database_manager: Optional database manager
            phi_service: Optional PHI detection service
        """
        # Always add system resources check
        self.register_health_check(SystemResourcesHealthCheck())
        
        # Add database check if manager provided
        if database_manager:
            self.register_health_check(DatabaseHealthCheck(database_manager))
        
        # Add PHI service check if provided
        if phi_service:
            self.register_health_check(PHIDetectionServiceHealthCheck(phi_service))
        
        # Add external service checks from environment
        external_services = {
            "elasticsearch": os.getenv("ELASTICSEARCH_HEALTH_URL"),
            "redis": os.getenv("REDIS_HEALTH_URL"),
            "ml_service": os.getenv("ML_SERVICE_HEALTH_URL")
        }
        
        for service_name, health_url in external_services.items():
            if health_url:
                self.register_health_check(
                    ExternalServiceHealthCheck(service_name, health_url)
                )
        
        logger.info(f"Setup {len(self.health_checks)} default health checks")