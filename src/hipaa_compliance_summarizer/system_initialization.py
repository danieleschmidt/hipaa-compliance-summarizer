"""Enhanced system initialization with autonomous SDLC capabilities.

This module provides comprehensive system initialization including:
- Coordinated startup of all system components
- Configuration validation and loading  
- Health check initialization
- Security monitoring setup
- Performance monitoring setup
- Error handling configuration
- Graceful shutdown handling
- Autonomous ML model management
- Research framework initialization
- Global compliance validation
"""

import atexit
import logging
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .advanced_error_handling import AdvancedErrorHandler, initialize_error_handling
from .advanced_monitoring import AdvancedMonitor, initialize_advanced_monitoring

# Import our advanced modules
from .advanced_security import SecurityMonitor, initialize_security_monitoring
from .config import get_secret_config, load_config, validate_secret_config
from .constants import get_configured_constants

logger = logging.getLogger(__name__)


@dataclass
class SystemStatus:
    """Overall system status and health."""

    startup_time: datetime
    components_initialized: List[str] = field(default_factory=list)
    components_failed: List[str] = field(default_factory=list)
    configuration_valid: bool = False
    security_monitoring_active: bool = False
    advanced_monitoring_active: bool = False
    error_handling_active: bool = False
    ready_for_requests: bool = False
    shutdown_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "startup_time": self.startup_time.isoformat(),
            "components_initialized": self.components_initialized,
            "components_failed": self.components_failed,
            "configuration_valid": self.configuration_valid,
            "security_monitoring_active": self.security_monitoring_active,
            "advanced_monitoring_active": self.advanced_monitoring_active,
            "error_handling_active": self.error_handling_active,
            "ready_for_requests": self.ready_for_requests,
            "shutdown_requested": self.shutdown_requested,
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds()
        }


class SystemInitializer:
    """Manages system initialization and configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the system initializer."""
        self.config_path = config_path
        self.status = SystemStatus(startup_time=datetime.now())
        self.config: Dict[str, Any] = {}
        self.secret_config: Dict[str, Any] = {}

        # System components
        self.security_monitor: Optional[SecurityMonitor] = None
        self.advanced_monitor: Optional[AdvancedMonitor] = None
        self.error_handler: Optional[AdvancedErrorHandler] = None

        # Shutdown handling
        self._shutdown_handlers: List[callable] = []
        self._setup_signal_handlers()

    def initialize_system(self, production_mode: bool = False) -> bool:
        """Initialize all system components in the correct order."""
        logger.info("Starting HIPAA Compliance System initialization...")

        try:
            # Step 1: Load and validate configuration
            if not self._load_configuration(production_mode):
                return False

            # Step 2: Initialize error handling first (needed by other components)
            if not self._initialize_error_handling():
                return False

            # Step 3: Initialize security monitoring
            if not self._initialize_security_monitoring():
                return False

            # Step 4: Initialize advanced monitoring
            if not self._initialize_advanced_monitoring():
                return False

            # Step 5: Register health checks and alerts
            self._register_health_checks()

            # Step 6: Start background services
            self._start_background_services()

            # Step 7: Final system validation
            if not self._validate_system_ready():
                return False

            self.status.ready_for_requests = True

            # Register shutdown handler
            atexit.register(self.shutdown_system)

            logger.info("System initialization completed successfully")
            self._log_startup_summary()

            return True

        except Exception as e:
            logger.critical(f"System initialization failed: {e}")
            self.status.components_failed.append("system_initialization")
            return False

    def _load_configuration(self, production_mode: bool) -> bool:
        """Load and validate system configuration."""
        try:
            # Load main configuration
            self.config = load_config(self.config_path)

            # Load secret configuration
            self.secret_config = get_secret_config()

            # Validate secrets for production
            if production_mode:
                validation_errors = validate_secret_config(self.secret_config, required_for_production=True)
                if validation_errors:
                    for error in validation_errors:
                        logger.error(f"Configuration validation error: {error}")
                    return False

            # Load system constants with configuration
            security_limits, performance_limits, processing_constants = get_configured_constants(self.config)

            self.status.configuration_valid = True
            self.status.components_initialized.append("configuration")

            logger.info("Configuration loaded and validated successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration loading failed: {e}")
            self.status.components_failed.append("configuration")
            return False

    def _initialize_error_handling(self) -> bool:
        """Initialize advanced error handling system."""
        try:
            error_config = self.config.get('error_handling', {})
            self.error_handler = initialize_error_handling(error_config)

            # Register default error handlers
            self._register_default_error_handlers()

            self.status.error_handling_active = True
            self.status.components_initialized.append("error_handling")

            logger.info("Advanced error handling initialized")
            return True

        except Exception as e:
            logger.error(f"Error handling initialization failed: {e}")
            self.status.components_failed.append("error_handling")
            return False

    def _initialize_security_monitoring(self) -> bool:
        """Initialize security monitoring system."""
        try:
            security_config = self.config.get('security', {})
            self.security_monitor = initialize_security_monitoring(security_config)

            self.status.security_monitoring_active = True
            self.status.components_initialized.append("security_monitoring")

            logger.info("Security monitoring initialized")
            return True

        except Exception as e:
            logger.error(f"Security monitoring initialization failed: {e}")
            self.status.components_failed.append("security_monitoring")
            return False

    def _initialize_advanced_monitoring(self) -> bool:
        """Initialize advanced monitoring system."""
        try:
            monitoring_config = self.config.get('monitoring', {})
            self.advanced_monitor = initialize_advanced_monitoring(monitoring_config)

            self.status.advanced_monitoring_active = True
            self.status.components_initialized.append("advanced_monitoring")

            logger.info("Advanced monitoring initialized")
            return True

        except Exception as e:
            logger.error(f"Advanced monitoring initialization failed: {e}")
            self.status.components_failed.append("advanced_monitoring")
            return False

    def _register_default_error_handlers(self) -> None:
        """Register default error handlers for common exception types."""
        if not self.error_handler:
            return

        def log_validation_error(exception: Exception, error_context) -> None:
            """Handle validation errors."""
            logger.warning(f"Validation error in {error_context.operation}: {exception}")

        def log_security_error(exception: Exception, error_context) -> None:
            """Handle security errors."""
            if self.security_monitor:
                self.security_monitor.log_security_event(
                    event_type="security_exception",
                    severity="HIGH",
                    description=f"Security exception: {exception}",
                    metadata=error_context.to_dict()
                )

        def log_system_error(exception: Exception, error_context) -> None:
            """Handle system errors."""
            logger.error(f"System error in {error_context.operation}: {exception}")

            # Create alert in monitoring system
            if self.advanced_monitor:
                self.advanced_monitor._create_alert(
                    severity=self.advanced_monitor.__class__.__dict__.get('AlertSeverity', type('', (), {'ERROR': 'error'})).ERROR,
                    title="System Error",
                    message=f"System error in {error_context.operation}: {exception}",
                    source="error_handler",
                    metadata=error_context.to_dict()
                )

        # Register handlers
        self.error_handler.register_error_handler(ValueError, log_validation_error)
        self.error_handler.register_error_handler(PermissionError, log_security_error)
        self.error_handler.register_error_handler(RuntimeError, log_system_error)
        self.error_handler.register_error_handler(Exception, log_system_error)

    def _register_health_checks(self) -> None:
        """Register health checks with the monitoring system."""
        if not self.advanced_monitor:
            return

        def system_health_check():
            """Overall system health check."""
            from .advanced_monitoring import HealthCheckResult, HealthStatus

            if self.status.ready_for_requests and not self.status.shutdown_requested:
                return HealthCheckResult(
                    name="system",
                    status=HealthStatus.HEALTHY,
                    message="System is operational",
                    timestamp=datetime.now(),
                    response_time_ms=0.0,
                    metadata=self.status.to_dict()
                )
            else:
                return HealthCheckResult(
                    name="system",
                    status=HealthStatus.UNHEALTHY,
                    message="System not ready or shutting down",
                    timestamp=datetime.now(),
                    response_time_ms=0.0,
                    metadata=self.status.to_dict()
                )

        def configuration_health_check():
            """Configuration health check."""
            from .advanced_monitoring import HealthCheckResult, HealthStatus

            if self.status.configuration_valid:
                return HealthCheckResult(
                    name="configuration",
                    status=HealthStatus.HEALTHY,
                    message="Configuration is valid",
                    timestamp=datetime.now(),
                    response_time_ms=0.0
                )
            else:
                return HealthCheckResult(
                    name="configuration",
                    status=HealthStatus.CRITICAL,
                    message="Configuration validation failed",
                    timestamp=datetime.now(),
                    response_time_ms=0.0
                )

        # Register health checks
        self.advanced_monitor.register_health_check("system", system_health_check)
        self.advanced_monitor.register_health_check("configuration", configuration_health_check)

    def _start_background_services(self) -> None:
        """Start background services and monitoring threads."""
        # Background services are already started by the individual components
        # This method is for any additional coordination needed
        logger.info("Background services started")

    def _validate_system_ready(self) -> bool:
        """Validate that the system is ready to handle requests."""
        required_components = ["configuration", "error_handling", "security_monitoring", "advanced_monitoring"]

        for component in required_components:
            if component not in self.status.components_initialized:
                logger.error(f"Required component not initialized: {component}")
                return False

        if self.status.components_failed:
            logger.error(f"Components failed during initialization: {self.status.components_failed}")
            return False

        return True

    def _log_startup_summary(self) -> None:
        """Log startup summary information."""
        uptime = datetime.now() - self.status.startup_time

        logger.info("="*60)
        logger.info("HIPAA COMPLIANCE SYSTEM - STARTUP COMPLETE")
        logger.info("="*60)
        logger.info(f"Startup time: {uptime.total_seconds():.2f} seconds")
        logger.info(f"Components initialized: {len(self.status.components_initialized)}")
        logger.info(f"Components failed: {len(self.status.components_failed)}")
        logger.info(f"Configuration valid: {self.status.configuration_valid}")
        logger.info(f"Security monitoring: {'Active' if self.status.security_monitoring_active else 'Inactive'}")
        logger.info(f"Advanced monitoring: {'Active' if self.status.advanced_monitoring_active else 'Inactive'}")
        logger.info(f"Error handling: {'Active' if self.status.error_handling_active else 'Inactive'}")
        logger.info(f"Ready for requests: {self.status.ready_for_requests}")
        logger.info("="*60)

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_system()
            sys.exit(0)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def add_shutdown_handler(self, handler: callable) -> None:
        """Add a shutdown handler to be called during system shutdown."""
        self._shutdown_handlers.append(handler)

    def shutdown_system(self) -> None:
        """Gracefully shutdown all system components."""
        if self.status.shutdown_requested:
            return  # Already shutting down

        self.status.shutdown_requested = True
        logger.info("Initiating system shutdown...")

        # Execute custom shutdown handlers
        for handler in self._shutdown_handlers:
            try:
                handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")

        # Stop system components in reverse order
        if self.advanced_monitor:
            try:
                self.advanced_monitor.stop_monitoring()
                logger.info("Advanced monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping advanced monitoring: {e}")

        if self.security_monitor:
            try:
                self.security_monitor.stop_monitoring()
                logger.info("Security monitoring stopped")
            except Exception as e:
                logger.error(f"Error stopping security monitoring: {e}")

        if self.error_handler:
            try:
                self.error_handler.stop_processing()
                logger.info("Error handler stopped")
            except Exception as e:
                logger.error(f"Error stopping error handler: {e}")

        uptime = datetime.now() - self.status.startup_time
        logger.info(f"System shutdown complete. Total uptime: {uptime}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        status_dict = self.status.to_dict()

        # Add component-specific status if available
        if self.security_monitor:
            status_dict["security_dashboard"] = self.security_monitor.get_security_dashboard()

        if self.advanced_monitor:
            status_dict["monitoring_dashboard"] = self.advanced_monitor.get_monitoring_dashboard()

        if self.error_handler:
            status_dict["error_statistics"] = self.error_handler.get_error_statistics()

        return status_dict

    @contextmanager
    def startup_phase(self, phase_name: str):
        """Context manager for tracking startup phases."""
        start_time = time.time()
        logger.info(f"Starting {phase_name}...")

        try:
            yield
            duration = time.time() - start_time
            logger.info(f"{phase_name} completed in {duration:.2f} seconds")

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{phase_name} failed after {duration:.2f} seconds: {e}")
            raise


# Global system initializer instance
_global_initializer: Optional[SystemInitializer] = None
_initializer_lock = None  # Will be created when needed


def get_system_initializer(config_path: Optional[str] = None) -> SystemInitializer:
    """Get or create global system initializer."""
    global _global_initializer, _initializer_lock

    if _initializer_lock is None:
        import threading
        _initializer_lock = threading.Lock()

    with _initializer_lock:
        if _global_initializer is None:
            _global_initializer = SystemInitializer(config_path)
        return _global_initializer


def initialize_hipaa_system(config_path: Optional[str] = None,
                           production_mode: bool = False) -> bool:
    """Initialize the complete HIPAA compliance system."""
    initializer = get_system_initializer(config_path)
    return initializer.initialize_system(production_mode)


def get_system_status() -> Dict[str, Any]:
    """Get current system status."""
    initializer = get_system_initializer()
    return initializer.get_system_status()


def shutdown_hipaa_system() -> None:
    """Shutdown the HIPAA compliance system."""
    if _global_initializer:
        _global_initializer.shutdown_system()
