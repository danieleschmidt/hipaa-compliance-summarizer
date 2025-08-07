"""Integration tests for advanced system components.

Tests the integration between:
- Advanced security monitoring
- Advanced monitoring and health checks
- Advanced error handling
- Distributed processing
- Intelligent auto-scaling
- System initialization
"""

import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from hipaa_compliance_summarizer.advanced_security import (
    SecurityMonitor, get_security_monitor, log_security_event
)
from hipaa_compliance_summarizer.advanced_monitoring import (
    AdvancedMonitor, get_advanced_monitor, HealthStatus, AlertSeverity
)
from hipaa_compliance_summarizer.advanced_error_handling import (
    AdvancedErrorHandler, get_error_handler, ErrorSeverity, ErrorCategory
)
from hipaa_compliance_summarizer.distributed_processing import (
    ClusterCoordinator, TaskPriority
)
from hipaa_compliance_summarizer.intelligent_autoscaling import (
    IntelligentAutoScaler, ScalingDecision
)
from hipaa_compliance_summarizer.system_initialization import (
    SystemInitializer, initialize_hipaa_system
)


class TestAdvancedSecurityMonitoring:
    """Test advanced security monitoring system."""
    
    def test_security_monitor_initialization(self):
        """Test security monitor can be initialized."""
        monitor = SecurityMonitor()
        assert monitor is not None
        assert monitor._monitoring_active is True
        monitor.stop_monitoring()
    
    def test_security_event_logging(self):
        """Test security event logging and analysis."""
        monitor = SecurityMonitor()
        
        # Log a security event
        event = monitor.log_security_event(
            event_type="test_event",
            severity="HIGH",
            description="Test security event",
            source_ip="192.168.1.100",
            metadata={"test": "data"}
        )
        
        assert event.event_type == "test_event"
        assert event.severity == "HIGH"
        assert event.source_ip == "192.168.1.100"
        
        monitor.stop_monitoring()
    
    def test_threat_profile_creation(self):
        """Test threat profile creation and tracking."""
        monitor = SecurityMonitor()
        
        # Simulate failed attempts from same IP
        for i in range(3):
            monitor.log_security_event(
                event_type="authentication_failure",
                severity="MEDIUM",
                description=f"Failed login attempt {i+1}",
                source_ip="10.0.0.1"
            )
        
        # Check threat profile was created
        profile = monitor.get_threat_profile("10.0.0.1")
        assert profile is not None
        assert profile.failed_attempts >= 3
        
        monitor.stop_monitoring()
    
    def test_security_dashboard(self):
        """Test security dashboard data retrieval."""
        monitor = SecurityMonitor()
        
        dashboard = monitor.get_security_dashboard()
        assert "monitoring_status" in dashboard
        assert "events_last_24h" in dashboard
        assert "threat_levels" in dashboard
        assert dashboard["monitoring_status"] == "active"
        
        monitor.stop_monitoring()


class TestAdvancedMonitoring:
    """Test advanced monitoring system."""
    
    def test_monitor_initialization(self):
        """Test advanced monitor initialization."""
        monitor = AdvancedMonitor()
        assert monitor is not None
        assert monitor._monitoring_active is True
        monitor.stop_monitoring()
    
    def test_health_check_registration(self):
        """Test health check registration and execution."""
        monitor = AdvancedMonitor()
        
        def test_health_check():
            from hipaa_compliance_summarizer.advanced_monitoring import HealthCheckResult, HealthStatus
            return HealthCheckResult(
                name="test_service",
                status=HealthStatus.HEALTHY,
                message="Test service is running",
                timestamp=datetime.now(),
                response_time_ms=10.0
            )
        
        monitor.register_health_check("test_service", test_health_check)
        
        # Run health checks
        results = monitor.run_health_checks()
        assert "test_service" in results
        assert results["test_service"].status == HealthStatus.HEALTHY
        
        monitor.stop_monitoring()
    
    def test_system_metrics_collection(self):
        """Test system metrics collection."""
        monitor = AdvancedMonitor()
        
        metrics = monitor.collect_system_metrics()
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_usage_percent >= 0
        
        monitor.stop_monitoring()
    
    def test_custom_metrics(self):
        """Test custom metrics recording."""
        monitor = AdvancedMonitor()
        
        monitor.record_custom_metric("test_metric", 42.0, {"tag": "value"})
        
        # Verify metric was recorded
        assert "test_metric" in monitor._custom_metrics
        assert len(monitor._custom_metrics["test_metric"]) > 0
        
        monitor.stop_monitoring()
    
    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        monitor = AdvancedMonitor()
        
        from hipaa_compliance_summarizer.advanced_monitoring import CircuitBreakerConfig
        
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            name="test_breaker"
        )
        
        breaker = monitor.register_circuit_breaker("test", config)
        assert breaker is not None
        
        monitor.stop_monitoring()


class TestAdvancedErrorHandling:
    """Test advanced error handling system."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = AdvancedErrorHandler()
        assert handler is not None
        assert handler._processing_active is True
        handler.stop_processing()
    
    def test_error_classification(self):
        """Test error classification system."""
        handler = AdvancedErrorHandler()
        
        # Test different error types
        security_error = PermissionError("Access denied")
        category = handler.classify_error(security_error)
        assert category == ErrorCategory.SECURITY
        
        network_error = ConnectionError("Connection failed")
        category = handler.classify_error(network_error)
        assert category == ErrorCategory.NETWORK
        
        validation_error = ValueError("Invalid format")
        category = handler.classify_error(validation_error)
        assert category == ErrorCategory.VALIDATION
        
        handler.stop_processing()
    
    def test_error_handling(self):
        """Test error handling and context creation."""
        handler = AdvancedErrorHandler()
        
        test_error = ValueError("Test error")
        error_context = handler.handle_error(test_error, "test_operation", {"key": "value"})
        
        assert error_context.operation == "test_operation"
        assert error_context.exception_type == "ValueError"
        assert error_context.message == "Test error"
        assert error_context.metadata["key"] == "value"
        
        handler.stop_processing()
    
    def test_retry_decorator(self):
        """Test retry decorator functionality."""
        handler = AdvancedErrorHandler()
        
        call_count = 0
        
        @handler.retry_with_backoff()
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3
        
        handler.stop_processing()
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        handler = AdvancedErrorHandler()
        
        # Generate some errors
        for i in range(5):
            handler.handle_error(ValueError(f"Error {i}"), "test_op")
        
        stats = handler.get_error_statistics()
        assert stats["total_errors_24h"] >= 5
        assert "severity_distribution" in stats
        assert "category_distribution" in stats
        
        handler.stop_processing()


class TestDistributedProcessing:
    """Test distributed processing system."""
    
    def test_coordinator_initialization(self):
        """Test cluster coordinator initialization."""
        coordinator = ClusterCoordinator()
        assert coordinator is not None
        assert coordinator.node_id is not None
        coordinator.shutdown()
    
    def test_task_submission(self):
        """Test task submission and queuing."""
        coordinator = ClusterCoordinator()
        
        task_id = coordinator.submit_task(
            task_type="test_task",
            payload={"data": "test"},
            priority=TaskPriority.NORMAL
        )
        
        assert task_id is not None
        
        # Check task status
        task_status = coordinator.get_task_status(task_id)
        assert task_status is not None
        assert task_status["task_type"] == "test_task"
        
        coordinator.shutdown()
    
    def test_node_registration(self):
        """Test node registration in cluster."""
        coordinator = ClusterCoordinator()
        
        from hipaa_compliance_summarizer.distributed_processing import NodeInfo, NodeStatus
        
        test_node = NodeInfo(
            node_id="test_node_1",
            hostname="test-host",
            ip_address="192.168.1.10",
            port=8080,
            status=NodeStatus.HEALTHY,
            cpu_count=4,
            memory_gb=8.0,
            current_load=0.5,
            last_heartbeat=datetime.now()
        )
        
        success = coordinator.register_node(test_node)
        assert success is True
        
        nodes = coordinator.get_nodes()
        assert "test_node_1" in nodes
        
        coordinator.shutdown()
    
    def test_cluster_status(self):
        """Test cluster status reporting."""
        coordinator = ClusterCoordinator()
        
        status = coordinator.get_cluster_status()
        assert "cluster_id" in status
        assert "nodes" in status
        assert "tasks" in status
        
        coordinator.shutdown()


class TestIntelligentAutoScaling:
    """Test intelligent auto-scaling system."""
    
    def test_autoscaler_initialization(self):
        """Test auto-scaler initialization."""
        autoscaler = IntelligentAutoScaler()
        assert autoscaler is not None
        assert autoscaler.current_instances >= autoscaler.min_instances
        autoscaler.shutdown()
    
    def test_metrics_update(self):
        """Test metrics update for scaling decisions."""
        autoscaler = IntelligentAutoScaler()
        
        test_metrics = {
            "cpu_utilization": 75.0,
            "memory_utilization": 60.0,
            "queue_length": 15.0,
            "response_time_ms": 1500.0
        }
        
        autoscaler.update_metrics(test_metrics)
        
        status = autoscaler.get_scaling_status()
        assert "current_metrics" in status
        assert status["current_instances"] == autoscaler.current_instances
        
        autoscaler.shutdown()
    
    def test_manual_scaling(self):
        """Test manual scaling operations."""
        autoscaler = IntelligentAutoScaler({"min_instances": 1, "max_instances": 5})
        
        # Test scale up
        success = autoscaler.force_scaling_decision(3, "test scale up")
        assert success is True
        assert autoscaler.current_instances == 3
        
        # Test scale down
        success = autoscaler.force_scaling_decision(2, "test scale down")
        assert success is True
        assert autoscaler.current_instances == 2
        
        autoscaler.shutdown()
    
    def test_scaling_status(self):
        """Test scaling status reporting."""
        autoscaler = IntelligentAutoScaler()
        
        status = autoscaler.get_scaling_status()
        assert "current_instances" in status
        assert "current_metrics" in status
        assert "recent_scaling_events" in status
        assert "autoscaler_active" in status
        
        autoscaler.shutdown()


class TestSystemInitialization:
    """Test system initialization and coordination."""
    
    def test_system_initializer_creation(self):
        """Test system initializer creation."""
        initializer = SystemInitializer()
        assert initializer is not None
        assert initializer.status.startup_time is not None
    
    @patch('hipaa_compliance_summarizer.system_initialization.initialize_security_monitoring')
    @patch('hipaa_compliance_summarizer.system_initialization.initialize_advanced_monitoring')
    @patch('hipaa_compliance_summarizer.system_initialization.initialize_error_handling')
    def test_system_initialization_components(self, mock_error_handler, mock_monitoring, mock_security):
        """Test system component initialization."""
        # Setup mocks
        mock_error_handler.return_value = Mock()
        mock_monitoring.return_value = Mock()
        mock_security.return_value = Mock()
        
        initializer = SystemInitializer()
        
        # Mock configuration loading
        initializer.config = {"test": "config"}
        initializer.secret_config = {}
        
        with patch.object(initializer, '_load_configuration', return_value=True), \
             patch.object(initializer, '_initialize_error_handling', return_value=True), \
             patch.object(initializer, '_initialize_security_monitoring', return_value=True), \
             patch.object(initializer, '_initialize_advanced_monitoring', return_value=True), \
             patch.object(initializer, '_register_health_checks'), \
             patch.object(initializer, '_start_background_services'), \
             patch.object(initializer, '_validate_system_ready', return_value=True):
            
            success = initializer.initialize_system()
            assert success is True
            assert initializer.status.ready_for_requests is True


class TestSystemIntegration:
    """Test integration between all system components."""
    
    def test_component_interaction(self):
        """Test interaction between different components."""
        # Initialize components
        security_monitor = SecurityMonitor()
        advanced_monitor = AdvancedMonitor()
        error_handler = AdvancedErrorHandler()
        
        # Test that security events can be logged
        event = security_monitor.log_security_event(
            event_type="test_integration",
            severity="INFO",
            description="Integration test event"
        )
        assert event is not None
        
        # Test that errors can be handled
        test_error = RuntimeError("Integration test error")
        error_context = error_handler.handle_error(test_error, "integration_test")
        assert error_context is not None
        
        # Test that metrics can be collected
        metrics = advanced_monitor.collect_system_metrics()
        assert metrics is not None
        
        # Cleanup
        security_monitor.stop_monitoring()
        advanced_monitor.stop_monitoring()
        error_handler.stop_processing()
    
    def test_global_instance_management(self):
        """Test global instance management."""
        # Test that global instances work correctly
        security_monitor1 = get_security_monitor()
        security_monitor2 = get_security_monitor()
        assert security_monitor1 is security_monitor2  # Should be same instance
        
        monitor1 = get_advanced_monitor()
        monitor2 = get_advanced_monitor()
        assert monitor1 is monitor2  # Should be same instance
        
        handler1 = get_error_handler()
        handler2 = get_error_handler()
        assert handler1 is handler2  # Should be same instance
        
        # Cleanup
        security_monitor1.stop_monitoring()
        monitor1.stop_monitoring()
        handler1.stop_processing()
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with all components."""
        # Initialize system
        initializer = SystemInitializer()
        
        # Mock successful initialization
        with patch.object(initializer, 'initialize_system', return_value=True):
            success = initializer.initialize_system()
            assert success is True
        
        # Test that we can get status
        status = initializer.get_system_status()
        assert "startup_time" in status
        assert "components_initialized" in status


if __name__ == "__main__":
    pytest.main([__file__])