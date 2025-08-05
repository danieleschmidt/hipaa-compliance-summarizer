"""Tests for advanced HIPAA compliance features."""

import pytest
import tempfile
import time
from datetime import datetime, timedelta
from hipaa_compliance_summarizer.error_handling import (
    HIPAAError, ValidationError, ErrorCategory, ErrorSeverity, ErrorContext,
    global_error_handler, handle_errors
)
from hipaa_compliance_summarizer.resilience import (
    ResilientExecutor, RetryConfig, RetryPolicy, resilient_operation
)
from hipaa_compliance_summarizer.performance import (
    PerformanceOptimizer, ConcurrentProcessor, AdaptiveCache, performance_monitor
)
from hipaa_compliance_summarizer.scaling import (
    AutoScaler, WorkerPool, ResourceType, ScalingRule
)


class TestErrorHandling:
    """Test advanced error handling features."""
    
    def test_hipaa_error_creation(self):
        """Test HIPAA error creation and serialization."""
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            timestamp=datetime.utcnow(),
            source="test",
            operation="test_operation",
            user_id="test_user"
        )
        
        error = ValidationError(
            message="Test validation error",
            error_code="TEST_ERROR",
            context=context
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "Test validation error"
        assert error_dict["category"] == "validation"
        assert error_dict["severity"] == "high"
        assert error_dict["user_id"] == "test_user"
        
    def test_error_handler_registration(self):
        """Test error handler callback registration."""
        callback_called = False
        
        def test_callback(error):
            nonlocal callback_called
            callback_called = True
            
        global_error_handler.register_error_callback(ErrorCategory.VALIDATION, test_callback)
        
        # Trigger error
        context = ErrorContext(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            timestamp=datetime.utcnow(),
            source="test",
            operation="test"
        )
        
        error = ValidationError("Test", "TEST", context)
        global_error_handler.handle_error(error, context)
        
        assert callback_called
        
    def test_handle_errors_decorator(self):
        """Test error handling decorator."""
        @handle_errors(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.LOW,
            source="test",
            operation="test_func"
        )
        def test_function():
            raise ValueError("Test error")
            
        with pytest.raises(ValueError):
            test_function()
            
        # Check error was recorded
        stats = global_error_handler.get_error_statistics()
        assert stats["total_errors"] > 0


class TestResilience:
    """Test resilience and retry mechanisms."""
    
    def test_retry_config_creation(self):
        """Test retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            policy=RetryPolicy.EXPONENTIAL_BACKOFF,
            base_delay_seconds=2.0
        )
        
        assert config.max_attempts == 5
        assert config.policy == RetryPolicy.EXPONENTIAL_BACKOFF
        assert config.base_delay_seconds == 2.0
        
    def test_resilient_executor(self):
        """Test resilient execution with retry."""
        executor = ResilientExecutor()
        
        call_count = 0
        
        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
            
        # Configure retry
        config = RetryConfig(max_attempts=3, policy=RetryPolicy.IMMEDIATE)
        executor.register_retry_config("test_op", config)
        
        result = executor.execute_with_retry(failing_function, "test_op")
        
        assert result == "success"
        assert call_count == 3
        
    def test_resilient_operation_decorator(self):
        """Test resilient operation decorator."""
        call_count = 0
        
        @resilient_operation("test_decorated", RetryConfig(max_attempts=2, policy=RetryPolicy.IMMEDIATE))
        def test_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            return "success"
            
        result = test_function()
        assert result == "success"
        assert call_count == 2


class TestPerformance:
    """Test performance optimization features."""
    
    def test_performance_optimizer(self):
        """Test performance tracking."""
        optimizer = PerformanceOptimizer()
        
        metrics = optimizer.start_performance_tracking("test_operation")
        
        # Simulate some work
        time.sleep(0.1)
        metrics.items_processed = 10
        
        optimizer.finish_performance_tracking(metrics)
        
        assert metrics.operation_name == "test_operation"
        assert metrics.duration_seconds > 0
        assert metrics.throughput_ops_per_second > 0
        assert len(optimizer.metrics_history) > 0
        
    def test_performance_monitor_decorator(self):
        """Test performance monitoring decorator."""
        @performance_monitor("test_decorated_op")
        def test_function():
            time.sleep(0.05)
            return "result"
            
        result = test_function()
        assert result == "result"
        
        # Check metrics were recorded
        from hipaa_compliance_summarizer.performance import performance_optimizer
        summary = performance_optimizer.get_performance_summary(hours=1)
        assert "test_decorated_op" in summary.get("by_operation", {})
        
    def test_adaptive_cache(self):
        """Test adaptive caching."""
        cache = AdaptiveCache(max_size=5)
        
        # Test cache operations
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("nonexistent") is None
        
        # Test cache statistics
        stats = cache.get_statistics()
        assert stats["cache_size"] == 2
        assert stats["hit_count"] == 2
        assert stats["miss_count"] == 1
        
    def test_concurrent_processor(self):
        """Test concurrent processing."""
        def square_function(x):
            return x * x
            
        with ConcurrentProcessor(max_workers=2, chunk_size=2) as processor:
            numbers = list(range(10))
            results = processor.process_batch(numbers, square_function)
            
            expected = [x * x for x in numbers]
            assert results == expected


class TestScaling:
    """Test auto-scaling features."""
    
    def test_scaling_rule_creation(self):
        """Test scaling rule configuration."""
        rule = ScalingRule(
            resource_type=ResourceType.CPU,
            scale_up_threshold=80.0,
            scale_down_threshold=20.0,
            scale_up_amount=2,
            cooldown_seconds=300
        )
        
        assert rule.resource_type == ResourceType.CPU
        assert rule.should_scale_up(85.0)
        assert not rule.should_scale_up(75.0)
        assert rule.should_scale_down(15.0)
        assert not rule.should_scale_down(25.0)
        
    def test_worker_pool(self):
        """Test worker pool functionality."""
        pool = WorkerPool(initial_workers=2, max_workers=5, min_workers=1)
        
        try:
            pool.start()
            
            # Submit some tasks
            results = []
            
            def test_task(value):
                results.append(value * 2)
                
            for i in range(5):
                pool.submit_task(test_task, i)
                
            # Give workers time to process
            time.sleep(0.5)
            
            stats = pool.get_statistics()
            assert stats["active_workers"] >= 1
            assert stats["tasks_completed"] >= 0
            
        finally:
            pool.stop()
            
    def test_auto_scaler(self):
        """Test auto-scaler configuration."""
        scaler = AutoScaler()
        
        # Register scaling rule
        rule = ScalingRule(
            resource_type=ResourceType.CPU,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0
        )
        scaler.register_scaling_rule(rule)
        
        # Register callback
        callback_called = False
        
        def test_callback(action, amount):
            nonlocal callback_called
            callback_called = True
            
        scaler.register_scaling_callback(ResourceType.CPU, test_callback)
        
        status = scaler.get_scaling_status()
        assert status["total_rules"] == 1
        assert not status["monitoring_active"]


class TestIntegration:
    """Test integration of all advanced features."""
    
    def test_error_handling_with_performance_monitoring(self):
        """Test error handling integrated with performance monitoring."""
        @performance_monitor("error_test_op")
        @handle_errors(
            category=ErrorCategory.PROCESSING,
            severity=ErrorSeverity.LOW,
            source="integration_test",
            operation="error_perf_test"
        )
        def test_function_with_error():
            time.sleep(0.05)
            raise ValueError("Test integration error")
            
        with pytest.raises(ValueError):
            test_function_with_error()
            
        # Check both systems recorded the event
        error_stats = global_error_handler.get_error_statistics()
        assert error_stats["total_errors"] > 0
        
        from hipaa_compliance_summarizer.performance import performance_optimizer
        perf_summary = performance_optimizer.get_performance_summary(hours=1)
        assert "error_test_op" in perf_summary.get("by_operation", {})
        
    def test_resilient_performance_monitoring(self):
        """Test resilient operations with performance monitoring."""
        call_count = 0
        
        @performance_monitor("resilient_perf_op")
        @resilient_operation("resilient_test", RetryConfig(max_attempts=2, policy=RetryPolicy.IMMEDIATE))
        def resilient_function():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First attempt fails")
            time.sleep(0.05)
            return "success"
            
        result = resilient_function()
        assert result == "success"
        assert call_count == 2
        
        # Check performance was monitored
        from hipaa_compliance_summarizer.performance import performance_optimizer
        perf_summary = performance_optimizer.get_performance_summary(hours=1)
        assert "resilient_perf_op" in perf_summary.get("by_operation", {})


def test_system_health_check():
    """Test overall system health."""
    # Test that all major components can be imported and initialized
    from hipaa_compliance_summarizer import (
        HIPAAProcessor, BatchProcessor, 
        HIPAAError, resilient_operation, performance_monitor,
        initialize_scaling_infrastructure, get_scaling_status
    )
    
    # Test basic functionality
    processor = HIPAAProcessor()
    assert processor is not None
    
    batch_processor = BatchProcessor()
    assert batch_processor is not None
    
    # Test scaling infrastructure
    initialize_scaling_infrastructure()
    status = get_scaling_status()
    assert "worker_pool" in status
    assert "auto_scaler" in status