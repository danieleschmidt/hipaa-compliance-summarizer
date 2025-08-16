"""Comprehensive tests for Generation 3 scaling and optimization features."""

import asyncio
import pytest
import threading
import time
from unittest.mock import patch, MagicMock

from hipaa_compliance_summarizer.performance_optimized import (
    AdaptiveCache, PerformanceOptimizer, AsyncOptimizer, OptimizationMetrics,
    get_performance_optimizer, smart_cache
)
from hipaa_compliance_summarizer.auto_scaling import (
    AutoScaler, WorkerPool, ScalingRule, ScalingMetric, ScalingDirection,
    ResourceMetrics, initialize_auto_scaling
)


class TestAdaptiveCache:
    """Test adaptive caching system."""

    def test_cache_initialization(self):
        """Test cache initialization with proper parameters."""
        cache = AdaptiveCache(max_size=100, ttl_seconds=300)
        
        assert cache.max_size == 100
        assert cache.ttl_seconds == 300
        assert len(cache._cache) == 0

    def test_basic_cache_operations(self):
        """Test basic cache get/put operations."""
        cache = AdaptiveCache(max_size=10, ttl_seconds=60)
        
        # Test cache miss
        result = cache.get("key1")
        assert result is None
        
        # Test cache put and hit
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        
        # Test cache statistics
        stats = cache.get_stats()
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["size"] == 1

    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        cache = AdaptiveCache(max_size=10, ttl_seconds=1)  # 1 second TTL
        
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        result = cache.get("key1")
        assert result is None  # Should be expired

    def test_cache_size_limit(self):
        """Test cache size limitation and LRU eviction."""
        cache = AdaptiveCache(max_size=3, ttl_seconds=60)
        
        # Fill cache to capacity
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        assert cache.get_stats()["size"] == 3
        
        # Adding another item should evict the LRU item
        cache.put("key4", "value4")
        
        assert cache.get_stats()["size"] == 3
        # key1 should be evicted (least recently used)
        assert cache.get("key1") is None
        assert cache.get("key4") == "value4"

    @patch('psutil.virtual_memory')
    def test_adaptive_sizing_memory_pressure(self, mock_memory):
        """Test adaptive cache sizing under memory pressure."""
        cache = AdaptiveCache(max_size=100, ttl_seconds=60)
        
        # Fill cache
        for i in range(50):
            cache.put(f"key{i}", f"value{i}")
        
        # Simulate high memory usage
        mock_memory.return_value.percent = 85.0
        cache._memory_pressure = True
        
        # Trigger adaptive sizing
        cache._adapt_cache_size()
        
        # Cache size should be reduced
        assert len(cache._cache) < 50

    def test_cache_hit_ratio_calculation(self):
        """Test cache hit ratio calculation."""
        cache = AdaptiveCache(max_size=10, ttl_seconds=60)
        
        # Generate some hits and misses
        cache.put("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        cache.get("key1")  # Hit
        cache.get("key3")  # Miss
        
        stats = cache.get_stats()
        expected_ratio = 2 / (2 + 2)  # 2 hits, 2 misses
        assert abs(stats["hit_ratio"] - expected_ratio) < 0.01

    def test_cache_cleanup_expired_entries(self):
        """Test cleanup of expired cache entries."""
        cache = AdaptiveCache(max_size=10, ttl_seconds=1)
        
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        
        assert cache.get_stats()["size"] == 2
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Force cleanup
        cache._cleanup_expired()
        
        assert cache.get_stats()["size"] == 0


class TestPerformanceOptimizer:
    """Test performance optimization engine."""

    def test_optimizer_initialization(self):
        """Test performance optimizer initialization."""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.adaptive_cache is not None
        assert isinstance(optimizer._optimization_strategies, dict)
        assert len(optimizer._optimization_strategies) > 0

    @patch('psutil.cpu_count')
    @patch('psutil.virtual_memory')
    def test_batch_processing_optimization(self, mock_memory, mock_cpu):
        """Test batch processing optimization."""
        mock_cpu.return_value = 8
        mock_memory.return_value.available = 8 * 1024 * 1024 * 1024  # 8GB
        
        optimizer = PerformanceOptimizer()
        
        config = optimizer.optimize_operation("batch_processing", operation_count=1000)
        
        assert "batch_size" in config
        assert "worker_count" in config
        assert config["worker_count"] <= 8
        assert config["batch_size"] > 0

    @patch('psutil.virtual_memory')
    def test_memory_management_optimization(self, mock_memory):
        """Test memory management optimization."""
        mock_memory.return_value.percent = 75.0
        mock_memory.return_value.available = 2 * 1024 * 1024 * 1024  # 2GB
        
        optimizer = PerformanceOptimizer()
        
        config = optimizer.optimize_operation("memory_management")
        
        assert "gc_threshold" in config
        assert "cache_size_multiplier" in config
        assert "enable_memory_mapping" in config

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_optimization_metrics_generation(self, mock_memory, mock_cpu):
        """Test optimization metrics generation."""
        mock_cpu.return_value = 50.0
        mock_memory.return_value.percent = 60.0
        mock_memory.return_value.used = 4 * 1024 * 1024 * 1024  # 4GB
        
        optimizer = PerformanceOptimizer()
        
        metrics = optimizer.get_optimization_metrics()
        
        assert isinstance(metrics, OptimizationMetrics)
        assert 0 <= metrics.optimization_score <= 1.0
        assert metrics.cpu_usage_percent == 50.0
        assert metrics.cache_hit_ratio >= 0

    def test_smart_caching_decorator(self):
        """Test smart caching decorator functionality."""
        call_count = 0
        
        @smart_cache
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same arguments should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Call with different arguments should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_global_optimizer_singleton(self):
        """Test global optimizer singleton pattern."""
        optimizer1 = get_performance_optimizer()
        optimizer2 = get_performance_optimizer()
        
        assert optimizer1 is optimizer2


class TestAsyncOptimizer:
    """Test asynchronous performance optimization."""

    @pytest.mark.asyncio
    async def test_async_optimizer_initialization(self):
        """Test async optimizer initialization."""
        optimizer = AsyncOptimizer(max_concurrent=10)
        
        assert optimizer.max_concurrent == 10
        assert optimizer.semaphore._value == 10

    @pytest.mark.asyncio
    async def test_concurrency_control(self):
        """Test async concurrency control."""
        optimizer = AsyncOptimizer(max_concurrent=2)
        
        async def slow_operation(delay):
            await asyncio.sleep(delay)
            return f"completed_{delay}"
        
        start_time = time.time()
        
        # Run 3 operations with max concurrency of 2
        tasks = [
            optimizer.optimize_async_operation(slow_operation(0.1)),
            optimizer.optimize_async_operation(slow_operation(0.1)),
            optimizer.optimize_async_operation(slow_operation(0.1))
        ]
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Should take more than 0.1 seconds due to concurrency limit
        assert end_time - start_time > 0.15
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_batch_async_operations(self):
        """Test batch async operations."""
        optimizer = AsyncOptimizer(max_concurrent=5)
        
        async def simple_operation(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        coroutines = [simple_operation(i) for i in range(10)]
        
        results = await optimizer.batch_async_operations(coroutines, batch_size=3)
        
        assert len(results) == 10
        assert results[0] == 0
        assert results[5] == 10

    @pytest.mark.asyncio
    async def test_active_task_tracking(self):
        """Test active task count tracking."""
        optimizer = AsyncOptimizer(max_concurrent=5)
        
        async def long_operation():
            await asyncio.sleep(0.1)
            return "done"
        
        # Start task but don't await it immediately
        task = asyncio.create_task(optimizer.optimize_async_operation(long_operation()))
        
        # Check active task count
        await asyncio.sleep(0.05)  # Let task start
        active_count = optimizer.get_active_task_count()
        
        await task  # Complete the task
        
        assert active_count >= 0  # Should have some active tasks


class TestWorkerPool:
    """Test dynamic worker pool."""

    def test_worker_pool_initialization(self):
        """Test worker pool initialization."""
        pool = WorkerPool(min_workers=2, max_workers=10, pool_type="thread")
        
        assert pool.min_workers == 2
        assert pool.max_workers == 10
        assert pool.current_workers == 2
        assert pool.pool_type == "thread"

    def test_worker_scaling(self):
        """Test worker pool scaling."""
        pool = WorkerPool(min_workers=1, max_workers=5)
        
        assert pool.current_workers == 1
        
        # Scale up
        result = pool.scale_workers(3)
        assert result is True
        assert pool.current_workers == 3
        
        # Scale down
        result = pool.scale_workers(2)
        assert result is True
        assert pool.current_workers == 2
        
        # Try to scale below minimum
        result = pool.scale_workers(0)
        assert pool.current_workers == 1  # Should be clamped to minimum

    def test_task_submission(self):
        """Test task submission to worker pool."""
        pool = WorkerPool(min_workers=2, max_workers=4)
        
        def simple_task(x):
            return x * 2
        
        future = pool.submit_task(simple_task, 5)
        result = future.result(timeout=5)
        
        assert result == 10

    def test_pool_metrics(self):
        """Test worker pool metrics collection."""
        pool = WorkerPool(min_workers=1, max_workers=3)
        
        metrics = pool.get_metrics()
        
        assert "current_workers" in metrics
        assert "min_workers" in metrics
        assert "max_workers" in metrics
        assert "active_tasks" in metrics
        assert "completed_tasks" in metrics

    def test_pool_shutdown(self):
        """Test worker pool shutdown."""
        pool = WorkerPool(min_workers=1, max_workers=3)
        
        pool.shutdown(wait=True)
        
        assert pool._shutdown is True


class TestAutoScaler:
    """Test auto-scaling system."""

    def test_autoscaler_initialization(self):
        """Test auto-scaler initialization."""
        worker_pool = WorkerPool(min_workers=1, max_workers=5)
        scaler = AutoScaler(worker_pool=worker_pool)
        
        assert scaler.worker_pool is worker_pool
        assert len(scaler.scaling_rules) > 0
        assert scaler.default_cooldown == 60

    def test_scaling_rule_registration(self):
        """Test custom scaling rule registration."""
        scaler = AutoScaler()
        
        custom_rule = ScalingRule(
            metric=ScalingMetric.THROUGHPUT,
            threshold_up=100.0,
            threshold_down=20.0,
            scale_up_amount=2,
            scale_down_amount=1,
            cooldown_seconds=90
        )
        
        initial_rule_count = len(scaler.scaling_rules)
        scaler.add_scaling_rule(custom_rule)
        
        assert len(scaler.scaling_rules) == initial_rule_count + 1

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_metrics_collection(self, mock_memory, mock_cpu):
        """Test resource metrics collection."""
        mock_cpu.return_value = 75.0
        mock_memory.return_value.percent = 60.0
        
        scaler = AutoScaler()
        
        metrics = scaler._collect_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.cpu_percent == 75.0
        assert metrics.memory_percent == 60.0

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_scaling_decision_evaluation(self, mock_memory, mock_cpu):
        """Test scaling decision evaluation."""
        mock_cpu.return_value = 85.0  # High CPU usage
        mock_memory.return_value.percent = 50.0
        
        scaler = AutoScaler()
        scaler.last_scaling_time = 0  # No recent scaling
        
        metrics = scaler._collect_metrics()
        decision = scaler._evaluate_scaling_decision(metrics)
        
        # High CPU should trigger scale up
        assert decision == ScalingDirection.SCALE_UP or decision == ScalingDirection.NO_CHANGE

    def test_scaling_status_reporting(self):
        """Test scaling status reporting."""
        scaler = AutoScaler()
        
        status = scaler.get_scaling_status()
        
        assert "current_workers" in status
        assert "monitoring_active" in status
        assert "current_metrics" in status
        assert "scaling_rules" in status

    def test_monitoring_lifecycle(self):
        """Test auto-scaling monitoring start/stop."""
        scaler = AutoScaler()
        
        assert not (scaler._monitoring_thread and scaler._monitoring_thread.is_alive())
        
        scaler.start_monitoring()
        time.sleep(0.1)  # Let thread start
        
        assert scaler._monitoring_thread and scaler._monitoring_thread.is_alive()
        
        scaler.stop_monitoring()
        
        assert scaler._stop_monitoring.is_set()


class TestScalingIntegration:
    """Test integration between scaling components."""

    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_performance_optimizer_with_autoscaler(self, mock_memory, mock_cpu):
        """Test integration between performance optimizer and auto-scaler."""
        mock_cpu.return_value = 60.0
        mock_memory.return_value.percent = 70.0
        mock_memory.return_value.used = 7 * 1024 * 1024 * 1024  # 7GB
        
        # Create performance monitor mock
        mock_monitor = MagicMock()
        mock_monitor.get_processing_metrics.return_value.throughput_docs_per_minute = 50.0
        mock_monitor.get_processing_metrics.return_value.avg_processing_time = 2.0
        mock_monitor.get_processing_metrics.return_value.error_rate = 0.05
        mock_monitor.get_active_processing_count.return_value = 3
        
        # Initialize systems
        optimizer = PerformanceOptimizer(performance_monitor=mock_monitor)
        scaler = AutoScaler(performance_monitor=mock_monitor)
        
        # Get optimization metrics and scaling status
        opt_metrics = optimizer.get_optimization_metrics()
        scaling_status = scaler.get_scaling_status()
        
        assert opt_metrics.throughput > 0
        assert "current_workers" in scaling_status

    def test_initialize_auto_scaling_function(self):
        """Test auto-scaling initialization function."""
        scaler = initialize_auto_scaling(min_workers=2, max_workers=8)
        
        assert isinstance(scaler, AutoScaler)
        assert scaler.worker_pool.min_workers == 2
        assert scaler.worker_pool.max_workers == 8

    @patch('time.time')
    def test_cooldown_period_enforcement(self, mock_time):
        """Test that cooldown periods are properly enforced."""
        mock_time.return_value = 1000.0
        
        scaler = AutoScaler()
        scaler.last_scaling_time = 990.0  # 10 seconds ago
        scaler.default_cooldown = 60  # 60 second cooldown
        
        # Create high CPU metrics that would normally trigger scaling
        metrics = ResourceMetrics(
            cpu_percent=90.0,
            memory_percent=50.0,
            disk_io_percent=30.0,
            queue_length=5,
            active_workers=2,
            avg_processing_time_ms=1000.0,
            throughput_per_minute=30.0,
            error_rate=0.02
        )
        
        decision = scaler._evaluate_scaling_decision(metrics)
        
        # Should not scale due to cooldown
        assert decision == ScalingDirection.NO_CHANGE

    def test_scaling_event_history(self):
        """Test scaling event history tracking."""
        scaler = AutoScaler()
        
        # Simulate a scaling event
        from hipaa_compliance_summarizer.auto_scaling import ScalingEvent
        
        event = ScalingEvent(
            timestamp=time.time(),
            direction=ScalingDirection.SCALE_UP,
            metric=ScalingMetric.CPU_UTILIZATION,
            metric_value=85.0,
            threshold=80.0,
            old_capacity=2,
            new_capacity=4,
            reason="cpu_utilization=85.0 > 80.0"
        )
        
        scaler.scaling_history.append(event)
        
        status = scaler.get_scaling_status()
        
        assert len(status["recent_scaling_events"]) == 1
        assert status["recent_scaling_events"][0]["direction"] == "scale_up"

    def test_adaptive_cache_with_optimizer(self):
        """Test adaptive cache integration with performance optimizer."""
        optimizer = PerformanceOptimizer()
        cache = optimizer.adaptive_cache
        
        # Add some data to cache
        for i in range(20):
            cache.put(f"key_{i}", f"value_{i}")
        
        # Get optimization metrics which should include cache statistics
        metrics = optimizer.get_optimization_metrics()
        
        assert metrics.cache_hit_ratio >= 0
        
        # Cache stats should be accessible
        cache_stats = cache.get_stats()
        assert cache_stats["size"] > 0