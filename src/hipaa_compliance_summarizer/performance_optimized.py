"""Advanced performance optimization system for HIPAA compliance processing."""

import asyncio
import gc
import hashlib
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache, wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from weakref import WeakKeyDictionary

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .constants import PERFORMANCE_LIMITS
from .monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


@dataclass
class OptimizationMetrics:
    """Performance optimization metrics."""
    cache_hit_ratio: float
    memory_usage_mb: float
    cpu_usage_percent: float
    processing_throughput: float
    average_response_time_ms: float
    concurrent_operations: int
    optimization_score: float


class AdaptiveCache:
    """Intelligent adaptive caching system with memory management."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum number of cache entries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_count: Dict[str, int] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        
        # Adaptive sizing
        self._hit_count = 0
        self._miss_count = 0
        self._memory_pressure = False
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(60):  # Check every minute
                self._cleanup_expired()
                self._adapt_cache_size()
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            for key in expired_keys:
                self._remove_key(key)

    def _adapt_cache_size(self):
        """Dynamically adapt cache size based on memory pressure."""
        try:
            if HAS_PSUTIL:
                memory_percent = psutil.virtual_memory().percent
            else:
                memory_percent = 60.0  # Assume moderate memory usage
            self._memory_pressure = memory_percent > 80
            
            if self._memory_pressure:
                # Reduce cache size under memory pressure
                new_size = max(100, int(self.max_size * 0.7))
                if new_size < len(self._cache):
                    self._evict_lru_entries(len(self._cache) - new_size)
                    logger.warning(f"Cache size reduced to {new_size} due to memory pressure")
            elif memory_percent < 60:
                # Increase cache size when memory is available
                hit_ratio = self._hit_count / (self._hit_count + self._miss_count) if (self._hit_count + self._miss_count) > 0 else 0
                if hit_ratio > 0.8:  # High hit ratio indicates cache is effective
                    self.max_size = min(PERFORMANCE_LIMITS.MAX_CACHE_SIZE, int(self.max_size * 1.1))
        except Exception as e:
            logger.warning(f"Error adapting cache size: {e}")

    def _evict_lru_entries(self, count: int):
        """Evict least recently used entries."""
        if count <= 0:
            return
        
        # Sort by access time and remove oldest
        sorted_keys = sorted(
            self._access_times.keys(),
            key=lambda k: self._access_times.get(k, 0)
        )
        
        for key in sorted_keys[:count]:
            self._remove_key(key)

    def _remove_key(self, key: str):
        """Remove key from all cache structures."""
        self._cache.pop(key, None)
        self._access_count.pop(key, None)
        self._access_times.pop(key, None)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                current_time = time.time()
                
                # Check if expired
                if current_time - timestamp > self.ttl_seconds:
                    self._remove_key(key)
                    self._miss_count += 1
                    return None
                
                # Update access statistics
                self._access_count[key] = self._access_count.get(key, 0) + 1
                self._access_times[key] = current_time
                self._hit_count += 1
                return value
            
            self._miss_count += 1
            return None

    def put(self, key: str, value: Any):
        """Put value in cache."""
        with self._lock:
            current_time = time.time()
            
            # Check if cache is full
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Evict LRU entry
                self._evict_lru_entries(1)
            
            self._cache[key] = (value, current_time)
            self._access_count[key] = 1
            self._access_times[key] = current_time

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
            self._access_times.clear()
            self._hit_count = 0
            self._miss_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_ratio = self._hit_count / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_ratio": hit_ratio,
                "memory_pressure": self._memory_pressure,
                "ttl_seconds": self.ttl_seconds
            }

    def __del__(self):
        """Cleanup on destruction."""
        if self._cleanup_thread:
            self._stop_cleanup.set()


class PerformanceOptimizer:
    """Advanced performance optimization engine."""

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None):
        """Initialize performance optimizer."""
        self.performance_monitor = performance_monitor
        self.adaptive_cache = AdaptiveCache()
        
        # Resource monitoring
        self._resource_monitor_thread = None
        self._stop_monitoring = threading.Event()
        
        # Optimization strategies
        self._optimization_strategies = {}
        self._register_default_strategies()
        
        # Performance baselines
        self._performance_baselines = {}
        self._start_resource_monitoring()

    def _start_resource_monitoring(self):
        """Start background resource monitoring."""
        def monitor_worker():
            while not self._stop_monitoring.wait(5):  # Check every 5 seconds
                try:
                    self._update_performance_baselines()
                    self._apply_dynamic_optimizations()
                except Exception as e:
                    logger.warning(f"Error in resource monitoring: {e}")
        
        self._resource_monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self._resource_monitor_thread.start()

    def _update_performance_baselines(self):
        """Update performance baselines for adaptive optimization."""
        try:
            if HAS_PSUTIL:
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk_io = psutil.disk_io_counters()
            else:
                # Fallback values when psutil is not available
                cpu_percent = 50.0
                memory = type('MockMemory', (), {'percent': 60.0, 'available': 2*1024*1024*1024})()
                disk_io = type('MockDiskIO', (), {'read_bytes': 0, 'write_bytes': 0})()
            
            current_time = time.time()
            baseline = {
                "timestamp": current_time,
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_read_mb_s": getattr(disk_io, 'read_bytes', 0) / (1024 * 1024) if disk_io else 0,
                "disk_write_mb_s": getattr(disk_io, 'write_bytes', 0) / (1024 * 1024) if disk_io else 0
            }
            
            self._performance_baselines[current_time] = baseline
            
            # Keep only recent baselines (last hour)
            cutoff_time = current_time - 3600
            self._performance_baselines = {
                t: b for t, b in self._performance_baselines.items() if t > cutoff_time
            }
            
        except Exception as e:
            logger.warning(f"Error updating performance baselines: {e}")

    def _apply_dynamic_optimizations(self):
        """Apply dynamic optimizations based on current performance."""
        if not self._performance_baselines:
            return
        
        latest_baseline = list(self._performance_baselines.values())[-1]
        
        # CPU-based optimizations
        if latest_baseline["cpu_percent"] > 80:
            self._optimize_for_high_cpu()
        elif latest_baseline["cpu_percent"] < 20:
            self._optimize_for_low_cpu()
        
        # Memory-based optimizations
        if latest_baseline["memory_percent"] > 85:
            self._optimize_for_high_memory()
        
        # Disk I/O optimizations
        if latest_baseline["disk_read_mb_s"] > 100 or latest_baseline["disk_write_mb_s"] > 100:
            self._optimize_for_high_disk_io()

    def _optimize_for_high_cpu(self):
        """Optimize for high CPU usage scenarios."""
        logger.info("Applying high CPU optimizations")
        
        # Reduce cache cleanup frequency
        if hasattr(self.adaptive_cache, '_cleanup_interval'):
            self.adaptive_cache._cleanup_interval = 300  # 5 minutes
        
        # Force garbage collection to free up resources
        gc.collect()

    def _optimize_for_low_cpu(self):
        """Optimize for low CPU usage scenarios."""
        logger.debug("Applying low CPU optimizations")
        
        # Increase cache cleanup frequency
        if hasattr(self.adaptive_cache, '_cleanup_interval'):
            self.adaptive_cache._cleanup_interval = 30  # 30 seconds

    def _optimize_for_high_memory(self):
        """Optimize for high memory usage scenarios."""
        logger.warning("Applying high memory optimizations")
        
        # Trigger cache cleanup
        self.adaptive_cache._cleanup_expired()
        
        # Force garbage collection
        gc.collect()
        
        # Reduce cache size temporarily
        if len(self.adaptive_cache._cache) > 100:
            self.adaptive_cache._evict_lru_entries(len(self.adaptive_cache._cache) // 4)

    def _optimize_for_high_disk_io(self):
        """Optimize for high disk I/O scenarios."""
        logger.info("Applying high disk I/O optimizations")
        
        # Increase cache TTL to reduce disk reads
        self.adaptive_cache.ttl_seconds = min(7200, self.adaptive_cache.ttl_seconds * 2)

    def _register_default_strategies(self):
        """Register default optimization strategies."""
        self._optimization_strategies.update({
            "batch_processing": self._optimize_batch_processing,
            "memory_management": self._optimize_memory_management,
            "io_operations": self._optimize_io_operations,
            "cpu_utilization": self._optimize_cpu_utilization
        })

    def _optimize_batch_processing(self, operation_count: int) -> Dict[str, Any]:
        """Optimize batch processing configuration."""
        if HAS_PSUTIL:
            available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
            cpu_count = psutil.cpu_count()
        else:
            available_memory = 2048.0  # 2GB fallback
            cpu_count = 4  # 4 CPU fallback
        
        # Calculate optimal batch size
        optimal_batch_size = min(
            operation_count,
            max(1, int(available_memory / 10)),  # Assume 10MB per operation
            cpu_count * 4  # 4x CPU cores
        )
        
        # Calculate optimal worker count
        optimal_workers = min(
            cpu_count,
            max(1, optimal_batch_size // 10)
        )
        
        return {
            "batch_size": optimal_batch_size,
            "worker_count": optimal_workers,
            "memory_per_batch_mb": available_memory / optimal_workers if optimal_workers > 0 else available_memory
        }

    def _optimize_memory_management(self) -> Dict[str, Any]:
        """Optimize memory management configuration."""
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
        else:
            memory = type('MockMemory', (), {'percent': 60.0})()
        
        return {
            "gc_threshold": 0.8 if memory.percent > 70 else 0.9,
            "cache_size_multiplier": 0.5 if memory.percent > 80 else 1.0,
            "enable_memory_mapping": memory.available > 1024 * 1024 * 1024  # 1GB
        }

    def _optimize_io_operations(self) -> Dict[str, Any]:
        """Optimize I/O operations configuration."""
        if HAS_PSUTIL:
            disk_io = psutil.disk_io_counters()
        else:
            disk_io = type('MockDiskIO', (), {'read_bytes': 0, 'write_bytes': 0})()
        
        return {
            "read_buffer_size": 64 * 1024 if disk_io else 8 * 1024,  # 64KB or 8KB
            "write_buffer_size": 64 * 1024 if disk_io else 8 * 1024,
            "async_io_enabled": True,
            "io_queue_depth": 32
        }

    def _optimize_cpu_utilization(self) -> Dict[str, Any]:
        """Optimize CPU utilization configuration."""
        if HAS_PSUTIL:
            cpu_count = psutil.cpu_count()
            cpu_percent = psutil.cpu_percent(interval=1)
        else:
            cpu_count = 4
            cpu_percent = 50.0
        
        return {
            "thread_pool_size": max(1, int(cpu_count * (100 - cpu_percent) / 100)),
            "process_pool_size": max(1, cpu_count - 1),
            "enable_hyperthreading": cpu_count > 4,
            "cpu_affinity_enabled": cpu_count > 2
        }

    def optimize_operation(self, operation_type: str, **kwargs) -> Dict[str, Any]:
        """Get optimization configuration for specific operation type."""
        if operation_type in self._optimization_strategies:
            return self._optimization_strategies[operation_type](**kwargs)
        else:
            logger.warning(f"No optimization strategy for operation type: {operation_type}")
            return {}

    def get_optimization_metrics(self) -> OptimizationMetrics:
        """Get current optimization metrics."""
        cache_stats = self.adaptive_cache.get_stats()
        
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
        else:
            cpu_percent = 50.0
            memory = type('MockMemory', (), {'percent': 60.0, 'used': 4*1024*1024*1024})()
        
        # Calculate throughput from performance monitor if available
        throughput = 0.0
        response_time = 0.0
        concurrent_ops = 0
        
        if self.performance_monitor:
            processing_metrics = self.performance_monitor.get_processing_metrics()
            throughput = processing_metrics.throughput_docs_per_minute
            response_time = processing_metrics.avg_processing_time * 1000  # Convert to ms
            concurrent_ops = self.performance_monitor.get_active_processing_count()
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            cache_stats["hit_ratio"],
            cpu_percent,
            memory.percent,
            throughput
        )
        
        return OptimizationMetrics(
            cache_hit_ratio=cache_stats["hit_ratio"],
            memory_usage_mb=memory.used / (1024 * 1024),
            cpu_usage_percent=cpu_percent,
            processing_throughput=throughput,
            average_response_time_ms=response_time,
            concurrent_operations=concurrent_ops,
            optimization_score=optimization_score
        )

    def _calculate_optimization_score(self, cache_hit_ratio: float, cpu_percent: float, 
                                    memory_percent: float, throughput: float) -> float:
        """Calculate overall optimization score (0.0 to 1.0)."""
        # Weight different factors
        cache_score = cache_hit_ratio  # Higher is better
        cpu_score = max(0, 1.0 - (cpu_percent / 100))  # Lower usage is better (not overloaded)
        memory_score = max(0, 1.0 - (memory_percent / 100))  # Lower usage is better
        throughput_score = min(1.0, throughput / 100)  # Normalize to reasonable baseline
        
        # Weighted average
        return (cache_score * 0.3 + cpu_score * 0.2 + memory_score * 0.2 + throughput_score * 0.3)

    def enable_smart_caching(self, func: Callable) -> Callable:
        """Decorator to enable smart caching for functions."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Try to get from cache
            cached_result = self.adaptive_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            self.adaptive_cache.put(cache_key, result)
            
            return result
        
        return wrapper

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        # Create a stable hash of the arguments
        key_data = {
            "function": func_name,
            "args": [str(arg) for arg in args],
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())}
        }
        
        key_json = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_json.encode()).hexdigest()[:16]

    def __del__(self):
        """Cleanup on destruction."""
        if self._resource_monitor_thread:
            self._stop_monitoring.set()


class AsyncOptimizer:
    """Asynchronous performance optimization for I/O bound operations."""

    def __init__(self, max_concurrent: int = 100):
        """Initialize async optimizer."""
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks = set()

    async def optimize_async_operation(self, coroutine):
        """Optimize asynchronous operation with concurrency control."""
        async with self.semaphore:
            task = asyncio.create_task(coroutine)
            self._active_tasks.add(task)
            
            try:
                result = await task
                return result
            finally:
                self._active_tasks.discard(task)

    async def batch_async_operations(self, coroutines: List, batch_size: int = None):
        """Execute coroutines in optimized batches."""
        if batch_size is None:
            batch_size = min(self.max_concurrent, len(coroutines))
        
        results = []
        for i in range(0, len(coroutines), batch_size):
            batch = coroutines[i:i + batch_size]
            batch_results = await asyncio.gather(*[
                self.optimize_async_operation(coro) for coro in batch
            ], return_exceptions=True)
            results.extend(batch_results)
        
        return results

    def get_active_task_count(self) -> int:
        """Get number of active async tasks."""
        return len(self._active_tasks)


# Global optimizer instance
_global_optimizer: Optional[PerformanceOptimizer] = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer


def smart_cache(func: Callable) -> Callable:
    """Decorator for intelligent caching."""
    optimizer = get_performance_optimizer()
    return optimizer.enable_smart_caching(func)


__all__ = [
    "OptimizationMetrics",
    "AdaptiveCache", 
    "PerformanceOptimizer",
    "AsyncOptimizer",
    "get_performance_optimizer",
    "smart_cache"
]