"""Enhanced performance optimization for HIPAA compliance system."""

import asyncio
import functools
import logging
import multiprocessing as mp
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from queue import PriorityQueue
from typing import Any, Awaitable, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Enhanced performance metrics container."""

    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    throughput_ops_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    items_processed: int = 0
    errors_encountered: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_workers: int = 1
    queue_wait_time: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()

        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

        if self.duration_seconds > 0 and self.items_processed > 0:
            self.throughput_ops_per_second = self.items_processed / self.duration_seconds

        # Calculate efficiency metrics
        self.additional_metrics['cache_hit_ratio'] = (
            self.cache_hits / max(self.cache_hits + self.cache_misses, 1)
        )
        self.additional_metrics['error_rate'] = (
            self.errors_encountered / max(self.items_processed, 1)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation_name': self.operation_name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'throughput_ops_per_second': self.throughput_ops_per_second,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'items_processed': self.items_processed,
            'errors_encountered': self.errors_encountered,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'parallel_workers': self.parallel_workers,
            'queue_wait_time': self.queue_wait_time,
            **self.additional_metrics
        }


@dataclass
class WorkerTask:
    """Task for worker processing."""

    priority: int
    task_id: str
    operation: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __lt__(self, other):
        return self.priority < other.priority


class AdaptiveCache:
    """Advanced adaptive cache with TTL and size limits."""

    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Any] = {}
        self._timestamps: Dict[str, float] = {}
        self._access_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            current_time = time.time()

            if key not in self._cache:
                self.misses += 1
                return None

            # Check TTL
            if current_time - self._timestamps[key] > self.default_ttl:
                self._remove_key(key)
                self.misses += 1
                return None

            # Update access tracking
            self._access_counts[key] += 1
            self.hits += 1
            return self._cache[key]

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set value in cache."""
        with self._lock:
            # Ensure cache size limits
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()

            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_counts[key] = 1

    def _remove_key(self, key: str) -> None:
        """Remove key from cache."""
        self._cache.pop(key, None)
        self._timestamps.pop(key, None)
        self._access_counts.pop(key, None)

    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._cache:
            return

        # Find least accessed key
        lru_key = min(self._access_counts.keys(), key=lambda k: self._access_counts[k])
        self._remove_key(lru_key)

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_counts.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / max(total_requests, 1)

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self.hits,
                'misses': self.misses,
                'hit_ratio': hit_ratio,
                'total_requests': total_requests
            }


class PerformanceOptimizer:
    """Enhanced performance optimizer with adaptive scaling."""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or mp.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.task_queue = PriorityQueue()
        self.metrics_history: List[PerformanceMetrics] = []
        self.cache = AdaptiveCache()
        self._active_tasks: Dict[str, Future] = {}
        self._performance_lock = threading.Lock()
        self.auto_scaling_enabled = True
        self.target_cpu_utilization = 0.7

    def execute_parallel(self,
                        operations: List[Callable],
                        use_processes: bool = False,
                        max_workers: Optional[int] = None) -> List[Future]:
        """Execute operations in parallel."""
        executor = self.process_pool if use_processes else self.thread_pool
        workers = min(max_workers or self.max_workers, len(operations))

        futures = []
        for operation in operations:
            future = executor.submit(operation)
            futures.append(future)

        return futures

    async def execute_async(self,
                           operations: List[Awaitable],
                           concurrency_limit: Optional[int] = None) -> List[Any]:
        """Execute async operations with concurrency limit."""
        semaphore = asyncio.Semaphore(concurrency_limit or self.max_workers)

        async def bounded_operation(op):
            async with semaphore:
                return await op

        return await asyncio.gather(*[bounded_operation(op) for op in operations])

    def cached_operation(self, cache_key: str, ttl: Optional[float] = None):
        """Decorator for caching operation results."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Try cache first
                result = self.cache.get(cache_key)
                if result is not None:
                    return result

                # Execute and cache
                result = func(*args, **kwargs)
                self.cache.set(cache_key, result, ttl)
                return result
            return wrapper
        return decorator

    def performance_monitor(self, operation_name: str):
        """Decorator for monitoring performance."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                metrics = PerformanceMetrics(
                    operation_name=operation_name,
                    start_time=datetime.utcnow()
                )

                # Capture initial system state
                process = psutil.Process()
                initial_memory = process.memory_info().rss / 1024 / 1024
                initial_cpu = process.cpu_percent()

                try:
                    result = func(*args, **kwargs)
                    metrics.items_processed = 1

                except Exception:
                    metrics.errors_encountered = 1
                    raise

                finally:
                    # Capture final system state
                    metrics.end_time = datetime.utcnow()
                    metrics.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                    metrics.cpu_usage_percent = process.cpu_percent()

                    # Update cache metrics
                    cache_stats = self.cache.get_stats()
                    metrics.cache_hits = cache_stats['hits']
                    metrics.cache_misses = cache_stats['misses']

                    metrics.finalize()

                    with self._performance_lock:
                        self.metrics_history.append(metrics)

                        # Keep only recent metrics
                        if len(self.metrics_history) > 1000:
                            self.metrics_history = self.metrics_history[-800:]

                return result
            return wrapper
        return decorator

    def batch_processor(self,
                       items: List[Any],
                       operation: Callable,
                       batch_size: int = 10,
                       max_workers: Optional[int] = None) -> List[Any]:
        """Process items in batches with parallel execution."""
        results = []
        workers = max_workers or self.max_workers

        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]

        def process_batch(batch):
            batch_results = []
            for item in batch:
                try:
                    result = operation(item)
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Error processing item: {e}")
                    batch_results.append(None)
            return batch_results

        # Execute batches in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): batch
                for batch in batches
            }

            for future in future_to_batch:
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")

        return results

    def adaptive_worker_scaling(self) -> int:
        """Calculate optimal worker count based on system metrics."""
        if not self.auto_scaling_enabled:
            return self.max_workers

        # Get current system utilization
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()

        # Scale based on CPU utilization
        if cpu_percent < self.target_cpu_utilization * 50:  # Under 35%
            optimal_workers = min(self.max_workers, self.max_workers + 2)
        elif cpu_percent > self.target_cpu_utilization * 100:  # Over 70%
            optimal_workers = max(1, self.max_workers - 2)
        else:
            optimal_workers = self.max_workers

        # Consider memory constraints
        if memory.percent > 85:
            optimal_workers = max(1, optimal_workers // 2)

        return optimal_workers

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.metrics_history
            if m.start_time > cutoff_time
        ]

        if not recent_metrics:
            return {'message': 'No metrics available for specified period'}

        # Calculate aggregated statistics
        total_operations = len(recent_metrics)
        total_items = sum(m.items_processed for m in recent_metrics)
        total_errors = sum(m.errors_encountered for m in recent_metrics)
        avg_duration = sum(m.duration_seconds or 0 for m in recent_metrics) / total_operations
        avg_throughput = sum(m.throughput_ops_per_second or 0 for m in recent_metrics) / total_operations

        # Group by operation type
        operations_stats = defaultdict(lambda: {
            'count': 0,
            'avg_duration': 0,
            'total_items': 0,
            'errors': 0
        })

        for metric in recent_metrics:
            op_name = metric.operation_name
            operations_stats[op_name]['count'] += 1
            operations_stats[op_name]['avg_duration'] += metric.duration_seconds or 0
            operations_stats[op_name]['total_items'] += metric.items_processed
            operations_stats[op_name]['errors'] += metric.errors_encountered

        # Finalize operation stats
        for op_name, stats in operations_stats.items():
            if stats['count'] > 0:
                stats['avg_duration'] /= stats['count']

        return {
            'report_period_hours': hours,
            'total_operations': total_operations,
            'total_items_processed': total_items,
            'total_errors': total_errors,
            'error_rate': total_errors / max(total_items, 1),
            'average_duration_seconds': avg_duration,
            'average_throughput_ops_per_second': avg_throughput,
            'cache_stats': self.cache.get_stats(),
            'operations_breakdown': dict(operations_stats),
            'system_metrics': {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'optimal_workers': self.adaptive_worker_scaling()
            },
            'recommendations': self._generate_recommendations(recent_metrics)
        }

    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []

        cache_stats = self.cache.get_stats()
        if cache_stats['hit_ratio'] < 0.5:
            recommendations.append("Consider increasing cache size or adjusting TTL settings")

        avg_cpu = psutil.cpu_percent()
        if avg_cpu > 80:
            recommendations.append("High CPU utilization detected - consider scaling horizontally")

        memory = psutil.virtual_memory()
        if memory.percent > 80:
            recommendations.append("High memory usage - consider optimizing memory allocation")

        error_rates = [m.additional_metrics.get('error_rate', 0) for m in metrics]
        avg_error_rate = sum(error_rates) / max(len(error_rates), 1)
        if avg_error_rate > 0.05:  # 5% error rate
            recommendations.append("High error rate detected - review error handling")

        return recommendations

    def shutdown(self):
        """Shutdown executors and cleanup."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


# Global performance optimizer
_performance_optimizer = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get global performance optimizer."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def performance_monitor(operation_name: str):
    """Performance monitoring decorator."""
    return get_performance_optimizer().performance_monitor(operation_name)


def cached_operation(cache_key: str, ttl: Optional[float] = None):
    """Caching decorator."""
    return get_performance_optimizer().cached_operation(cache_key, ttl)


__all__ = [
    'PerformanceMetrics',
    'WorkerTask',
    'AdaptiveCache',
    'PerformanceOptimizer',
    'get_performance_optimizer',
    'performance_monitor',
    'cached_operation'
]
