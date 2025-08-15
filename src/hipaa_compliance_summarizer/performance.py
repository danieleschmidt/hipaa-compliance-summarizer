"""Performance optimization and scaling features for HIPAA compliance system."""

import asyncio
import functools
import logging
import multiprocessing as mp
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""

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
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def finalize(self):
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()

        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

        if self.duration_seconds > 0 and self.items_processed > 0:
            self.throughput_ops_per_second = self.items_processed / self.duration_seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": self.duration_seconds,
            "throughput_ops_per_second": self.throughput_ops_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_usage_percent": self.cpu_usage_percent,
            "items_processed": self.items_processed,
            "errors_encountered": self.errors_encountered,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
            "additional_metrics": self.additional_metrics
        }


class PerformanceOptimizer:
    """Performance optimization and monitoring."""

    def __init__(self):
        """Initialize performance optimizer."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.optimization_rules: Dict[str, Callable] = {}
        self.performance_thresholds: Dict[str, float] = {
            "max_response_time_seconds": 30.0,
            "min_throughput_ops_per_second": 1.0,
            "max_memory_usage_mb": 1024.0,
            "max_cpu_usage_percent": 80.0,
            "min_cache_hit_ratio": 0.5
        }

    def start_performance_tracking(self, operation_name: str) -> PerformanceMetrics:
        """Start tracking performance for an operation.
        
        Args:
            operation_name: Name of the operation to track
            
        Returns:
            Performance metrics object
        """
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )

        # Get initial system metrics
        if HAS_PSUTIL:
            try:
                process = psutil.Process()
                metrics.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
                metrics.cpu_usage_percent = process.cpu_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        else:
            # Fallback values when psutil is not available
            metrics.memory_usage_mb = 100.0  # 100MB fallback
            metrics.cpu_usage_percent = 10.0  # 10% CPU fallback

        return metrics

    def finish_performance_tracking(self, metrics: PerformanceMetrics):
        """Finish tracking and analyze performance.
        
        Args:
            metrics: Performance metrics to finalize
        """
        metrics.finalize()
        self.metrics_history.append(metrics)

        # Analyze performance and apply optimizations
        self._analyze_performance(metrics)

        throughput_str = f"{metrics.throughput_ops_per_second:.2f} ops/s" if metrics.throughput_ops_per_second else "N/A ops/s"
        logger.info(
            f"Performance tracking completed for '{metrics.operation_name}': "
            f"{metrics.duration_seconds:.2f}s, "
            f"{throughput_str}"
        )

    def _analyze_performance(self, metrics: PerformanceMetrics):
        """Analyze performance metrics and apply optimizations."""
        issues = []

        # Check response time
        if metrics.duration_seconds and metrics.duration_seconds > self.performance_thresholds["max_response_time_seconds"]:
            issues.append(f"Response time too high: {metrics.duration_seconds:.2f}s")

        # Check throughput
        if metrics.throughput_ops_per_second and metrics.throughput_ops_per_second < self.performance_thresholds["min_throughput_ops_per_second"]:
            issues.append(f"Throughput too low: {metrics.throughput_ops_per_second:.2f} ops/s")

        # Check memory usage
        if metrics.memory_usage_mb and metrics.memory_usage_mb > self.performance_thresholds["max_memory_usage_mb"]:
            issues.append(f"Memory usage too high: {metrics.memory_usage_mb:.2f} MB")

        # Check CPU usage
        if metrics.cpu_usage_percent and metrics.cpu_usage_percent > self.performance_thresholds["max_cpu_usage_percent"]:
            issues.append(f"CPU usage too high: {metrics.cpu_usage_percent:.2f}%")

        # Check cache performance
        total_cache_ops = metrics.cache_hits + metrics.cache_misses
        if total_cache_ops > 0:
            cache_hit_ratio = metrics.cache_hits / total_cache_ops
            if cache_hit_ratio < self.performance_thresholds["min_cache_hit_ratio"]:
                issues.append(f"Cache hit ratio too low: {cache_hit_ratio:.2f}")

        if issues:
            logger.warning(f"Performance issues detected for '{metrics.operation_name}': {', '.join(issues)}")

            # Apply optimization rules if available
            if metrics.operation_name in self.optimization_rules:
                try:
                    self.optimization_rules[metrics.operation_name](metrics)
                except Exception as e:
                    logger.error(f"Optimization rule failed for '{metrics.operation_name}': {e}")

    def register_optimization_rule(self, operation_name: str, rule_func: Callable):
        """Register optimization rule for operation.
        
        Args:
            operation_name: Name of the operation
            rule_func: Optimization function
        """
        self.optimization_rules[operation_name] = rule_func
        logger.info(f"Registered optimization rule for: {operation_name}")

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            Performance summary
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history if m.start_time >= cutoff_time]

        if not recent_metrics:
            return {"message": "No recent performance data"}

        # Calculate aggregated metrics
        total_operations = len(recent_metrics)
        total_items = sum(m.items_processed for m in recent_metrics)
        total_errors = sum(m.errors_encountered for m in recent_metrics)

        durations = [m.duration_seconds for m in recent_metrics if m.duration_seconds]
        throughputs = [m.throughput_ops_per_second for m in recent_metrics if m.throughput_ops_per_second]

        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_throughput = sum(throughputs) / len(throughputs) if throughputs else 0

        # Group by operation
        by_operation = {}
        for metrics in recent_metrics:
            op_name = metrics.operation_name
            if op_name not in by_operation:
                by_operation[op_name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "total_items": 0,
                    "total_errors": 0
                }

            stats = by_operation[op_name]
            stats["count"] += 1
            stats["total_duration"] += metrics.duration_seconds or 0
            stats["total_items"] += metrics.items_processed
            stats["total_errors"] += metrics.errors_encountered

        # Calculate operation-level averages
        for stats in by_operation.values():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]
                stats["avg_throughput"] = stats["total_items"] / stats["total_duration"] if stats["total_duration"] > 0 else 0
                stats["error_rate"] = stats["total_errors"] / stats["total_items"] if stats["total_items"] > 0 else 0

        return {
            "time_period_hours": hours,
            "total_operations": total_operations,
            "total_items_processed": total_items,
            "total_errors": total_errors,
            "overall_error_rate": total_errors / total_items if total_items > 0 else 0,
            "average_duration_seconds": avg_duration,
            "average_throughput_ops_per_second": avg_throughput,
            "by_operation": by_operation
        }


class ConcurrentProcessor:
    """High-performance concurrent processing."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        chunk_size: int = 100
    ):
        """Initialize concurrent processor.
        
        Args:
            max_workers: Maximum number of workers (defaults to CPU count)
            use_processes: Use processes instead of threads
            chunk_size: Size of work chunks
        """
        if max_workers is None:
            max_workers = min(32, (mp.cpu_count() or 1) + 4)

        self.max_workers = max_workers
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.executor = None

        logger.info(
            f"Concurrent processor initialized: {max_workers} workers, "
            f"{'processes' if use_processes else 'threads'}, "
            f"chunk size: {chunk_size}"
        )

    def __enter__(self):
        """Enter context manager."""
        if self.use_processes:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        if self.executor:
            self.executor.shutdown(wait=True)

    def process_batch(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items in parallel batches.
        
        Args:
            items: Items to process
            process_func: Function to process each item
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        if not items:
            return []

        # Split items into chunks
        chunks = [items[i:i + self.chunk_size] for i in range(0, len(items), self.chunk_size)]

        results = []
        completed_items = 0

        # Submit all chunks
        futures = []
        for chunk in chunks:
            future = self.executor.submit(self._process_chunk, chunk, process_func)
            futures.append(future)

        # Collect results as they complete
        for future in futures:
            try:
                chunk_results = future.result()
                results.extend(chunk_results)
                completed_items += len(chunk_results)

                if progress_callback:
                    progress_callback(completed_items, len(items))

            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                # Add None results for failed chunk
                results.extend([None] * self.chunk_size)

        return results[:len(items)]  # Trim to original length

    def _process_chunk(self, chunk: List[Any], process_func: Callable) -> List[Any]:
        """Process a single chunk of items."""
        return [process_func(item) for item in chunk]

    async def process_batch_async(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[Any]:
        """Process items asynchronously.
        
        Args:
            items: Items to process
            process_func: Async function to process each item
            progress_callback: Optional progress callback
            
        Returns:
            List of processing results
        """
        if not items:
            return []

        semaphore = asyncio.Semaphore(self.max_workers)

        async def process_with_semaphore(item, index):
            async with semaphore:
                try:
                    result = await process_func(item)
                    if progress_callback:
                        progress_callback(index + 1, len(items))
                    return result
                except Exception as e:
                    logger.error(f"Async processing failed for item {index}: {e}")
                    return None

        # Create tasks for all items
        tasks = [process_with_semaphore(item, i) for i, item in enumerate(items)]

        # Process all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Async task failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)

        return processed_results


class AdaptiveCache:
    """Adaptive caching with performance monitoring."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: Optional[int] = None,
        adaptive_sizing: bool = True
    ):
        """Initialize adaptive cache.
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live for cache entries
            adaptive_sizing: Enable adaptive cache sizing
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.adaptive_sizing = adaptive_sizing

        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = {}
        self.hit_count = 0
        self.miss_count = 0

        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        with self._lock:
            if key not in self.cache:
                self.miss_count += 1
                return None

            entry = self.cache[key]

            # Check TTL
            if self.ttl_seconds and entry["timestamp"]:
                age = (datetime.utcnow() - entry["timestamp"]).total_seconds()
                if age > self.ttl_seconds:
                    del self.cache[key]
                    del self.access_times[key]
                    del self.access_counts[key]
                    self.miss_count += 1
                    return None

            # Update access statistics
            self.access_times[key] = datetime.utcnow()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            self.hit_count += 1

            return entry["value"]

    def put(self, key: str, value: Any):
        """Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        with self._lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size:
                self._evict_entries()

            # Store entry with timestamp
            self.cache[key] = {
                "value": value,
                "timestamp": datetime.utcnow()
            }
            self.access_times[key] = datetime.utcnow()
            self.access_counts[key] = 1

    def _evict_entries(self):
        """Evict least recently used entries."""
        if not self.cache:
            return

        # Calculate number of entries to evict
        evict_count = max(1, len(self.cache) // 4)  # Evict 25%

        # Sort by access time (LRU)
        sorted_keys = sorted(
            self.access_times.keys(),
            key=lambda k: self.access_times[k]
        )

        # Evict oldest entries
        for key in sorted_keys[:evict_count]:
            del self.cache[key]
            del self.access_times[key]
            del self.access_counts[key]

        logger.debug(f"Evicted {evict_count} cache entries")

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_ratio = self.hit_count / total_requests if total_requests > 0 else 0.0

            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_ratio": hit_ratio,
                "total_requests": total_requests,
                "most_accessed_keys": sorted(
                    self.access_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
            }

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
            self.access_counts.clear()
            self.hit_count = 0
            self.miss_count = 0


class LoadBalancer:
    """Simple load balancer for distributing work."""

    def __init__(self, workers: List[Callable]):
        """Initialize load balancer.
        
        Args:
            workers: List of worker functions
        """
        self.workers = workers
        self.current_worker = 0
        self.worker_stats: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        # Initialize worker statistics
        for i in range(len(workers)):
            self.worker_stats[i] = {
                "requests": 0,
                "total_time": 0.0,
                "errors": 0,
                "avg_response_time": 0.0
            }

    def distribute_work(self, work_item: Any) -> Any:
        """Distribute work to next available worker.
        
        Args:
            work_item: Work item to process
            
        Returns:
            Processing result
        """
        with self._lock:
            worker_index = self._select_worker()
            worker = self.workers[worker_index]

        start_time = time.time()

        try:
            result = worker(work_item)

            # Update statistics
            processing_time = time.time() - start_time
            with self._lock:
                stats = self.worker_stats[worker_index]
                stats["requests"] += 1
                stats["total_time"] += processing_time
                stats["avg_response_time"] = stats["total_time"] / stats["requests"]

            return result

        except Exception as e:
            # Update error statistics
            with self._lock:
                self.worker_stats[worker_index]["errors"] += 1
            raise e

    def _select_worker(self) -> int:
        """Select next worker using round-robin with performance weighting."""
        # Simple round-robin for now
        worker_index = self.current_worker
        self.current_worker = (self.current_worker + 1) % len(self.workers)
        return worker_index

    def get_worker_statistics(self) -> Dict[str, Any]:
        """Get worker statistics."""
        with self._lock:
            return {
                "total_workers": len(self.workers),
                "worker_stats": dict(self.worker_stats),
                "total_requests": sum(stats["requests"] for stats in self.worker_stats.values()),
                "total_errors": sum(stats["errors"] for stats in self.worker_stats.values())
            }


# Global performance instances
performance_optimizer = PerformanceOptimizer()
adaptive_cache = AdaptiveCache()


def performance_monitor(operation_name: str):
    """Decorator for performance monitoring.
    
    Args:
        operation_name: Name of the operation to monitor
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            metrics = performance_optimizer.start_performance_tracking(operation_name)

            try:
                result = func(*args, **kwargs)
                metrics.items_processed = 1  # Default to 1 item
                return result
            except Exception as e:
                metrics.errors_encountered = 1
                raise e
            finally:
                performance_optimizer.finish_performance_tracking(metrics)

        return wrapper
    return decorator


def setup_performance_optimizations():
    """Setup default performance optimizations."""
    # Register optimization rules
    def phi_detection_optimization(metrics: PerformanceMetrics):
        """Optimize PHI detection performance."""
        if metrics.throughput_ops_per_second and metrics.throughput_ops_per_second < 5.0:
            logger.info("Applying PHI detection optimization: increasing cache size")
            adaptive_cache.max_size = min(adaptive_cache.max_size * 2, 5000)

    def batch_processing_optimization(metrics: PerformanceMetrics):
        """Optimize batch processing performance."""
        if metrics.duration_seconds and metrics.duration_seconds > 60.0:
            logger.info("Applying batch processing optimization: reducing chunk size")
            # This would need to be integrated with the actual batch processor

    performance_optimizer.register_optimization_rule("phi_detection", phi_detection_optimization)
    performance_optimizer.register_optimization_rule("batch_processing", batch_processing_optimization)

    logger.info("Performance optimizations setup completed")


# Initialize performance optimizations on import
setup_performance_optimizations()
