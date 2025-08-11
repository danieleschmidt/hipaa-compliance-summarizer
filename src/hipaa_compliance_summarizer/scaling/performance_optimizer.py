"""Advanced performance optimization and scaling components."""

import asyncio
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
import logging
import weakref

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class CachePolicy(str, Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    TTL = "ttl"  # Time To Live


class ResourceType(str, Enum):
    """Types of system resources to manage."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    CONNECTIONS = "connections"


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    size_bytes: int = 0

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self):
        """Update last accessed time and increment counter."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    success: bool = True
    error: Optional[str] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0


class AdvancedCache:
    """High-performance cache with multiple eviction policies."""
    
    def __init__(self, max_size: int = 10000, max_memory_mb: int = 512, 
                 policy: CachePolicy = CachePolicy.LRU, default_ttl: Optional[float] = None):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.policy = policy
        self.default_ttl = default_ttl
        
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        self.insertion_order = deque()  # For FIFO
        
        self.current_memory_usage = 0
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        
        self.lock = threading.RLock()
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread to clean expired entries."""
        def cleanup():
            while True:
                try:
                    self._cleanup_expired()
                    time.sleep(60)  # Cleanup every minute
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)
        
        cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        cleanup_thread.start()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.miss_count += 1
                return None
            
            entry = self.cache[key]
            
            # Check expiration
            if entry.is_expired():
                del self.cache[key]
                self._remove_from_tracking(key)
                self.miss_count += 1
                return None
            
            # Update access tracking
            entry.touch()
            self._update_access_tracking(key)
            
            self.hit_count += 1
            return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[float] = None) -> bool:
        """Set value in cache."""
        with self.lock:
            # Calculate size (rough estimate)
            try:
                import sys
                size_bytes = sys.getsizeof(value)
            except:
                size_bytes = 64  # Default estimate
            
            # Check if we need to evict entries
            if key not in self.cache:
                self._ensure_capacity(size_bytes)
            
            # Create/update entry
            entry = CacheEntry(
                key=key,
                value=value,
                ttl_seconds=ttl_seconds or self.default_ttl,
                size_bytes=size_bytes
            )
            
            # Update memory usage
            if key in self.cache:
                self.current_memory_usage -= self.cache[key].size_bytes
            
            self.cache[key] = entry
            self.current_memory_usage += size_bytes
            
            # Update tracking structures
            self._add_to_tracking(key)
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                self.current_memory_usage -= entry.size_bytes
                del self.cache[key]
                self._remove_from_tracking(key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            self.insertion_order.clear()
            self.current_memory_usage = 0
    
    def _ensure_capacity(self, additional_bytes: int):
        """Ensure cache has capacity for new entry."""
        # Check size limit
        while len(self.cache) >= self.max_size:
            self._evict_one_entry()
        
        # Check memory limit
        while self.current_memory_usage + additional_bytes > self.max_memory_bytes:
            if not self._evict_one_entry():
                break  # No more entries to evict
    
    def _evict_one_entry(self) -> bool:
        """Evict one entry based on policy."""
        if not self.cache:
            return False
        
        key_to_evict = None
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used
            while self.access_order:
                candidate = self.access_order.popleft()
                if candidate in self.cache:
                    key_to_evict = candidate
                    break
        
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            if self.frequency_counter:
                key_to_evict = min(self.frequency_counter.keys(), 
                                 key=lambda k: self.frequency_counter[k] if k in self.cache else float('inf'))
        
        elif self.policy == CachePolicy.FIFO:
            # Remove first inserted
            while self.insertion_order:
                candidate = self.insertion_order.popleft()
                if candidate in self.cache:
                    key_to_evict = candidate
                    break
        
        elif self.policy == CachePolicy.TTL:
            # Remove expired or oldest
            now = datetime.utcnow()
            expired_keys = [k for k, v in self.cache.items() if v.is_expired()]
            if expired_keys:
                key_to_evict = expired_keys[0]
            else:
                # Fallback to oldest
                key_to_evict = min(self.cache.keys(), 
                                 key=lambda k: self.cache[k].created_at)
        
        if key_to_evict and key_to_evict in self.cache:
            entry = self.cache[key_to_evict]
            self.current_memory_usage -= entry.size_bytes
            del self.cache[key_to_evict]
            self._remove_from_tracking(key_to_evict)
            self.eviction_count += 1
            return True
        
        return False
    
    def _add_to_tracking(self, key: str):
        """Add key to tracking structures."""
        if self.policy in [CachePolicy.LRU, CachePolicy.TTL]:
            self.access_order.append(key)
        
        if self.policy == CachePolicy.LFU:
            self.frequency_counter[key] = 0
        
        if self.policy == CachePolicy.FIFO:
            self.insertion_order.append(key)
    
    def _remove_from_tracking(self, key: str):
        """Remove key from tracking structures."""
        if key in self.frequency_counter:
            del self.frequency_counter[key]
    
    def _update_access_tracking(self, key: str):
        """Update access tracking for key."""
        if self.policy == CachePolicy.LRU:
            # Move to end of access order
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
        
        if self.policy == CachePolicy.LFU:
            self.frequency_counter[key] += 1
    
    def _cleanup_expired(self):
        """Clean up expired entries."""
        with self.lock:
            expired_keys = []
            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.delete(key)
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "memory_usage_mb": self.current_memory_usage / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": hit_rate,
                "eviction_count": self.eviction_count,
                "policy": self.policy.value
            }


class ConnectionPool:
    """Generic connection pool for resource management."""
    
    def __init__(self, factory_func: Callable, max_connections: int = 20, 
                 min_connections: int = 5, connection_timeout: float = 30.0):
        self.factory_func = factory_func
        self.max_connections = max_connections
        self.min_connections = min_connections
        self.connection_timeout = connection_timeout
        
        self.active_connections: List[Any] = []
        self.idle_connections: deque = deque()
        self.connection_times: Dict[Any, float] = {}
        
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        
        # Pre-create minimum connections
        self._ensure_minimum_connections()
        
        # Start maintenance thread
        self._start_maintenance_thread()
    
    def _ensure_minimum_connections(self):
        """Ensure minimum number of connections exist."""
        with self.lock:
            total_connections = len(self.active_connections) + len(self.idle_connections)
            needed = self.min_connections - total_connections
            
            for _ in range(needed):
                try:
                    conn = self.factory_func()
                    self.idle_connections.append(conn)
                    self.connection_times[conn] = time.time()
                except Exception as e:
                    logger.error(f"Failed to create connection: {e}")
                    break
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Acquire connection from pool."""
        end_time = time.time() + (timeout or self.connection_timeout)
        
        with self.condition:
            while True:
                # Try to get idle connection
                if self.idle_connections:
                    conn = self.idle_connections.popleft()
                    self.active_connections.append(conn)
                    return conn
                
                # Try to create new connection
                if len(self.active_connections) < self.max_connections:
                    try:
                        conn = self.factory_func()
                        self.active_connections.append(conn)
                        self.connection_times[conn] = time.time()
                        return conn
                    except Exception as e:
                        logger.error(f"Failed to create connection: {e}")
                
                # Wait for connection to be released
                remaining_time = end_time - time.time()
                if remaining_time <= 0:
                    return None
                
                self.condition.wait(timeout=min(remaining_time, 1.0))
    
    def release(self, connection: Any):
        """Release connection back to pool."""
        with self.condition:
            if connection in self.active_connections:
                self.active_connections.remove(connection)
                self.idle_connections.append(connection)
                self.connection_times[connection] = time.time()
                self.condition.notify()
    
    def _start_maintenance_thread(self):
        """Start maintenance thread to manage connections."""
        def maintain():
            while True:
                try:
                    self._maintain_connections()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    logger.error(f"Connection pool maintenance error: {e}")
                    time.sleep(30)
        
        maintenance_thread = threading.Thread(target=maintain, daemon=True)
        maintenance_thread.start()
    
    def _maintain_connections(self):
        """Maintain connection pool health."""
        current_time = time.time()
        
        with self.lock:
            # Remove old idle connections
            while len(self.idle_connections) > self.min_connections:
                conn = self.idle_connections.popleft()
                if current_time - self.connection_times[conn] > 300:  # 5 minutes idle
                    try:
                        self._close_connection(conn)
                    except Exception as e:
                        logger.error(f"Error closing connection: {e}")
                    finally:
                        if conn in self.connection_times:
                            del self.connection_times[conn]
                else:
                    # Put it back if not old enough
                    self.idle_connections.appendleft(conn)
                    break
            
            # Ensure minimum connections
            self._ensure_minimum_connections()
    
    def _close_connection(self, connection: Any):
        """Close a connection (override in subclass if needed)."""
        if hasattr(connection, 'close'):
            connection.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            return {
                "active_connections": len(self.active_connections),
                "idle_connections": len(self.idle_connections),
                "total_connections": len(self.active_connections) + len(self.idle_connections),
                "max_connections": self.max_connections,
                "min_connections": self.min_connections
            }


class PerformanceProfiler:
    """Performance profiling and measurement."""
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "error_count": 0
        })
        self.lock = threading.Lock()
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                return self._execute_with_profiling(func, operation_name, *args, **kwargs)
            return wrapper
        return decorator
    
    def _execute_with_profiling(self, func: Callable, operation_name: str, *args, **kwargs):
        """Execute function with performance profiling."""
        start_time = time.perf_counter()
        start_memory = 0
        start_cpu = 0
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB
                start_cpu = process.cpu_percent()
            except:
                pass
        
        success = True
        error = None
        result = None
        
        try:
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            success = False
            error = str(e)
            raise
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            end_memory = start_memory
            end_cpu = start_cpu
            
            if PSUTIL_AVAILABLE:
                try:
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    end_cpu = process.cpu_percent()
                except:
                    pass
            
            # Create metrics
            metrics = PerformanceMetrics(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                success=success,
                error=error,
                memory_usage_mb=end_memory - start_memory,
                cpu_usage_percent=end_cpu - start_cpu
            )
            
            self._record_metrics(metrics)
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self.lock:
            # Add to metrics list
            self.metrics.append(metrics)
            if len(self.metrics) > self.max_samples:
                self.metrics = self.metrics[-self.max_samples:]
            
            # Update operation stats
            stats = self.operation_stats[metrics.operation]
            stats["count"] += 1
            stats["total_time"] += metrics.duration_ms
            stats["min_time"] = min(stats["min_time"], metrics.duration_ms)
            stats["max_time"] = max(stats["max_time"], metrics.duration_ms)
            
            if not metrics.success:
                stats["error_count"] += 1
    
    def get_operation_stats(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics for operations."""
        with self.lock:
            if operation_name:
                if operation_name not in self.operation_stats:
                    return {}
                
                stats = self.operation_stats[operation_name].copy()
                if stats["count"] > 0:
                    stats["avg_time"] = stats["total_time"] / stats["count"]
                    stats["error_rate"] = stats["error_count"] / stats["count"]
                else:
                    stats["avg_time"] = 0
                    stats["error_rate"] = 0
                
                return stats
            else:
                # Return all operations
                result = {}
                for op_name, op_stats in self.operation_stats.items():
                    stats = op_stats.copy()
                    if stats["count"] > 0:
                        stats["avg_time"] = stats["total_time"] / stats["count"]
                        stats["error_rate"] = stats["error_count"] / stats["count"]
                    else:
                        stats["avg_time"] = 0
                        stats["error_rate"] = 0
                    result[op_name] = stats
                
                return result
    
    def get_recent_metrics(self, minutes: int = 10) -> List[PerformanceMetrics]:
        """Get metrics from recent time period."""
        cutoff_time = time.perf_counter() - (minutes * 60)
        
        with self.lock:
            return [m for m in self.metrics if m.start_time >= cutoff_time]


class AutoScaler:
    """Automatic scaling based on performance metrics."""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 20, 
                 scale_up_threshold: float = 80.0, scale_down_threshold: float = 20.0):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
        self.current_workers = min_workers
        self.worker_pool: List[Any] = []
        self.scaling_callbacks: List[Callable] = []
        self.metrics_history: deque = deque(maxlen=100)
        
        self.lock = threading.Lock()
        self.running = False
        self.scaling_thread = None
    
    def start(self):
        """Start auto-scaling monitoring."""
        self.running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        logger.info("Auto-scaler started")
    
    def stop(self):
        """Stop auto-scaling monitoring."""
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join()
        logger.info("Auto-scaler stopped")
    
    def add_scaling_callback(self, callback: Callable[[int], None]):
        """Add callback for scaling events."""
        self.scaling_callbacks.append(callback)
    
    def report_metrics(self, cpu_percent: float, memory_percent: float, 
                      queue_length: int, response_time_ms: float):
        """Report current system metrics."""
        metrics = {
            "timestamp": time.time(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "queue_length": queue_length,
            "response_time_ms": response_time_ms
        }
        
        with self.lock:
            self.metrics_history.append(metrics)
    
    def _scaling_loop(self):
        """Main scaling monitoring loop."""
        while self.running:
            try:
                self._evaluate_scaling()
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                time.sleep(30)
    
    def _evaluate_scaling(self):
        """Evaluate whether scaling is needed."""
        if len(self.metrics_history) < 5:
            return  # Need enough data points
        
        with self.lock:
            recent_metrics = list(self.metrics_history)[-10:]  # Last 10 data points
            
            # Calculate averages
            avg_cpu = sum(m["cpu_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_memory = sum(m["memory_percent"] for m in recent_metrics) / len(recent_metrics)
            avg_queue = sum(m["queue_length"] for m in recent_metrics) / len(recent_metrics)
            avg_response = sum(m["response_time_ms"] for m in recent_metrics) / len(recent_metrics)
            
            # Determine scaling need
            scale_score = self._calculate_scale_score(avg_cpu, avg_memory, avg_queue, avg_response)
            
            if scale_score > self.scale_up_threshold and self.current_workers < self.max_workers:
                self._scale_up()
            elif scale_score < self.scale_down_threshold and self.current_workers > self.min_workers:
                self._scale_down()
    
    def _calculate_scale_score(self, cpu: float, memory: float, 
                              queue_length: float, response_time: float) -> float:
        """Calculate scaling score based on metrics."""
        # Weighted scoring
        cpu_weight = 0.3
        memory_weight = 0.2
        queue_weight = 0.3
        response_weight = 0.2
        
        # Normalize queue length (assume max queue of 100)
        queue_score = min(queue_length / 100.0 * 100, 100)
        
        # Normalize response time (assume target of 1000ms)
        response_score = min(response_time / 1000.0 * 100, 100)
        
        total_score = (
            cpu * cpu_weight +
            memory * memory_weight +
            queue_score * queue_weight +
            response_score * response_weight
        )
        
        return total_score
    
    def _scale_up(self):
        """Scale up the number of workers."""
        new_worker_count = min(self.current_workers + 1, self.max_workers)
        if new_worker_count != self.current_workers:
            self.current_workers = new_worker_count
            logger.info(f"Scaling up to {self.current_workers} workers")
            
            for callback in self.scaling_callbacks:
                try:
                    callback(self.current_workers)
                except Exception as e:
                    logger.error(f"Scaling callback error: {e}")
    
    def _scale_down(self):
        """Scale down the number of workers."""
        new_worker_count = max(self.current_workers - 1, self.min_workers)
        if new_worker_count != self.current_workers:
            self.current_workers = new_worker_count
            logger.info(f"Scaling down to {self.current_workers} workers")
            
            for callback in self.scaling_callbacks:
                try:
                    callback(self.current_workers)
                except Exception as e:
                    logger.error(f"Scaling callback error: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        with self.lock:
            recent_metrics = list(self.metrics_history)[-1:] if self.metrics_history else [{}]
            current_metrics = recent_metrics[0] if recent_metrics else {}
            
            return {
                "current_workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "scale_up_threshold": self.scale_up_threshold,
                "scale_down_threshold": self.scale_down_threshold,
                "recent_metrics": current_metrics,
                "running": self.running
            }


# Global performance instances
global_cache = AdvancedCache(max_size=10000, max_memory_mb=256)
global_profiler = PerformanceProfiler()
global_autoscaler = AutoScaler()


def cached(ttl_seconds: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            
            # Try cache first
            result = global_cache.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            global_cache.set(cache_key, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator


def profiled(operation_name: Optional[str] = None):
    """Decorator for profiling function performance."""
    def decorator(func: Callable):
        op_name = operation_name or func.__name__
        return global_profiler.profile_operation(op_name)(func)
    return decorator