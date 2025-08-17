"""Auto-scaling system for dynamic resource management in HIPAA compliance processing."""

import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Callable, Dict, List, Optional, Tuple

from .constants import PERFORMANCE_LIMITS
from .monitoring import PerformanceMonitor

logger = logging.getLogger(__name__)


class ScalingMetric(str, Enum):
    """Metrics used for auto-scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    PROCESSING_TIME = "processing_time"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"


class ScalingDirection(str, Enum):
    """Direction of scaling operations."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_CHANGE = "no_change"


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""
    metric: ScalingMetric
    threshold_up: float
    threshold_down: float
    scale_up_amount: int
    scale_down_amount: int
    cooldown_seconds: int
    weight: float = 1.0


@dataclass
class ScalingEvent:
    """Record of a scaling event."""
    timestamp: float
    direction: ScalingDirection
    metric: ScalingMetric
    metric_value: float
    threshold: float
    old_capacity: int
    new_capacity: int
    reason: str


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    disk_io_percent: float
    queue_length: int
    active_workers: int
    avg_processing_time_ms: float
    throughput_per_minute: float
    error_rate: float


class WorkerPool:
    """Dynamic worker pool with auto-scaling capabilities."""

    def __init__(self, min_workers: int = 1, max_workers: int = None,
                 pool_type: str = "thread"):
        """Initialize worker pool.
        
        Args:
            min_workers: Minimum number of workers
            max_workers: Maximum number of workers (defaults to CPU count * 4)
            pool_type: Type of pool ("thread" or "process")
        """
        self.min_workers = max(1, min_workers)
        cpu_count = psutil.cpu_count() if HAS_PSUTIL else 4
        self.max_workers = max_workers or min(cpu_count * 4, PERFORMANCE_LIMITS.MAX_CONCURRENT_JOBS)
        self.pool_type = pool_type

        # Worker management
        self.current_workers = self.min_workers
        self._executor: Optional[ThreadPoolExecutor] = None
        self._task_queue = Queue()
        self._results_queue = Queue()
        self._active_tasks = 0
        self._completed_tasks = 0
        self._failed_tasks = 0

        # Synchronization
        self._lock = threading.RLock()
        self._shutdown = False

        # Initialize executor
        self._recreate_executor()

    def _recreate_executor(self):
        """Recreate executor with current worker count."""
        with self._lock:
            if self._executor:
                self._executor.shutdown(wait=False)

            if self.pool_type == "thread":
                self._executor = ThreadPoolExecutor(max_workers=self.current_workers)
            else:
                self._executor = ProcessPoolExecutor(max_workers=self.current_workers)

            logger.info(f"Recreated {self.pool_type} pool with {self.current_workers} workers")

    def submit_task(self, func: Callable, *args, **kwargs):
        """Submit task to worker pool."""
        if self._shutdown:
            raise RuntimeError("Worker pool is shutdown")

        with self._lock:
            self._active_tasks += 1
            future = self._executor.submit(self._wrapped_task, func, *args, **kwargs)
            return future

    def _wrapped_task(self, func: Callable, *args, **kwargs):
        """Wrapper for tracking task execution."""
        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._completed_tasks += 1
                self._active_tasks -= 1
            return result
        except Exception as e:
            with self._lock:
                self._failed_tasks += 1
                self._active_tasks -= 1
            raise e

    def scale_workers(self, new_worker_count: int) -> bool:
        """Scale worker pool to new size."""
        new_worker_count = max(self.min_workers, min(self.max_workers, new_worker_count))

        if new_worker_count == self.current_workers:
            return False

        old_count = self.current_workers
        self.current_workers = new_worker_count
        self._recreate_executor()

        logger.info(f"Scaled worker pool from {old_count} to {new_worker_count} workers")
        return True

    def get_metrics(self) -> Dict[str, int]:
        """Get current worker pool metrics."""
        with self._lock:
            return {
                "current_workers": self.current_workers,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers,
                "active_tasks": self._active_tasks,
                "completed_tasks": self._completed_tasks,
                "failed_tasks": self._failed_tasks,
                "queue_length": self._task_queue.qsize()
            }

    def shutdown(self, wait: bool = True):
        """Shutdown worker pool."""
        self._shutdown = True
        if self._executor:
            self._executor.shutdown(wait=wait)


class AutoScaler:
    """Intelligent auto-scaling system for HIPAA processing workloads."""

    def __init__(self, performance_monitor: Optional[PerformanceMonitor] = None,
                 worker_pool: Optional[WorkerPool] = None):
        """Initialize auto-scaler.
        
        Args:
            performance_monitor: Performance monitoring system
            worker_pool: Worker pool to scale
        """
        self.performance_monitor = performance_monitor
        self.worker_pool = worker_pool or WorkerPool()

        # Scaling configuration
        self.scaling_rules: List[ScalingRule] = []
        self.scaling_history: List[ScalingEvent] = []
        self.last_scaling_time = 0
        self.default_cooldown = 60  # seconds

        # Monitoring
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        self._monitoring_interval = 10  # seconds

        # Metrics tracking
        self._metrics_history: List[Tuple[float, ResourceMetrics]] = []
        self._max_history_size = 100

        # Register default scaling rules
        self._register_default_rules()

    def _register_default_rules(self):
        """Register default auto-scaling rules."""
        self.scaling_rules = [
            ScalingRule(
                metric=ScalingMetric.CPU_UTILIZATION,
                threshold_up=80.0,
                threshold_down=30.0,
                scale_up_amount=2,
                scale_down_amount=1,
                cooldown_seconds=120,
                weight=1.0
            ),
            ScalingRule(
                metric=ScalingMetric.MEMORY_UTILIZATION,
                threshold_up=85.0,
                threshold_down=40.0,
                scale_up_amount=1,
                scale_down_amount=1,
                cooldown_seconds=180,
                weight=0.8
            ),
            ScalingRule(
                metric=ScalingMetric.QUEUE_LENGTH,
                threshold_up=10.0,
                threshold_down=2.0,
                scale_up_amount=3,
                scale_down_amount=1,
                cooldown_seconds=60,
                weight=1.2
            ),
            ScalingRule(
                metric=ScalingMetric.PROCESSING_TIME,
                threshold_up=5000.0,  # 5 seconds
                threshold_down=1000.0,  # 1 second
                scale_up_amount=2,
                scale_down_amount=1,
                cooldown_seconds=90,
                weight=0.9
            )
        ]

    def add_scaling_rule(self, rule: ScalingRule):
        """Add custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule for {rule.metric.value}")

    def start_monitoring(self):
        """Start auto-scaling monitoring."""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            logger.warning("Auto-scaling monitoring already running")
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started auto-scaling monitoring")

    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Stopped auto-scaling monitoring")

    def _monitoring_loop(self):
        """Main monitoring loop for auto-scaling."""
        while not self._stop_monitoring.wait(self._monitoring_interval):
            try:
                # Collect current metrics
                metrics = self._collect_metrics()
                current_time = time.time()

                # Store metrics history
                self._metrics_history.append((current_time, metrics))
                if len(self._metrics_history) > self._max_history_size:
                    self._metrics_history.pop(0)

                # Evaluate scaling decisions
                scaling_decision = self._evaluate_scaling_decision(metrics)

                # Apply scaling if needed
                if scaling_decision != ScalingDirection.NO_CHANGE:
                    self._apply_scaling_decision(scaling_decision, metrics)

            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring: {e}")

    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        # System metrics
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk_io = psutil.disk_io_counters()
        else:
            # Fallback values when psutil is not available
            cpu_percent = 50.0  # Assume moderate CPU usage
            memory = type('MockMemory', (), {'percent': 60.0})()
            disk_io = None

        # Worker pool metrics
        pool_metrics = self.worker_pool.get_metrics()

        # Performance monitor metrics
        avg_processing_time = 0.0
        throughput = 0.0
        error_rate = 0.0

        if self.performance_monitor:
            processing_metrics = self.performance_monitor.get_processing_metrics()
            avg_processing_time = processing_metrics.avg_processing_time * 1000  # Convert to ms
            throughput = processing_metrics.throughput_docs_per_minute
            error_rate = processing_metrics.error_rate

        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            disk_io_percent=min(100.0, (disk_io.read_bytes + disk_io.write_bytes) / (1024*1024) if disk_io else 0),
            queue_length=pool_metrics["queue_length"],
            active_workers=pool_metrics["active_tasks"],
            avg_processing_time_ms=avg_processing_time,
            throughput_per_minute=throughput,
            error_rate=error_rate
        )

    def _evaluate_scaling_decision(self, metrics: ResourceMetrics) -> ScalingDirection:
        """Evaluate whether scaling is needed based on current metrics."""
        current_time = time.time()

        # Check cooldown period
        if current_time - self.last_scaling_time < self.default_cooldown:
            return ScalingDirection.NO_CHANGE

        scale_up_score = 0.0
        scale_down_score = 0.0

        for rule in self.scaling_rules:
            # Check if rule is in cooldown
            rule_last_scaling = self._get_last_scaling_time_for_metric(rule.metric)
            if current_time - rule_last_scaling < rule.cooldown_seconds:
                continue

            metric_value = self._get_metric_value(metrics, rule.metric)

            if metric_value > rule.threshold_up:
                scale_up_score += rule.weight
            elif metric_value < rule.threshold_down:
                scale_down_score += rule.weight

        # Make scaling decision
        if scale_up_score > scale_down_score and scale_up_score > 0:
            return ScalingDirection.SCALE_UP
        elif scale_down_score > scale_up_score and scale_down_score > 0:
            return ScalingDirection.SCALE_DOWN
        else:
            return ScalingDirection.NO_CHANGE

    def _get_metric_value(self, metrics: ResourceMetrics, metric_type: ScalingMetric) -> float:
        """Get metric value from metrics object."""
        metric_map = {
            ScalingMetric.CPU_UTILIZATION: metrics.cpu_percent,
            ScalingMetric.MEMORY_UTILIZATION: metrics.memory_percent,
            ScalingMetric.QUEUE_LENGTH: float(metrics.queue_length),
            ScalingMetric.PROCESSING_TIME: metrics.avg_processing_time_ms,
            ScalingMetric.THROUGHPUT: metrics.throughput_per_minute,
            ScalingMetric.ERROR_RATE: metrics.error_rate * 100  # Convert to percentage
        }
        return metric_map.get(metric_type, 0.0)

    def _get_last_scaling_time_for_metric(self, metric: ScalingMetric) -> float:
        """Get the last time scaling occurred for a specific metric."""
        for event in reversed(self.scaling_history):
            if event.metric == metric:
                return event.timestamp
        return 0.0

    def _apply_scaling_decision(self, direction: ScalingDirection, metrics: ResourceMetrics):
        """Apply scaling decision to worker pool."""
        current_workers = self.worker_pool.current_workers

        if direction == ScalingDirection.SCALE_UP:
            # Find the rule with the highest threshold violation
            scale_amount = self._calculate_scale_up_amount(metrics)
            new_workers = min(self.worker_pool.max_workers, current_workers + scale_amount)
        else:  # SCALE_DOWN
            scale_amount = self._calculate_scale_down_amount(metrics)
            new_workers = max(self.worker_pool.min_workers, current_workers - scale_amount)

        # Apply scaling
        if self.worker_pool.scale_workers(new_workers):
            # Record scaling event
            primary_metric, metric_value, threshold = self._get_primary_scaling_trigger(metrics, direction)

            event = ScalingEvent(
                timestamp=time.time(),
                direction=direction,
                metric=primary_metric,
                metric_value=metric_value,
                threshold=threshold,
                old_capacity=current_workers,
                new_capacity=new_workers,
                reason=f"{primary_metric.value}={metric_value:.1f} {'>' if direction == ScalingDirection.SCALE_UP else '<'} {threshold}"
            )

            self.scaling_history.append(event)
            self.last_scaling_time = event.timestamp

            # Limit history size
            if len(self.scaling_history) > 1000:
                self.scaling_history = self.scaling_history[-500:]

            logger.info(f"Auto-scaling {direction.value}: {current_workers} -> {new_workers} workers ({event.reason})")

    def _calculate_scale_up_amount(self, metrics: ResourceMetrics) -> int:
        """Calculate how many workers to add when scaling up."""
        max_scale_amount = 1

        for rule in self.scaling_rules:
            metric_value = self._get_metric_value(metrics, rule.metric)
            if metric_value > rule.threshold_up:
                # Scale more aggressively for higher threshold violations
                violation_ratio = metric_value / rule.threshold_up
                scaled_amount = int(rule.scale_up_amount * min(violation_ratio, 2.0))
                max_scale_amount = max(max_scale_amount, scaled_amount)

        return max_scale_amount

    def _calculate_scale_down_amount(self, metrics: ResourceMetrics) -> int:
        """Calculate how many workers to remove when scaling down."""
        max_scale_amount = 1

        for rule in self.scaling_rules:
            metric_value = self._get_metric_value(metrics, rule.metric)
            if metric_value < rule.threshold_down:
                max_scale_amount = max(max_scale_amount, rule.scale_down_amount)

        return max_scale_amount

    def _get_primary_scaling_trigger(self, metrics: ResourceMetrics,
                                   direction: ScalingDirection) -> Tuple[ScalingMetric, float, float]:
        """Get the primary metric that triggered scaling."""
        max_violation = 0.0
        primary_metric = ScalingMetric.CPU_UTILIZATION
        metric_value = 0.0
        threshold = 0.0

        for rule in self.scaling_rules:
            value = self._get_metric_value(metrics, rule.metric)

            if direction == ScalingDirection.SCALE_UP and value > rule.threshold_up:
                violation = (value - rule.threshold_up) / rule.threshold_up * rule.weight
                if violation > max_violation:
                    max_violation = violation
                    primary_metric = rule.metric
                    metric_value = value
                    threshold = rule.threshold_up
            elif direction == ScalingDirection.SCALE_DOWN and value < rule.threshold_down:
                violation = (rule.threshold_down - value) / rule.threshold_down * rule.weight
                if violation > max_violation:
                    max_violation = violation
                    primary_metric = rule.metric
                    metric_value = value
                    threshold = rule.threshold_down

        return primary_metric, metric_value, threshold

    def get_scaling_status(self) -> Dict[str, any]:
        """Get current auto-scaling status."""
        current_metrics = self._collect_metrics()

        recent_events = [
            {
                "timestamp": event.timestamp,
                "direction": event.direction.value,
                "metric": event.metric.value,
                "reason": event.reason,
                "old_capacity": event.old_capacity,
                "new_capacity": event.new_capacity
            }
            for event in self.scaling_history[-10:]
        ]

        return {
            "current_workers": self.worker_pool.current_workers,
            "min_workers": self.worker_pool.min_workers,
            "max_workers": self.worker_pool.max_workers,
            "monitoring_active": self._monitoring_thread and self._monitoring_thread.is_alive(),
            "last_scaling_time": self.last_scaling_time,
            "current_metrics": {
                "cpu_percent": current_metrics.cpu_percent,
                "memory_percent": current_metrics.memory_percent,
                "queue_length": current_metrics.queue_length,
                "active_workers": current_metrics.active_workers,
                "avg_processing_time_ms": current_metrics.avg_processing_time_ms,
                "throughput_per_minute": current_metrics.throughput_per_minute,
                "error_rate": current_metrics.error_rate
            },
            "scaling_rules": [
                {
                    "metric": rule.metric.value,
                    "threshold_up": rule.threshold_up,
                    "threshold_down": rule.threshold_down,
                    "cooldown_seconds": rule.cooldown_seconds
                }
                for rule in self.scaling_rules
            ],
            "recent_scaling_events": recent_events
        }

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_monitoring()
        if self.worker_pool:
            self.worker_pool.shutdown()


def initialize_auto_scaling(performance_monitor: Optional[PerformanceMonitor] = None,
                          min_workers: int = 1,
                          max_workers: Optional[int] = None) -> AutoScaler:
    """Initialize auto-scaling system."""
    worker_pool = WorkerPool(min_workers=min_workers, max_workers=max_workers)
    auto_scaler = AutoScaler(performance_monitor=performance_monitor, worker_pool=worker_pool)
    auto_scaler.start_monitoring()

    logger.info(f"Auto-scaling initialized with {min_workers}-{max_workers or 'auto'} workers")
    return auto_scaler


__all__ = [
    "ScalingMetric",
    "ScalingDirection",
    "ScalingRule",
    "ScalingEvent",
    "ResourceMetrics",
    "WorkerPool",
    "AutoScaler",
    "initialize_auto_scaling"
]
