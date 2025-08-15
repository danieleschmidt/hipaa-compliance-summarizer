"""Auto-scaling and resource management for HIPAA compliance system."""

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """Resource types for scaling."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    WORKERS = "workers"


class ScalingDirection(str, Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""

    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_percent: float = 0.0
    disk_free_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_workers: int = 0
    queue_size: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_available_gb": self.memory_available_gb,
            "disk_percent": self.disk_percent,
            "disk_free_gb": self.disk_free_gb,
            "network_bytes_sent": self.network_bytes_sent,
            "network_bytes_recv": self.network_bytes_recv,
            "active_workers": self.active_workers,
            "queue_size": self.queue_size,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class ScalingRule:
    """Auto-scaling rule configuration."""

    resource_type: ResourceType
    scale_up_threshold: float
    scale_down_threshold: float
    scale_up_amount: int = 1
    scale_down_amount: int = 1
    cooldown_seconds: int = 300
    min_value: int = 1
    max_value: int = 10
    enabled: bool = True

    def should_scale_up(self, current_value: float) -> bool:
        """Check if should scale up."""
        return self.enabled and current_value > self.scale_up_threshold

    def should_scale_down(self, current_value: float) -> bool:
        """Check if should scale down."""
        return self.enabled and current_value < self.scale_down_threshold


class AutoScaler:
    """Automatic scaling based on resource utilization."""

    def __init__(self):
        """Initialize auto-scaler."""
        self.scaling_rules: Dict[ResourceType, ScalingRule] = {}
        self.metrics_history: List[ResourceMetrics] = []
        self.scaling_history: List[Dict[str, Any]] = []
        self.last_scaling_actions: Dict[ResourceType, datetime] = {}

        self.monitoring_active = False
        self.monitor_thread = None
        self.check_interval_seconds = 30

    def get_scaling_recommendation(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get scaling recommendation based on current metrics."""
        cpu_usage = current_metrics.get('cpu_usage', 0)
        memory_usage = current_metrics.get('memory_usage', 0)
        queue_length = current_metrics.get('queue_length', 0)
        processing_time = current_metrics.get('processing_time', 0)
        
        action = 'none'
        reason = 'metrics within normal thresholds'
        
        # Simple scaling logic
        if cpu_usage > 80 or memory_usage > 85:
            action = 'scale_up'
            reason = f'High resource usage: CPU {cpu_usage}%, Memory {memory_usage}%'
        elif queue_length > 50:
            action = 'scale_up'
            reason = f'Large queue backlog: {queue_length} items'
        elif processing_time > 5000:  # 5 seconds
            action = 'scale_up'
            reason = f'Slow processing time: {processing_time}ms'
        elif cpu_usage < 20 and memory_usage < 30 and queue_length == 0:
            action = 'scale_down'
            reason = f'Low resource usage: CPU {cpu_usage}%, Memory {memory_usage}%'
            
        return {
            'action': action,
            'reason': reason,
            'current_metrics': current_metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Scaling callbacks
        self.scaling_callbacks: Dict[ResourceType, Callable] = {}

    def register_scaling_rule(self, rule: ScalingRule):
        """Register scaling rule.
        
        Args:
            rule: Scaling rule configuration
        """
        self.scaling_rules[rule.resource_type] = rule
        logger.info(f"Registered scaling rule for {rule.resource_type}: {rule.scale_up_threshold}% up, {rule.scale_down_threshold}% down")

    def register_scaling_callback(self, resource_type: ResourceType, callback: Callable):
        """Register scaling callback function.
        
        Args:
            resource_type: Resource type
            callback: Function to call for scaling actions
        """
        self.scaling_callbacks[resource_type] = callback
        logger.info(f"Registered scaling callback for {resource_type}")

    def start_monitoring(self, interval_seconds: int = 30):
        """Start resource monitoring and auto-scaling.
        
        Args:
            interval_seconds: Monitoring interval
        """
        self.check_interval_seconds = interval_seconds
        self.monitoring_active = True

        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        logger.info(f"Auto-scaling monitoring started with {interval_seconds}s interval")

    def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Auto-scaling monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                metrics = self._collect_resource_metrics()
                self.metrics_history.append(metrics)

                # Keep only recent metrics (last 24 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                self.metrics_history = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

                # Check scaling rules
                self._check_scaling_rules(metrics)

                time.sleep(self.check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in auto-scaling monitoring loop: {e}")
                time.sleep(5)

    def _collect_resource_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        metrics = ResourceMetrics()

        if HAS_PSUTIL:
            try:
                # CPU metrics
                metrics.cpu_percent = psutil.cpu_percent(interval=1)

                # Memory metrics
                memory = psutil.virtual_memory()
                metrics.memory_percent = memory.percent
                metrics.memory_available_gb = memory.available / (1024**3)

                # Disk metrics
                disk = psutil.disk_usage('/')
                metrics.disk_percent = disk.percent
                metrics.disk_free_gb = disk.free / (1024**3)

                # Network metrics
                network = psutil.net_io_counters()
                metrics.network_bytes_sent = network.bytes_sent
                metrics.network_bytes_recv = network.bytes_recv

            except Exception as e:
                logger.error(f"Error collecting resource metrics: {e}")
        else:
            # Fallback values when psutil is not available
            metrics.cpu_percent = 50.0
            metrics.memory_percent = 60.0
            metrics.memory_available_gb = 2.0
            metrics.disk_percent = 70.0
            metrics.disk_free_gb = 10.0
            metrics.network_bytes_sent = 0
            metrics.network_bytes_recv = 0

        return metrics

    def _check_scaling_rules(self, metrics: ResourceMetrics):
        """Check if any scaling rules should be triggered."""
        for resource_type, rule in self.scaling_rules.items():
            if not rule.enabled:
                continue

            # Check cooldown period
            if resource_type in self.last_scaling_actions:
                time_since_last = (datetime.utcnow() - self.last_scaling_actions[resource_type]).total_seconds()
                if time_since_last < rule.cooldown_seconds:
                    continue

            # Get current value for resource type
            current_value = self._get_resource_value(metrics, resource_type)
            if current_value is None:
                continue

            scaling_direction = ScalingDirection.NONE

            # Check scaling conditions
            if rule.should_scale_up(current_value):
                scaling_direction = ScalingDirection.UP
            elif rule.should_scale_down(current_value):
                scaling_direction = ScalingDirection.DOWN

            if scaling_direction != ScalingDirection.NONE:
                self._execute_scaling_action(resource_type, scaling_direction, rule, current_value)

    def _get_resource_value(self, metrics: ResourceMetrics, resource_type: ResourceType) -> Optional[float]:
        """Get current value for resource type."""
        if resource_type == ResourceType.CPU:
            return metrics.cpu_percent
        elif resource_type == ResourceType.MEMORY:
            return metrics.memory_percent
        elif resource_type == ResourceType.DISK:
            return metrics.disk_percent
        elif resource_type == ResourceType.WORKERS:
            return float(metrics.active_workers)
        elif resource_type == ResourceType.NETWORK:
            # Simple network utilization approximation
            return metrics.network_bytes_sent + metrics.network_bytes_recv
        else:
            return None

    def _execute_scaling_action(
        self,
        resource_type: ResourceType,
        direction: ScalingDirection,
        rule: ScalingRule,
        current_value: float
    ):
        """Execute scaling action."""
        action_info = {
            "resource_type": resource_type.value,
            "direction": direction.value,
            "current_value": current_value,
            "threshold": rule.scale_up_threshold if direction == ScalingDirection.UP else rule.scale_down_threshold,
            "timestamp": datetime.utcnow().isoformat(),
            "success": False
        }

        try:
            # Execute scaling callback if registered
            if resource_type in self.scaling_callbacks:
                callback = self.scaling_callbacks[resource_type]

                if direction == ScalingDirection.UP:
                    callback("scale_up", rule.scale_up_amount)
                else:
                    callback("scale_down", rule.scale_down_amount)

                action_info["success"] = True
                self.last_scaling_actions[resource_type] = datetime.utcnow()

                logger.info(
                    f"Scaling action executed: {resource_type.value} {direction.value} "
                    f"(current: {current_value:.2f})"
                )
            else:
                logger.warning(f"No scaling callback registered for {resource_type}")

        except Exception as e:
            logger.error(f"Scaling action failed for {resource_type}: {e}")
            action_info["error"] = str(e)

        self.scaling_history.append(action_info)

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        recent_actions = self.scaling_history[-10:] if self.scaling_history else []

        return {
            "monitoring_active": self.monitoring_active,
            "active_rules": len([r for r in self.scaling_rules.values() if r.enabled]),
            "total_rules": len(self.scaling_rules),
            "recent_metrics": [m.to_dict() for m in recent_metrics],
            "recent_scaling_actions": recent_actions,
            "last_check": datetime.utcnow().isoformat()
        }


class WorkerPool:
    """Dynamic worker pool with auto-scaling."""

    def __init__(
        self,
        initial_workers: int = 2,
        max_workers: int = 10,
        min_workers: int = 1,
        task_timeout_seconds: int = 300
    ):
        """Initialize worker pool.
        
        Args:
            initial_workers: Initial number of workers
            max_workers: Maximum number of workers
            min_workers: Minimum number of workers
            task_timeout_seconds: Task timeout
        """
        self.initial_workers = initial_workers
        self.max_workers = max_workers
        self.min_workers = min_workers
        self.task_timeout_seconds = task_timeout_seconds

        self.work_queue = queue.Queue()
        self.workers: List[threading.Thread] = []
        self.worker_active: List[bool] = []
        self.active = False

        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_processing_time = 0.0

        self._lock = threading.Lock()

    def start(self):
        """Start the worker pool."""
        self.active = True

        # Start initial workers
        for i in range(self.initial_workers):
            self._add_worker()

        logger.info(f"Worker pool started with {self.initial_workers} workers")

    def stop(self, timeout: int = 30):
        """Stop the worker pool.
        
        Args:
            timeout: Shutdown timeout in seconds
        """
        self.active = False

        # Add sentinel values to wake up workers
        for _ in self.workers:
            self.work_queue.put(None)

        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=timeout)

        logger.info("Worker pool stopped")

    def submit_task(self, task_func: Callable, *args, **kwargs) -> bool:
        """Submit task to worker pool.
        
        Args:
            task_func: Task function to execute
            *args: Task arguments
            **kwargs: Task keyword arguments
            
        Returns:
            True if task was submitted successfully
        """
        if not self.active:
            return False

        task = {
            "function": task_func,
            "args": args,
            "kwargs": kwargs,
            "submitted_at": datetime.utcnow()
        }

        try:
            self.work_queue.put(task, timeout=1)
            return True
        except queue.Full:
            logger.warning("Work queue is full, task rejected")
            return False

    def scale_workers(self, action: str, amount: int = 1):
        """Scale worker pool.
        
        Args:
            action: "scale_up" or "scale_down"
            amount: Number of workers to add/remove
        """
        with self._lock:
            if action == "scale_up":
                current_workers = len(self.workers)
                new_workers = min(current_workers + amount, self.max_workers)

                for _ in range(new_workers - current_workers):
                    self._add_worker()

                logger.info(f"Scaled up: {current_workers} -> {len(self.workers)} workers")

            elif action == "scale_down":
                current_workers = len(self.workers)
                new_workers = max(current_workers - amount, self.min_workers)

                # Mark workers for removal
                for i in range(current_workers - new_workers):
                    if i < len(self.worker_active):
                        self.worker_active[i] = False

                logger.info(f"Scaled down: {current_workers} -> {new_workers} workers")

    def _add_worker(self):
        """Add a new worker thread."""
        worker_id = len(self.workers)
        worker = threading.Thread(
            target=self._worker_loop,
            args=(worker_id,),
            daemon=True
        )

        self.workers.append(worker)
        self.worker_active.append(True)
        worker.start()

    def _worker_loop(self, worker_id: int):
        """Main worker loop."""
        logger.debug(f"Worker {worker_id} started")

        while self.active and worker_id < len(self.worker_active) and self.worker_active[worker_id]:
            try:
                # Get task from queue
                task = self.work_queue.get(timeout=5)

                if task is None:  # Sentinel value
                    break

                # Execute task
                start_time = time.time()

                try:
                    task["function"](*task["args"], **task["kwargs"])

                    processing_time = time.time() - start_time
                    self.tasks_completed += 1
                    self.total_processing_time += processing_time

                except Exception as e:
                    self.tasks_failed += 1
                    logger.error(f"Task failed in worker {worker_id}: {e}")

                finally:
                    self.work_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.debug(f"Worker {worker_id} stopped")

    def get_statistics(self) -> Dict[str, Any]:
        """Get worker pool statistics."""
        with self._lock:
            active_workers = sum(1 for active in self.worker_active if active)
            avg_processing_time = (
                self.total_processing_time / self.tasks_completed
                if self.tasks_completed > 0 else 0.0
            )

            return {
                "active_workers": active_workers,
                "total_workers": len(self.workers),
                "queue_size": self.work_queue.qsize(),
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "success_rate": (
                    self.tasks_completed / (self.tasks_completed + self.tasks_failed)
                    if (self.tasks_completed + self.tasks_failed) > 0 else 0.0
                ),
                "average_processing_time_seconds": avg_processing_time,
                "min_workers": self.min_workers,
                "max_workers": self.max_workers
            }


def setup_auto_scaling(worker_pool: WorkerPool) -> AutoScaler:
    """Setup auto-scaling with default configuration.
    
    Args:
        worker_pool: Worker pool to scale
        
    Returns:
        Configured auto-scaler
    """
    auto_scaler = AutoScaler()

    # CPU-based scaling rule
    cpu_rule = ScalingRule(
        resource_type=ResourceType.CPU,
        scale_up_threshold=70.0,
        scale_down_threshold=20.0,
        scale_up_amount=2,
        scale_down_amount=1,
        cooldown_seconds=300,
        min_value=1,
        max_value=10
    )
    auto_scaler.register_scaling_rule(cpu_rule)

    # Memory-based scaling rule
    memory_rule = ScalingRule(
        resource_type=ResourceType.MEMORY,
        scale_up_threshold=80.0,
        scale_down_threshold=30.0,
        scale_up_amount=1,
        scale_down_amount=1,
        cooldown_seconds=300,
        min_value=1,
        max_value=8
    )
    auto_scaler.register_scaling_rule(memory_rule)

    # Register scaling callbacks
    auto_scaler.register_scaling_callback(
        ResourceType.CPU,
        worker_pool.scale_workers
    )
    auto_scaler.register_scaling_callback(
        ResourceType.MEMORY,
        worker_pool.scale_workers
    )

    logger.info("Auto-scaling setup completed")
    return auto_scaler


# Global instances for easy access
global_worker_pool = None
global_auto_scaler = None


def initialize_scaling_infrastructure():
    """Initialize global scaling infrastructure."""
    global global_worker_pool, global_auto_scaler

    if global_worker_pool is None:
        global_worker_pool = WorkerPool(
            initial_workers=2,
            max_workers=10,
            min_workers=1
        )
        global_worker_pool.start()

    if global_auto_scaler is None:
        global_auto_scaler = setup_auto_scaling(global_worker_pool)
        global_auto_scaler.start_monitoring()

    logger.info("Global scaling infrastructure initialized")


def get_scaling_status() -> Dict[str, Any]:
    """Get overall scaling status."""
    if global_worker_pool is None or global_auto_scaler is None:
        return {"message": "Scaling infrastructure not initialized"}

    return {
        "worker_pool": global_worker_pool.get_statistics(),
        "auto_scaler": global_auto_scaler.get_scaling_status()
    }
