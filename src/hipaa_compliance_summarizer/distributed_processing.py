"""Distributed processing and horizontal scaling for HIPAA compliance system.

This module provides distributed processing capabilities including:
- Multi-node document processing
- Load balancing and work distribution
- Distributed task queues
- Node health monitoring and failover
- Auto-scaling based on workload
- Coordination and consensus mechanisms
"""

import logging
import multiprocessing as mp
import socket
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Lock, Thread
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Node status in the distributed system."""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    BUSY = "busy"
    DEGRADED = "degraded"
    FAILED = "failed"
    OFFLINE = "offline"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


@dataclass
class NodeInfo:
    """Information about a processing node."""

    node_id: str
    hostname: str
    ip_address: str
    port: int
    status: NodeStatus
    cpu_count: int
    memory_gb: float
    current_load: float
    last_heartbeat: datetime
    capabilities: List[str] = field(default_factory=list)
    active_tasks: int = 0
    max_tasks: int = 10
    task_success_rate: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "port": self.port,
            "status": self.status.value,
            "cpu_count": self.cpu_count,
            "memory_gb": self.memory_gb,
            "current_load": self.current_load,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "capabilities": self.capabilities,
            "active_tasks": self.active_tasks,
            "max_tasks": self.max_tasks,
            "task_success_rate": self.task_success_rate
        }


@dataclass
class DistributedTask:
    """Represents a task in the distributed processing system."""

    task_id: str
    task_type: str
    priority: TaskPriority
    payload: Dict[str, Any]
    created_at: datetime
    assigned_node: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    retries: int = 0
    max_retries: int = 3
    execution_timeout: int = 300  # 5 minutes
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "priority": self.priority.value,
            "payload": self.payload,
            "created_at": self.created_at.isoformat(),
            "assigned_node": self.assigned_node,
            "status": self.status.value,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "execution_timeout": self.execution_timeout,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "result": self.result,
            "error": self.error
        }


@dataclass
class WorkloadMetrics:
    """Workload and performance metrics for auto-scaling."""

    timestamp: datetime
    total_tasks_pending: int
    total_tasks_running: int
    total_tasks_completed_last_hour: int
    average_task_duration: float
    cluster_cpu_utilization: float
    cluster_memory_utilization: float
    nodes_healthy: int
    nodes_total: int
    queue_depth: int
    throughput_tasks_per_minute: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class LoadBalancer:
    """Intelligent load balancer for distributing tasks across nodes."""

    def __init__(self, strategy: str = "least_loaded"):
        """Initialize load balancer with specified strategy."""
        self.strategy = strategy
        self._task_assignments: Dict[str, str] = {}  # task_id -> node_id

    def select_node(self, task: DistributedTask, available_nodes: List[NodeInfo]) -> Optional[NodeInfo]:
        """Select the best node for a given task."""
        if not available_nodes:
            return None

        # Filter nodes that can handle this task
        capable_nodes = [
            node for node in available_nodes
            if (node.status == NodeStatus.HEALTHY and
                node.active_tasks < node.max_tasks and
                self._node_can_handle_task(node, task))
        ]

        if not capable_nodes:
            return None

        if self.strategy == "least_loaded":
            return min(capable_nodes, key=lambda n: n.current_load)
        elif self.strategy == "round_robin":
            return capable_nodes[hash(task.task_id) % len(capable_nodes)]
        elif self.strategy == "performance_weighted":
            return max(capable_nodes, key=lambda n: n.task_success_rate / (n.current_load + 0.1))
        else:
            # Default to least loaded
            return min(capable_nodes, key=lambda n: n.current_load)

    def _node_can_handle_task(self, node: NodeInfo, task: DistributedTask) -> bool:
        """Check if node can handle the specific task type."""
        # For now, assume all nodes can handle all task types
        # In a real implementation, this would check node capabilities
        return True


class DistributedTaskQueue:
    """Distributed task queue with priority support."""

    def __init__(self, max_size: int = 10000):
        """Initialize distributed task queue."""
        self.max_size = max_size
        self._queues: Dict[TaskPriority, deque] = {
            priority: deque() for priority in TaskPriority
        }
        self._tasks: Dict[str, DistributedTask] = {}
        self._lock = Lock()

    def enqueue(self, task: DistributedTask) -> bool:
        """Add task to the queue."""
        with self._lock:
            if len(self._tasks) >= self.max_size:
                logger.warning("Task queue is full, rejecting task")
                return False

            self._queues[task.priority].append(task)
            self._tasks[task.task_id] = task

            logger.debug(f"Enqueued task {task.task_id} with priority {task.priority.name}")
            return True

    def dequeue(self) -> Optional[DistributedTask]:
        """Get next task from queue (highest priority first)."""
        with self._lock:
            # Check queues in priority order
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                queue = self._queues[priority]
                if queue:
                    task = queue.popleft()
                    logger.debug(f"Dequeued task {task.task_id} with priority {priority.name}")
                    return task

            return None

    def get_task(self, task_id: str) -> Optional[DistributedTask]:
        """Get task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def update_task(self, task: DistributedTask) -> None:
        """Update task status."""
        with self._lock:
            self._tasks[task.task_id] = task

    def remove_task(self, task_id: str) -> bool:
        """Remove task from queue."""
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                # Remove from priority queue if still there
                queue = self._queues[task.priority]
                try:
                    queue.remove(task)
                except ValueError:
                    pass  # Task not in queue (already dequeued)
                return True
            return False

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._lock:
            stats = {
                "total_tasks": len(self._tasks),
                "by_priority": {},
                "by_status": defaultdict(int)
            }

            for priority, queue in self._queues.items():
                stats["by_priority"][priority.name] = len(queue)

            for task in self._tasks.values():
                stats["by_status"][task.status.value] += 1

            return stats


class ClusterCoordinator:
    """Coordinates distributed processing across multiple nodes."""

    def __init__(self, node_id: str = None, config: Optional[Dict[str, Any]] = None):
        """Initialize cluster coordinator."""
        self.node_id = node_id or str(uuid.uuid4())
        self.config = config or {}
        self._lock = Lock()

        # Cluster state
        self._nodes: Dict[str, NodeInfo] = {}
        self._is_leader = False
        self._leader_id: Optional[str] = None

        # Task management
        self._task_queue = DistributedTaskQueue()
        self._load_balancer = LoadBalancer(self.config.get('load_balancing_strategy', 'least_loaded'))

        # Metrics and monitoring
        self._workload_metrics: deque = deque(maxlen=1000)
        self._last_metrics_time = datetime.now()

        # Background processing
        self._coordinator_active = True
        self._start_coordinator_threads()

        # Initialize local node info
        self._local_node = self._create_local_node_info()
        self._nodes[self.node_id] = self._local_node

        logger.info(f"Cluster coordinator initialized with node ID: {self.node_id}")

    def _create_local_node_info(self) -> NodeInfo:
        """Create node info for the local node."""
        import psutil

        return NodeInfo(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            ip_address=self._get_local_ip(),
            port=self.config.get('node_port', 8080),
            status=NodeStatus.INITIALIZING,
            cpu_count=mp.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3),
            current_load=0.0,
            last_heartbeat=datetime.now(),
            capabilities=self.config.get('node_capabilities', ['phi_processing', 'document_analysis']),
            max_tasks=self.config.get('max_concurrent_tasks', min(mp.cpu_count() * 2, 20))
        )

    def _get_local_ip(self) -> str:
        """Get local IP address."""
        try:
            # Connect to external address to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    def submit_task(self, task_type: str, payload: Dict[str, Any],
                   priority: TaskPriority = TaskPriority.NORMAL,
                   max_retries: int = 3, timeout: int = 300) -> str:
        """Submit a task for distributed processing."""
        task = DistributedTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            priority=priority,
            payload=payload,
            created_at=datetime.now(),
            max_retries=max_retries,
            execution_timeout=timeout
        )

        if self._task_queue.enqueue(task):
            logger.info(f"Submitted task {task.task_id} of type {task_type}")
            return task.task_id
        else:
            raise RuntimeError("Failed to enqueue task - queue full")

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        task = self._task_queue.get_task(task_id)
        return task.to_dict() if task else None

    def register_node(self, node_info: NodeInfo) -> bool:
        """Register a new node in the cluster."""
        with self._lock:
            self._nodes[node_info.node_id] = node_info

        logger.info(f"Registered node {node_info.node_id} ({node_info.hostname})")
        return True

    def update_node_heartbeat(self, node_id: str, metrics: Optional[Dict[str, Any]] = None) -> bool:
        """Update node heartbeat and metrics."""
        with self._lock:
            if node_id in self._nodes:
                node = self._nodes[node_id]
                node.last_heartbeat = datetime.now()

                if metrics:
                    node.current_load = metrics.get('cpu_percent', node.current_load)
                    node.active_tasks = metrics.get('active_tasks', node.active_tasks)

                return True

        return False

    def _start_coordinator_threads(self) -> None:
        """Start background coordinator threads."""

        def heartbeat_monitor():
            """Monitor node heartbeats and handle failures."""
            while self._coordinator_active:
                try:
                    self._check_node_health()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error(f"Heartbeat monitor error: {e}")
                    time.sleep(10)

        def task_scheduler():
            """Schedule tasks to available nodes."""
            while self._coordinator_active:
                try:
                    self._schedule_tasks()
                    time.sleep(1)  # Check every second
                except Exception as e:
                    logger.error(f"Task scheduler error: {e}")
                    time.sleep(5)

        def metrics_collector():
            """Collect cluster metrics for auto-scaling."""
            while self._coordinator_active:
                try:
                    self._collect_workload_metrics()
                    time.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    logger.error(f"Metrics collector error: {e}")
                    time.sleep(60)

        self._heartbeat_thread = Thread(target=heartbeat_monitor, daemon=True)
        self._scheduler_thread = Thread(target=task_scheduler, daemon=True)
        self._metrics_thread = Thread(target=metrics_collector, daemon=True)

        self._heartbeat_thread.start()
        self._scheduler_thread.start()
        self._metrics_thread.start()

    def _check_node_health(self) -> None:
        """Check health of all registered nodes."""
        current_time = datetime.now()
        heartbeat_timeout = timedelta(seconds=30)

        with self._lock:
            for node_id, node in self._nodes.items():
                if node_id == self.node_id:
                    continue  # Skip local node

                if current_time - node.last_heartbeat > heartbeat_timeout:
                    if node.status != NodeStatus.OFFLINE:
                        logger.warning(f"Node {node_id} appears offline (last heartbeat: {node.last_heartbeat})")
                        node.status = NodeStatus.OFFLINE

                        # Reschedule tasks assigned to this node
                        self._reschedule_node_tasks(node_id)

    def _reschedule_node_tasks(self, failed_node_id: str) -> None:
        """Reschedule tasks from a failed node."""
        tasks_to_reschedule = []

        # Find tasks assigned to the failed node
        for task in self._task_queue._tasks.values():
            if task.assigned_node == failed_node_id and task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.PENDING
                task.assigned_node = None
                task.retries += 1
                tasks_to_reschedule.append(task)

        logger.info(f"Rescheduled {len(tasks_to_reschedule)} tasks from failed node {failed_node_id}")

    def _schedule_tasks(self) -> None:
        """Schedule pending tasks to available nodes."""
        # Get available nodes
        with self._lock:
            available_nodes = [
                node for node in self._nodes.values()
                if node.status in [NodeStatus.HEALTHY, NodeStatus.BUSY]
            ]

        if not available_nodes:
            return

        # Schedule pending tasks
        while True:
            task = self._task_queue.dequeue()
            if not task:
                break

            selected_node = self._load_balancer.select_node(task, available_nodes)
            if selected_node:
                task.assigned_node = selected_node.node_id
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                # Update node active tasks
                selected_node.active_tasks += 1

                self._task_queue.update_task(task)

                # In a real implementation, would send task to the selected node
                logger.debug(f"Assigned task {task.task_id} to node {selected_node.node_id}")
            else:
                # No available nodes, put task back in queue
                self._task_queue.enqueue(task)
                break

    def _collect_workload_metrics(self) -> None:
        """Collect workload metrics for monitoring and auto-scaling."""
        current_time = datetime.now()

        # Calculate metrics
        queue_stats = self._task_queue.get_queue_stats()

        with self._lock:
            healthy_nodes = sum(1 for node in self._nodes.values() if node.status == NodeStatus.HEALTHY)
            total_nodes = len(self._nodes)

            cluster_cpu = sum(node.current_load for node in self._nodes.values()) / max(total_nodes, 1)

            # Calculate tasks completed in last hour
            hour_ago = current_time - timedelta(hours=1)
            completed_last_hour = sum(
                1 for task in self._task_queue._tasks.values()
                if (task.status == TaskStatus.COMPLETED and
                    task.completed_at and task.completed_at > hour_ago)
            )

        metrics = WorkloadMetrics(
            timestamp=current_time,
            total_tasks_pending=queue_stats["by_status"].get("pending", 0),
            total_tasks_running=queue_stats["by_status"].get("running", 0),
            total_tasks_completed_last_hour=completed_last_hour,
            average_task_duration=self._calculate_average_task_duration(),
            cluster_cpu_utilization=cluster_cpu,
            cluster_memory_utilization=0.0,  # Would need to implement memory tracking
            nodes_healthy=healthy_nodes,
            nodes_total=total_nodes,
            queue_depth=queue_stats["total_tasks"],
            throughput_tasks_per_minute=completed_last_hour / 60.0
        )

        self._workload_metrics.append(metrics)

        # Check for auto-scaling triggers
        self._check_autoscale_triggers(metrics)

    def _calculate_average_task_duration(self) -> float:
        """Calculate average task duration for completed tasks."""
        completed_tasks = [
            task for task in self._task_queue._tasks.values()
            if (task.status == TaskStatus.COMPLETED and
                task.started_at and task.completed_at)
        ]

        if not completed_tasks:
            return 0.0

        total_duration = sum(
            (task.completed_at - task.started_at).total_seconds()
            for task in completed_tasks
        )

        return total_duration / len(completed_tasks)

    def _check_autoscale_triggers(self, metrics: WorkloadMetrics) -> None:
        """Check if auto-scaling should be triggered."""
        # Scale up conditions
        if (metrics.total_tasks_pending > 50 or
            metrics.cluster_cpu_utilization > 80 or
            metrics.queue_depth > 100):

            logger.info("Auto-scaling: Scale up conditions met")
            self._trigger_scale_up()

        # Scale down conditions
        elif (metrics.total_tasks_pending == 0 and
              metrics.cluster_cpu_utilization < 20 and
              metrics.nodes_healthy > 1):

            logger.info("Auto-scaling: Scale down conditions met")
            self._trigger_scale_down()

    def _trigger_scale_up(self) -> None:
        """Trigger scale-up operations."""
        # In a real implementation, would integrate with container orchestration
        # or cloud auto-scaling services
        logger.info("Scale-up triggered - would provision additional nodes")

    def _trigger_scale_down(self) -> None:
        """Trigger scale-down operations."""
        # In a real implementation, would safely decommission nodes
        logger.info("Scale-down triggered - would decommission excess nodes")

    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self._lock:
            node_stats = {
                "total": len(self._nodes),
                "by_status": defaultdict(int)
            }

            for node in self._nodes.values():
                node_stats["by_status"][node.status.value] += 1

        queue_stats = self._task_queue.get_queue_stats()

        latest_metrics = self._workload_metrics[-1] if self._workload_metrics else None

        return {
            "cluster_id": f"cluster_{self.node_id[:8]}",
            "coordinator_node": self.node_id,
            "is_leader": self._is_leader,
            "nodes": node_stats,
            "tasks": queue_stats,
            "latest_metrics": latest_metrics.to_dict() if latest_metrics else None,
            "uptime_seconds": (datetime.now() - self._local_node.last_heartbeat).total_seconds(),
            "last_updated": datetime.now().isoformat()
        }

    def get_nodes(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all nodes."""
        with self._lock:
            return {
                node_id: node.to_dict()
                for node_id, node in self._nodes.items()
            }

    def shutdown(self) -> None:
        """Shutdown the cluster coordinator."""
        self._coordinator_active = False

        # Wait for threads to finish
        if hasattr(self, '_heartbeat_thread'):
            self._heartbeat_thread.join(timeout=5)
        if hasattr(self, '_scheduler_thread'):
            self._scheduler_thread.join(timeout=5)
        if hasattr(self, '_metrics_thread'):
            self._metrics_thread.join(timeout=5)

        logger.info("Cluster coordinator shutdown completed")


# Task execution functions (would be implemented based on actual task types)
def execute_phi_processing_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute PHI processing task."""
    # Placeholder implementation
    time.sleep(1)  # Simulate processing
    return {
        "status": "success",
        "processed_documents": payload.get("document_count", 1),
        "phi_entities_found": 15,
        "execution_time": 1.0
    }


def execute_document_analysis_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Execute document analysis task."""
    # Placeholder implementation
    time.sleep(2)  # Simulate processing
    return {
        "status": "success",
        "analyzed_documents": payload.get("document_count", 1),
        "compliance_score": 0.95,
        "execution_time": 2.0
    }


# Task registry for distributed execution
TASK_REGISTRY = {
    "phi_processing": execute_phi_processing_task,
    "document_analysis": execute_document_analysis_task,
}


# Global coordinator instance
_global_coordinator: Optional[ClusterCoordinator] = None
_coordinator_lock = None  # Will be created when needed


def get_cluster_coordinator(config: Optional[Dict[str, Any]] = None) -> ClusterCoordinator:
    """Get or create global cluster coordinator."""
    global _global_coordinator, _coordinator_lock

    if _coordinator_lock is None:
        import threading
        _coordinator_lock = threading.Lock()

    with _coordinator_lock:
        if _global_coordinator is None:
            _global_coordinator = ClusterCoordinator(config=config)
        return _global_coordinator


def initialize_distributed_processing(config: Optional[Dict[str, Any]] = None) -> ClusterCoordinator:
    """Initialize distributed processing system."""
    return get_cluster_coordinator(config)


def submit_distributed_task(task_type: str, payload: Dict[str, Any],
                          priority: TaskPriority = TaskPriority.NORMAL) -> str:
    """Submit a task for distributed processing."""
    coordinator = get_cluster_coordinator()
    return coordinator.submit_task(task_type, payload, priority)


def get_distributed_cluster_status() -> Dict[str, Any]:
    """Get distributed cluster status."""
    coordinator = get_cluster_coordinator()
    return coordinator.get_cluster_status()
