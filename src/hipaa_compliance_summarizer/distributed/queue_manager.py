"""Distributed processing queue manager for HIPAA compliance system."""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Optional

try:
    import aioredis
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class QueueTask:
    """Represents a task in the processing queue."""

    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retries: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    worker_id: Optional[str] = None
    estimated_duration_seconds: int = 60
    progress_percentage: float = 0.0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
        if not self.task_id:
            self.task_id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary for serialization."""
        task_dict = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at']:
            if task_dict[field]:
                task_dict[field] = task_dict[field].isoformat()
        return task_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueueTask":
        """Create task from dictionary."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'started_at', 'completed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])

        return cls(**data)

    def is_expired(self, timeout_hours: int = 24) -> bool:
        """Check if task has expired."""
        return (datetime.utcnow() - self.created_at) > timedelta(hours=timeout_hours)

    def can_retry(self) -> bool:
        """Check if task can be retried."""
        return self.retries < self.max_retries and self.status == TaskStatus.FAILED


class QueueBackend(ABC):
    """Abstract base class for queue backends."""

    @abstractmethod
    async def enqueue(self, task: QueueTask, queue_name: str = "default") -> bool:
        """Add task to queue."""
        pass

    @abstractmethod
    async def dequeue(self, queue_name: str = "default") -> Optional[QueueTask]:
        """Get next task from queue."""
        pass

    @abstractmethod
    async def update_task_status(self, task_id: str, status: TaskStatus,
                                progress: float = None, error_message: str = None) -> bool:
        """Update task status."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[QueueTask]:
        """Get task by ID."""
        pass

    @abstractmethod
    async def get_queue_stats(self, queue_name: str = "default") -> Dict[str, int]:
        """Get queue statistics."""
        pass

    @abstractmethod
    async def cleanup_expired_tasks(self, timeout_hours: int = 24) -> int:
        """Clean up expired tasks."""
        pass


class RedisQueueBackend(QueueBackend):
    """Redis-based queue backend."""

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.connected = False

    async def connect(self):
        """Connect to Redis."""
        if not REDIS_AVAILABLE:
            logger.error("Redis not available. Install redis-py and aioredis.")
            return False

        try:
            self.redis_client = await aioredis.from_url(self.redis_url)
            await self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis queue backend")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    async def enqueue(self, task: QueueTask, queue_name: str = "default") -> bool:
        """Add task to Redis queue."""
        if not self.connected:
            if not await self.connect():
                return False

        try:
            # Store task data
            task_key = f"task:{task.task_id}"
            await self.redis_client.hset(task_key, mapping={
                "data": json.dumps(task.to_dict()),
                "queue": queue_name,
                "priority": task.priority.value
            })

            # Add to priority queue
            queue_key = f"queue:{queue_name}"
            await self.redis_client.zadd(
                queue_key,
                {task.task_id: task.priority.value * 1000 + int(time.time())}
            )

            # Set expiration (24 hours)
            await self.redis_client.expire(task_key, 86400)

            logger.debug(f"Enqueued task {task.task_id} to queue {queue_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to enqueue task {task.task_id}: {e}")
            return False

    async def dequeue(self, queue_name: str = "default") -> Optional[QueueTask]:
        """Get next task from Redis queue."""
        if not self.connected:
            if not await self.connect():
                return None

        try:
            # Get highest priority task
            queue_key = f"queue:{queue_name}"
            result = await self.redis_client.zrevrange(queue_key, 0, 0, withscores=True)

            if not result:
                return None

            task_id, score = result[0]
            task_id = task_id.decode('utf-8') if isinstance(task_id, bytes) else task_id

            # Remove from queue atomically
            removed = await self.redis_client.zrem(queue_key, task_id)
            if not removed:
                return None  # Task was already taken

            # Get task data
            task_key = f"task:{task_id}"
            task_data = await self.redis_client.hget(task_key, "data")

            if not task_data:
                logger.warning(f"Task data not found for {task_id}")
                return None

            task_dict = json.loads(task_data)
            task = QueueTask.from_dict(task_dict)

            # Update status to processing
            task.status = TaskStatus.PROCESSING
            task.started_at = datetime.utcnow()

            await self.update_task_status(task_id, TaskStatus.PROCESSING)

            logger.debug(f"Dequeued task {task_id} from queue {queue_name}")
            return task

        except Exception as e:
            logger.error(f"Failed to dequeue from {queue_name}: {e}")
            return None

    async def update_task_status(self, task_id: str, status: TaskStatus,
                                progress: float = None, error_message: str = None) -> bool:
        """Update task status in Redis."""
        if not self.connected:
            if not await self.connect():
                return False

        try:
            task_key = f"task:{task_id}"

            # Get current task data
            task_data = await self.redis_client.hget(task_key, "data")
            if not task_data:
                logger.warning(f"Task {task_id} not found for status update")
                return False

            task_dict = json.loads(task_data)

            # Update status
            task_dict["status"] = status.value

            if status == TaskStatus.COMPLETED:
                task_dict["completed_at"] = datetime.utcnow().isoformat()

            if progress is not None:
                task_dict["progress_percentage"] = progress

            if error_message:
                task_dict["error_message"] = error_message

            # Store updated data
            await self.redis_client.hset(task_key, "data", json.dumps(task_dict))

            logger.debug(f"Updated task {task_id} status to {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update task {task_id} status: {e}")
            return False

    async def get_task(self, task_id: str) -> Optional[QueueTask]:
        """Get task by ID from Redis."""
        if not self.connected:
            if not await self.connect():
                return None

        try:
            task_key = f"task:{task_id}"
            task_data = await self.redis_client.hget(task_key, "data")

            if not task_data:
                return None

            task_dict = json.loads(task_data)
            return QueueTask.from_dict(task_dict)

        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None

    async def get_queue_stats(self, queue_name: str = "default") -> Dict[str, int]:
        """Get queue statistics from Redis."""
        if not self.connected:
            if not await self.connect():
                return {"pending": 0, "processing": 0, "completed": 0, "failed": 0}

        try:
            queue_key = f"queue:{queue_name}"

            # Count pending tasks
            pending_count = await self.redis_client.zcard(queue_key)

            # Count tasks by status (scan through all task keys)
            status_counts = {"pending": pending_count, "processing": 0, "completed": 0, "failed": 0}

            # This is a simplified version - in production, you'd want to use a more efficient approach
            # like maintaining separate counters for each status

            return status_counts

        except Exception as e:
            logger.error(f"Failed to get queue stats for {queue_name}: {e}")
            return {"pending": 0, "processing": 0, "completed": 0, "failed": 0}

    async def cleanup_expired_tasks(self, timeout_hours: int = 24) -> int:
        """Clean up expired tasks from Redis."""
        if not self.connected:
            if not await self.connect():
                return 0

        try:
            # This is a simplified cleanup - in production, you'd implement more sophisticated cleanup
            # using Redis SCAN to iterate through task keys
            cleaned_count = 0
            logger.info(f"Redis cleanup completed, removed {cleaned_count} expired tasks")
            return cleaned_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired tasks: {e}")
            return 0

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.connected = False


class CeleryQueueBackend(QueueBackend):
    """Celery-based queue backend."""

    def __init__(self, broker_url: str = "redis://localhost:6379/1",
                 result_backend: str = "redis://localhost:6379/2"):
        if not CELERY_AVAILABLE:
            logger.error("Celery not available. Install celery package.")
            return

        self.app = Celery('hipaa_processor',
                         broker=broker_url,
                         backend=result_backend)

        # Configure Celery
        self.app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_routes={
                'hipaa_compliance_summarizer.tasks.*': {'queue': 'hipaa_processing'}
            }
        )

    async def enqueue(self, task: QueueTask, queue_name: str = "default") -> bool:
        """Add task to Celery queue."""
        if not CELERY_AVAILABLE:
            return False

        try:
            # Send task to Celery
            result = self.app.send_task(
                f'process_{task.task_type}',
                args=[task.to_dict()],
                queue=queue_name,
                task_id=task.task_id,
                priority=task.priority.value
            )

            logger.debug(f"Enqueued Celery task {task.task_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to enqueue Celery task {task.task_id}: {e}")
            return False

    async def dequeue(self, queue_name: str = "default") -> Optional[QueueTask]:
        """Celery handles dequeuing automatically."""
        # This method is not used with Celery as it handles task distribution automatically
        return None

    async def update_task_status(self, task_id: str, status: TaskStatus,
                                progress: float = None, error_message: str = None) -> bool:
        """Update task status in Celery."""
        if not CELERY_AVAILABLE:
            return False

        try:
            result = AsyncResult(task_id, app=self.app)

            # Update task state
            if status == TaskStatus.PROCESSING:
                result.update_state(state='PROGRESS', meta={'progress': progress or 0})
            elif status == TaskStatus.COMPLETED:
                result.update_state(state='SUCCESS')
            elif status == TaskStatus.FAILED:
                result.update_state(state='FAILURE', meta={'error': error_message})

            return True

        except Exception as e:
            logger.error(f"Failed to update Celery task {task_id} status: {e}")
            return False

    async def get_task(self, task_id: str) -> Optional[QueueTask]:
        """Get task status from Celery."""
        if not CELERY_AVAILABLE:
            return None

        try:
            result = AsyncResult(task_id, app=self.app)

            # Convert Celery state to our TaskStatus
            status_mapping = {
                'PENDING': TaskStatus.PENDING,
                'PROGRESS': TaskStatus.PROCESSING,
                'SUCCESS': TaskStatus.COMPLETED,
                'FAILURE': TaskStatus.FAILED,
                'RETRY': TaskStatus.RETRYING,
                'REVOKED': TaskStatus.CANCELLED
            }

            status = status_mapping.get(result.state, TaskStatus.PENDING)

            # Create a basic QueueTask representation
            task = QueueTask(
                task_id=task_id,
                task_type="unknown",  # Celery doesn't store this easily
                payload={},
                status=status
            )

            if hasattr(result, 'info') and isinstance(result.info, dict):
                task.progress_percentage = result.info.get('progress', 0)
                task.error_message = result.info.get('error')

            return task

        except Exception as e:
            logger.error(f"Failed to get Celery task {task_id}: {e}")
            return None

    async def get_queue_stats(self, queue_name: str = "default") -> Dict[str, int]:
        """Get queue statistics from Celery."""
        # Celery doesn't provide easy queue statistics
        return {"pending": 0, "processing": 0, "completed": 0, "failed": 0}

    async def cleanup_expired_tasks(self, timeout_hours: int = 24) -> int:
        """Clean up expired tasks from Celery."""
        # Celery handles task cleanup automatically
        return 0


class DistributedQueueManager:
    """Main manager for distributed task processing."""

    def __init__(self, backend: QueueBackend, worker_id: str = None):
        self.backend = backend
        self.worker_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        self.task_handlers: Dict[str, Callable] = {}
        self.running = False
        self.worker_task = None

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler for a specific task type."""
        self.task_handlers[task_type] = handler
        logger.info(f"Registered handler for task type: {task_type}")

    async def submit_task(self, task_type: str, payload: Dict[str, Any],
                         priority: TaskPriority = TaskPriority.NORMAL,
                         queue_name: str = "default") -> str:
        """Submit a new task to the queue."""
        task = QueueTask(
            task_id=str(uuid.uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority
        )

        success = await self.backend.enqueue(task, queue_name)
        if success:
            logger.info(f"Submitted task {task.task_id} of type {task_type}")
            return task.task_id
        else:
            raise Exception(f"Failed to submit task {task.task_id}")

    async def get_task_status(self, task_id: str) -> Optional[QueueTask]:
        """Get the status of a specific task."""
        return await self.backend.get_task(task_id)

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        return await self.backend.update_task_status(task_id, TaskStatus.CANCELLED)

    async def start_worker(self, queue_name: str = "default", max_concurrent: int = 5):
        """Start worker to process tasks from queue."""
        self.running = True
        logger.info(f"Starting worker {self.worker_id} for queue {queue_name}")

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_task_with_semaphore(task: QueueTask):
            async with semaphore:
                await self._process_task(task)

        while self.running:
            try:
                # Get next task
                task = await self.backend.dequeue(queue_name)

                if task:
                    # Process task concurrently
                    asyncio.create_task(process_task_with_semaphore(task))
                else:
                    # No tasks available, wait briefly
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def stop_worker(self):
        """Stop the worker."""
        self.running = False
        logger.info(f"Stopping worker {self.worker_id}")

    async def _process_task(self, task: QueueTask):
        """Process a single task."""
        logger.info(f"Worker {self.worker_id} processing task {task.task_id} of type {task.task_type}")

        try:
            # Check if we have a handler for this task type
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise Exception(f"No handler registered for task type: {task.task_type}")

            # Update task as processing
            task.worker_id = self.worker_id
            await self.backend.update_task_status(task.task_id, TaskStatus.PROCESSING)

            # Execute the task handler
            result = await handler(task)

            # Mark as completed
            await self.backend.update_task_status(task.task_id, TaskStatus.COMPLETED, progress=100.0)

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")

            # Check if task can be retried
            if task.can_retry():
                task.retries += 1
                task.status = TaskStatus.RETRYING
                await self.backend.enqueue(task)  # Re-queue for retry
                await self.backend.update_task_status(
                    task.task_id, TaskStatus.RETRYING,
                    error_message=f"Retry {task.retries}/{task.max_retries}: {str(e)}"
                )
            else:
                await self.backend.update_task_status(
                    task.task_id, TaskStatus.FAILED,
                    error_message=str(e)
                )

    async def get_queue_stats(self, queue_name: str = "default") -> Dict[str, Any]:
        """Get comprehensive queue statistics."""
        stats = await self.backend.get_queue_stats(queue_name)

        return {
            "queue_name": queue_name,
            "worker_id": self.worker_id,
            "task_counts": stats,
            "registered_handlers": list(self.task_handlers.keys()),
            "worker_running": self.running
        }

    async def cleanup_expired_tasks(self, timeout_hours: int = 24) -> int:
        """Clean up expired tasks."""
        return await self.backend.cleanup_expired_tasks(timeout_hours)


# Factory function to create queue manager
def create_queue_manager(backend_type: str = "redis", **kwargs) -> DistributedQueueManager:
    """Create a distributed queue manager with the specified backend."""

    if backend_type.lower() == "redis":
        backend = RedisQueueBackend(kwargs.get("redis_url", "redis://localhost:6379/0"))
    elif backend_type.lower() == "celery":
        backend = CeleryQueueBackend(
            kwargs.get("broker_url", "redis://localhost:6379/1"),
            kwargs.get("result_backend", "redis://localhost:6379/2")
        )
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")

    worker_id = kwargs.get("worker_id")
    return DistributedQueueManager(backend, worker_id)
