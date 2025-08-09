"""Queue management for background job processing."""

import asyncio
import logging
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 100
    NORMAL = 50
    HIGH = 10
    CRITICAL = 1


@dataclass
class Job:
    """Represents a background job."""

    job_id: str
    job_type: str
    data: Dict[str, Any]
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary."""
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "data": self.data,
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "error_message": self.error_message,
            "result": self.result,
            "metadata": self.metadata
        }


@dataclass
class JobResult:
    """Result of job execution."""

    job_id: str
    success: bool
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "success": self.success,
            "result_data": self.result_data,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }


class JobQueue:
    """Priority queue for jobs."""

    def __init__(self, name: str):
        """Initialize job queue.
        
        Args:
            name: Queue name
        """
        self.name = name
        self.queues = {
            JobPriority.CRITICAL: deque(),
            JobPriority.HIGH: deque(),
            JobPriority.NORMAL: deque(),
            JobPriority.LOW: deque()
        }
        self._lock = threading.Lock()

    def put(self, job: Job):
        """Add job to queue."""
        with self._lock:
            self.queues[job.priority].append(job)

    def get(self) -> Optional[Job]:
        """Get next job from queue (highest priority first)."""
        with self._lock:
            for priority in [JobPriority.CRITICAL, JobPriority.HIGH, JobPriority.NORMAL, JobPriority.LOW]:
                if self.queues[priority]:
                    return self.queues[priority].popleft()
        return None

    def size(self) -> int:
        """Get total queue size."""
        with self._lock:
            return sum(len(q) for q in self.queues.values())

    def size_by_priority(self) -> Dict[str, int]:
        """Get queue size by priority."""
        with self._lock:
            return {priority.name: len(queue) for priority, queue in self.queues.items()}

    def clear(self):
        """Clear all jobs from queue."""
        with self._lock:
            for queue in self.queues.values():
                queue.clear()


class QueueManager:
    """Manages multiple job queues and workers."""

    def __init__(self, max_workers: int = 5):
        """Initialize queue manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers
        self.queues: Dict[str, JobQueue] = {}
        self.jobs: Dict[str, Job] = {}
        self.job_handlers: Dict[str, Callable] = {}
        self.workers: List[asyncio.Task] = []
        self.running = False
        self._lock = threading.Lock()

        # Job statistics
        self.job_stats = {
            "total_jobs": 0,
            "completed_jobs": 0,
            "failed_jobs": 0,
            "retried_jobs": 0
        }

        # Create default queue
        self.create_queue("default")

    def create_queue(self, queue_name: str):
        """Create a new job queue."""
        with self._lock:
            if queue_name not in self.queues:
                self.queues[queue_name] = JobQueue(queue_name)
                logger.info(f"Created queue: {queue_name}")

    def register_handler(self, job_type: str, handler: Callable):
        """Register a job handler.
        
        Args:
            job_type: Type of job this handler processes
            handler: Async function to handle the job
        """
        self.job_handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    def submit_job(self, job_type: str, data: Dict[str, Any],
                   priority: JobPriority = JobPriority.NORMAL,
                   queue_name: str = "default",
                   max_retries: int = 3,
                   metadata: Dict[str, Any] = None) -> str:
        """Submit a job to a queue.
        
        Args:
            job_type: Type of job
            data: Job data
            priority: Job priority
            queue_name: Target queue name
            max_retries: Maximum retry attempts
            metadata: Additional metadata
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())

        job = Job(
            job_id=job_id,
            job_type=job_type,
            data=data,
            priority=priority,
            max_retries=max_retries,
            metadata=metadata or {}
        )

        # Ensure queue exists
        if queue_name not in self.queues:
            self.create_queue(queue_name)

        # Add to queue and job registry
        with self._lock:
            self.queues[queue_name].put(job)
            self.jobs[job_id] = job
            self.job_stats["total_jobs"] += 1

        logger.info(f"Submitted job {job_id} (type: {job_type}) to queue: {queue_name}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        with self._lock:
            return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job."""
        with self._lock:
            job = self.jobs.get(job_id)
            if job and job.status == JobStatus.PENDING:
                job.status = JobStatus.CANCELLED
                logger.info(f"Cancelled job: {job_id}")
                return True
        return False

    async def start_workers(self):
        """Start background workers."""
        if self.running:
            return

        self.running = True
        logger.info(f"Starting {self.max_workers} workers")

        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i+1}"))
            self.workers.append(worker)

    async def stop_workers(self):
        """Stop all background workers."""
        if not self.running:
            return

        self.running = False
        logger.info("Stopping workers")

        # Cancel all worker tasks
        for worker in self.workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("All workers stopped")

    async def _worker_loop(self, worker_name: str):
        """Main worker loop."""
        logger.info(f"Worker {worker_name} started")

        while self.running:
            try:
                # Get next job from any queue
                job = self._get_next_job()

                if job:
                    await self._process_job(job, worker_name)
                else:
                    # No jobs available, wait briefly
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(5)  # Wait before retrying

        logger.info(f"Worker {worker_name} stopped")

    def _get_next_job(self) -> Optional[Job]:
        """Get next job from any queue."""
        with self._lock:
            # Check all queues for pending jobs
            for queue in self.queues.values():
                job = queue.get()
                if job and job.status == JobStatus.PENDING:
                    return job
        return None

    async def _process_job(self, job: Job, worker_name: str):
        """Process a single job."""
        start_time = datetime.utcnow()

        try:
            # Update job status
            with self._lock:
                job.status = JobStatus.RUNNING
                job.started_at = start_time

            logger.info(f"Worker {worker_name} processing job {job.job_id} (type: {job.job_type})")

            # Get handler for job type
            handler = self.job_handlers.get(job.job_type)
            if not handler:
                raise ValueError(f"No handler registered for job type: {job.job_type}")

            # Execute job
            result = await handler(job)

            # Update job with success
            end_time = datetime.utcnow()
            execution_time = (end_time - start_time).total_seconds() * 1000

            with self._lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = end_time
                job.result = result.result_data if isinstance(result, JobResult) else result
                self.job_stats["completed_jobs"] += 1

            logger.info(f"Job {job.job_id} completed in {execution_time:.2f}ms")

        except Exception as e:
            # Handle job failure
            end_time = datetime.utcnow()
            error_message = str(e)

            with self._lock:
                job.error_message = error_message
                job.completed_at = end_time

                # Check if job should be retried
                if job.retry_count < job.max_retries:
                    job.retry_count += 1
                    job.status = JobStatus.RETRYING
                    job.started_at = None
                    job.completed_at = None

                    # Re-queue job for retry
                    queue_name = job.metadata.get("queue_name", "default")
                    if queue_name in self.queues:
                        self.queues[queue_name].put(job)

                    self.job_stats["retried_jobs"] += 1
                    logger.warning(f"Job {job.job_id} failed, retry {job.retry_count}/{job.max_retries}: {error_message}")
                else:
                    job.status = JobStatus.FAILED
                    self.job_stats["failed_jobs"] += 1
                    logger.error(f"Job {job.job_id} failed permanently: {error_message}")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all queues."""
        with self._lock:
            queue_status = {}
            for name, queue in self.queues.items():
                queue_status[name] = {
                    "size": queue.size(),
                    "by_priority": queue.size_by_priority()
                }

            running_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.RUNNING)
            pending_jobs = sum(1 for job in self.jobs.values() if job.status == JobStatus.PENDING)

            return {
                "queues": queue_status,
                "workers": {
                    "total": len(self.workers),
                    "running": self.running
                },
                "jobs": {
                    "running": running_jobs,
                    "pending": pending_jobs,
                    **self.job_stats
                }
            }

    def get_job_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get job history."""
        with self._lock:
            # Get recent jobs, sorted by creation time
            recent_jobs = sorted(
                self.jobs.values(),
                key=lambda j: j.created_at,
                reverse=True
            )[:limit]

            return [job.to_dict() for job in recent_jobs]

    def cleanup_completed_jobs(self, older_than_hours: int = 24):
        """Clean up old completed jobs."""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)

        with self._lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if (job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED] and
                    job.completed_at and job.completed_at < cutoff_time):
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]

            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

    async def wait_for_job(self, job_id: str, timeout_seconds: int = 300) -> Optional[Job]:
        """Wait for a job to complete.
        
        Args:
            job_id: Job ID to wait for
            timeout_seconds: Maximum time to wait
            
        Returns:
            Completed job or None if timeout
        """
        start_time = datetime.utcnow()

        while (datetime.utcnow() - start_time).total_seconds() < timeout_seconds:
            job = self.get_job(job_id)
            if job and job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                return job

            await asyncio.sleep(1)

        return None  # Timeout

    async def process_job_sync(self, job_type: str, data: Dict[str, Any],
                              timeout_seconds: int = 300) -> Optional[JobResult]:
        """Submit job and wait for completion.
        
        Args:
            job_type: Type of job
            data: Job data
            timeout_seconds: Maximum time to wait
            
        Returns:
            Job result or None if timeout/failure
        """
        job_id = self.submit_job(job_type, data, priority=JobPriority.HIGH)
        completed_job = await self.wait_for_job(job_id, timeout_seconds)

        if completed_job:
            return JobResult(
                job_id=job_id,
                success=completed_job.status == JobStatus.COMPLETED,
                result_data=completed_job.result,
                error_message=completed_job.error_message,
                execution_time_ms=(completed_job.completed_at - completed_job.started_at).total_seconds() * 1000
                if completed_job.started_at and completed_job.completed_at else 0
            )

        return None
