"""Distributed tracing for HIPAA compliance system."""

import os
import time
import uuid
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
import json


logger = logging.getLogger(__name__)


@dataclass
class Span:
    """Represents a single span in a distributed trace."""
    
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"  # ok, error, timeout
    
    def finish(self):
        """Finish the span and calculate duration."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
            self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def add_tag(self, key: str, value: str):
        """Add a tag to the span."""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **fields):
        """Add a log entry to the span."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **fields
        }
        self.logs.append(log_entry)
    
    def set_error(self, error: Exception):
        """Mark span as error and add error details."""
        self.status = "error"
        self.add_tag("error", "true")
        self.add_tag("error.type", type(error).__name__)
        self.add_log(
            message=f"Error: {str(error)}",
            level="error",
            error_type=type(error).__name__,
            error_message=str(error)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "tags": self.tags,
            "logs": self.logs,
            "status": self.status
        }


@dataclass
class Trace:
    """Represents a complete distributed trace."""
    
    trace_id: str
    root_span: Span
    spans: List[Span] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans.append(span)
    
    def get_span_by_id(self, span_id: str) -> Optional[Span]:
        """Get a span by its ID."""
        for span in self.spans:
            if span.span_id == span_id:
                return span
        return None
    
    def get_total_duration_ms(self) -> float:
        """Get total trace duration."""
        if self.root_span.duration_ms is not None:
            return self.root_span.duration_ms
        return 0.0
    
    def get_critical_path(self) -> List[Span]:
        """Get the critical path (longest duration path) through the trace."""
        # Simple implementation - return spans sorted by start time
        return sorted(self.spans, key=lambda s: s.start_time)
    
    def has_errors(self) -> bool:
        """Check if any span in the trace has errors."""
        return any(span.status == "error" for span in self.spans)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "created_at": self.created_at.isoformat(),
            "total_duration_ms": self.get_total_duration_ms(),
            "span_count": len(self.spans),
            "has_errors": self.has_errors(),
            "root_span": self.root_span.to_dict(),
            "spans": [span.to_dict() for span in self.spans]
        }


class TracingContext:
    """Thread-local context for managing current trace and span."""
    
    _local = threading.local()
    
    @classmethod
    def get_current_trace(cls) -> Optional[Trace]:
        """Get the current trace from thread-local storage."""
        return getattr(cls._local, 'current_trace', None)
    
    @classmethod
    def set_current_trace(cls, trace: Trace):
        """Set the current trace in thread-local storage."""
        cls._local.current_trace = trace
    
    @classmethod
    def get_current_span(cls) -> Optional[Span]:
        """Get the current span from thread-local storage."""
        return getattr(cls._local, 'current_span', None)
    
    @classmethod
    def set_current_span(cls, span: Span):
        """Set the current span in thread-local storage."""
        cls._local.current_span = span
    
    @classmethod
    def clear_context(cls):
        """Clear the tracing context."""
        cls._local.current_trace = None
        cls._local.current_span = None


class Tracer:
    """Main tracer class for creating and managing traces."""
    
    def __init__(self, service_name: str = "hipaa-compliance-summarizer"):
        """Initialize tracer.
        
        Args:
            service_name: Name of the service being traced
        """
        self.service_name = service_name
        self.active_traces: Dict[str, Trace] = {}
        self.completed_traces: List[Trace] = []
        self.max_completed_traces = int(os.getenv("TRACING_MAX_COMPLETED", "1000"))
        self._lock = threading.Lock()
        
        # Sampling configuration
        self.sampling_rate = float(os.getenv("TRACING_SAMPLING_RATE", "1.0"))
        self.enable_tracing = os.getenv("TRACING_ENABLED", "true").lower() == "true"
    
    def should_sample(self) -> bool:
        """Determine if a trace should be sampled."""
        if not self.enable_tracing:
            return False
        
        import random
        return random.random() < self.sampling_rate
    
    def start_trace(self, operation_name: str, trace_id: str = None) -> Trace:
        """Start a new trace.
        
        Args:
            operation_name: Name of the root operation
            trace_id: Optional existing trace ID
            
        Returns:
            New trace object
        """
        if not self.should_sample():
            return None
        
        if trace_id is None:
            trace_id = str(uuid.uuid4())
        
        # Create root span
        root_span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=None,
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )
        
        # Add service tags
        root_span.add_tag("service.name", self.service_name)
        root_span.add_tag("span.kind", "server")
        
        # Create trace
        trace = Trace(trace_id=trace_id, root_span=root_span)
        trace.add_span(root_span)
        
        # Store in active traces
        with self._lock:
            self.active_traces[trace_id] = trace
        
        # Set as current context
        TracingContext.set_current_trace(trace)
        TracingContext.set_current_span(root_span)
        
        logger.debug(f"Started trace: {trace_id} for operation: {operation_name}")
        return trace
    
    def start_span(self, operation_name: str, parent_span: Span = None,
                   trace: Trace = None) -> Span:
        """Start a new span.
        
        Args:
            operation_name: Name of the operation
            parent_span: Parent span (uses current span if not provided)
            trace: Trace to add span to (uses current trace if not provided)
            
        Returns:
            New span object
        """
        # Get current context if not provided
        if trace is None:
            trace = TracingContext.get_current_trace()
        
        if trace is None:
            # No active trace, create a new one
            trace = self.start_trace(operation_name)
            return trace.root_span if trace else None
        
        if parent_span is None:
            parent_span = TracingContext.get_current_span()
        
        # Create new span
        span = Span(
            trace_id=trace.trace_id,
            span_id=str(uuid.uuid4()),
            parent_span_id=parent_span.span_id if parent_span else None,
            operation_name=operation_name,
            start_time=datetime.utcnow()
        )
        
        # Add to trace
        trace.add_span(span)
        
        # Set as current span
        TracingContext.set_current_span(span)
        
        logger.debug(f"Started span: {span.span_id} for operation: {operation_name}")
        return span
    
    def finish_span(self, span: Span):
        """Finish a span.
        
        Args:
            span: Span to finish
        """
        if span is None:
            return
        
        span.finish()
        
        # If this was the current span, reset to parent
        current_span = TracingContext.get_current_span()
        if current_span and current_span.span_id == span.span_id:
            # Find parent span
            trace = TracingContext.get_current_trace()
            if trace:
                parent_span = trace.get_span_by_id(span.parent_span_id) if span.parent_span_id else None
                TracingContext.set_current_span(parent_span)
        
        logger.debug(f"Finished span: {span.span_id} (duration: {span.duration_ms:.2f}ms)")
    
    def finish_trace(self, trace: Trace):
        """Finish a trace.
        
        Args:
            trace: Trace to finish
        """
        if trace is None:
            return
        
        # Finish root span if not already finished
        if trace.root_span.end_time is None:
            trace.root_span.finish()
        
        # Move to completed traces
        with self._lock:
            if trace.trace_id in self.active_traces:
                del self.active_traces[trace.trace_id]
            
            self.completed_traces.append(trace)
            
            # Trim completed traces if needed
            if len(self.completed_traces) > self.max_completed_traces:
                self.completed_traces = self.completed_traces[-self.max_completed_traces:]
        
        # Clear context if this was the current trace
        current_trace = TracingContext.get_current_trace()
        if current_trace and current_trace.trace_id == trace.trace_id:
            TracingContext.clear_context()
        
        logger.debug(f"Finished trace: {trace.trace_id} (duration: {trace.get_total_duration_ms():.2f}ms)")
    
    @contextmanager
    def trace(self, operation_name: str, **tags):
        """Context manager for tracing an operation.
        
        Args:
            operation_name: Name of the operation
            **tags: Additional tags to add to the span
        """
        span = self.start_span(operation_name)
        
        if span is None:
            yield None
            return
        
        try:
            # Add tags
            for key, value in tags.items():
                span.add_tag(key, str(value))
            
            yield span
            
        except Exception as e:
            if span:
                span.set_error(e)
            raise
        finally:
            self.finish_span(span)
    
    def get_trace_by_id(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID.
        
        Args:
            trace_id: Trace ID to search for
            
        Returns:
            Trace object if found
        """
        # Check active traces first
        with self._lock:
            if trace_id in self.active_traces:
                return self.active_traces[trace_id]
        
        # Check completed traces
        for trace in self.completed_traces:
            if trace.trace_id == trace_id:
                return trace
        
        return None
    
    def get_traces_by_operation(self, operation_name: str, 
                               limit: int = 100) -> List[Trace]:
        """Get traces by root operation name.
        
        Args:
            operation_name: Operation name to search for
            limit: Maximum number of traces to return
            
        Returns:
            List of matching traces
        """
        matching_traces = []
        
        # Search completed traces (most recent first)
        for trace in reversed(self.completed_traces):
            if trace.root_span.operation_name == operation_name:
                matching_traces.append(trace)
                if len(matching_traces) >= limit:
                    break
        
        return matching_traces
    
    def get_trace_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trace statistics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Statistics dictionary
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        # Filter traces by time
        recent_traces = [
            trace for trace in self.completed_traces
            if trace.created_at >= cutoff
        ]
        
        if not recent_traces:
            return {
                "period_hours": hours,
                "total_traces": 0,
                "avg_duration_ms": 0,
                "error_rate": 0,
                "operations": {}
            }
        
        # Calculate statistics
        total_traces = len(recent_traces)
        durations = [trace.get_total_duration_ms() for trace in recent_traces]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        error_traces = [trace for trace in recent_traces if trace.has_errors()]
        error_rate = len(error_traces) / total_traces if total_traces > 0 else 0
        
        # Operation breakdown
        operations = defaultdict(lambda: {"count": 0, "avg_duration": 0, "error_count": 0})
        
        for trace in recent_traces:
            op_name = trace.root_span.operation_name
            operations[op_name]["count"] += 1
            operations[op_name]["avg_duration"] += trace.get_total_duration_ms()
            if trace.has_errors():
                operations[op_name]["error_count"] += 1
        
        # Calculate averages
        for op_stats in operations.values():
            if op_stats["count"] > 0:
                op_stats["avg_duration"] /= op_stats["count"]
                op_stats["error_rate"] = op_stats["error_count"] / op_stats["count"]
        
        return {
            "period_hours": hours,
            "total_traces": total_traces,
            "avg_duration_ms": avg_duration,
            "error_rate": error_rate,
            "operations": dict(operations)
        }


# Global tracer instance
_global_tracer = None


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = Tracer()
    return _global_tracer


def trace_operation(operation_name: str, **tags):
    """Decorator for tracing functions.
    
    Args:
        operation_name: Name of the operation
        **tags: Additional tags to add to the span
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            with tracer.trace(operation_name, **tags) as span:
                if span:
                    # Add function metadata
                    span.add_tag("function.name", func.__name__)
                    span.add_tag("function.module", func.__module__)
                
                return func(*args, **kwargs)
        return wrapper
    return decorator


def trace_phi_detection(func):
    """Specialized decorator for PHI detection operations."""
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        with tracer.trace("phi_detection", component="phi_service") as span:
            if span:
                span.add_tag("function.name", func.__name__)
                
                # Extract method and confidence from kwargs if available
                if "detection_method" in kwargs:
                    span.add_tag("detection.method", kwargs["detection_method"])
                if "confidence_threshold" in kwargs:
                    span.add_tag("detection.confidence_threshold", str(kwargs["confidence_threshold"]))
            
            result = func(*args, **kwargs)
            
            if span and hasattr(result, 'entities'):
                span.add_tag("phi.entities_found", str(len(result.entities)))
                span.add_tag("phi.confidence_avg", str(result.average_confidence))
            
            return result
    return wrapper


def trace_document_processing(func):
    """Specialized decorator for document processing operations."""
    def wrapper(*args, **kwargs):
        tracer = get_tracer()
        with tracer.trace("document_processing", component="processor") as span:
            if span:
                span.add_tag("function.name", func.__name__)
                
                # Try to extract document ID from args
                if args and hasattr(args[0], 'id'):
                    span.add_tag("document.id", args[0].id)
            
            result = func(*args, **kwargs)
            
            if span and hasattr(result, 'compliance_score'):
                span.add_tag("compliance.score", str(result.compliance_score))
                span.add_tag("compliance.risk_level", result.risk_level)
            
            return result
    return wrapper