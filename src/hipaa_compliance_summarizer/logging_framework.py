"""Structured logging framework with metrics support for HIPAA compliance summarizer.

This module provides a comprehensive logging infrastructure with:
- Structured JSON logging for better log analysis
- Performance metrics collection
- Security event logging
- PHI detection event tracking
- Context-aware logging with metadata
"""

import json
import logging
import os
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, List, Optional, Union


@dataclass
class LoggingConfig:
    """Configuration for structured logging framework."""

    level: str = "INFO"
    format_type: str = "structured"  # "structured" or "simple"
    output_format: str = "json"  # "json" or "text"
    enable_metrics: bool = True
    max_context_size: int = 1000  # Max characters in context fields
    metrics_retention_hours: int = 24
    security_log_level: str = "WARNING"

    @classmethod
    def from_environment(cls) -> 'LoggingConfig':
        """Create configuration from environment variables."""
        return cls(
            level=os.environ.get("LOG_LEVEL", "INFO").upper(),
            format_type=os.environ.get("LOG_FORMAT_TYPE", "structured"),
            output_format=os.environ.get("LOG_OUTPUT_FORMAT", "json"),
            enable_metrics=os.environ.get("ENABLE_METRICS", "true").lower() == "true",
            max_context_size=int(os.environ.get("LOG_MAX_CONTEXT_SIZE", "1000")),
            metrics_retention_hours=int(os.environ.get("METRICS_RETENTION_HOURS", "24")),
            security_log_level=os.environ.get("SECURITY_LOG_LEVEL", "WARNING").upper()
        )


class MetricsCollector:
    """Thread-safe metrics collector for performance monitoring."""

    def __init__(self, retention_hours: int = 24):
        """Initialize metrics collector.
        
        Args:
            retention_hours: How long to retain metrics data
        """
        self._lock = Lock()
        self._counters = defaultdict(list)
        self._timings = defaultdict(list)
        self._gauges = {}
        self._histograms = defaultdict(list)
        self._retention_seconds = retention_hours * 3600

    def increment_counter(self, name: str, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            tags: Optional tags for metric filtering
        """
        with self._lock:
            timestamp = time.time()
            tags = tags or {}

            # Find existing counter with same tags or create new
            for counter in self._counters[name]:
                if counter["tags"] == tags:
                    counter["count"] += 1
                    counter["last_updated"] = timestamp
                    return

            # Create new counter
            self._counters[name].append({
                "count": 1,
                "tags": tags,
                "created": timestamp,
                "last_updated": timestamp
            })

            self._cleanup_old_metrics()

    def record_timing(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a timing metric.
        
        Args:
            name: Timing metric name
            duration_ms: Duration in milliseconds
            tags: Optional tags for metric filtering
        """
        with self._lock:
            self._timings[name].append({
                "duration_ms": duration_ms,
                "timestamp": time.time(),
                "tags": tags or {}
            })

            self._cleanup_old_metrics()

    @contextmanager
    def time_operation(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations.
        
        Args:
            name: Operation name
            tags: Optional tags for metric filtering
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.record_timing(name, duration_ms, tags)

    def set_gauge(self, name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None) -> None:
        """Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Gauge value
            tags: Optional tags for metric filtering
        """
        with self._lock:
            self._gauges[name] = {
                "value": value,
                "timestamp": time.time(),
                "tags": tags or {}
            }

    def record_histogram(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a value in a histogram.
        
        Args:
            name: Histogram name
            value: Value to record
            tags: Optional tags for metric filtering
        """
        with self._lock:
            timestamp = time.time()
            tags = tags or {}

            # Find existing histogram with same tags or create new
            for histogram in self._histograms[name]:
                if histogram["tags"] == tags:
                    histogram["values"].append(value)
                    histogram["count"] += 1
                    histogram["sum"] += value
                    histogram["min"] = min(histogram["min"], value)
                    histogram["max"] = max(histogram["max"], value)
                    histogram["avg"] = histogram["sum"] / histogram["count"]
                    histogram["last_updated"] = timestamp
                    return

            # Create new histogram
            self._histograms[name].append({
                "values": deque([value], maxlen=1000),  # Keep last 1000 values
                "count": 1,
                "sum": value,
                "min": value,
                "max": value,
                "avg": value,
                "tags": tags,
                "created": timestamp,
                "last_updated": timestamp
            })

            self._cleanup_old_metrics()

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot.
        
        Returns:
            Dictionary containing all current metrics
        """
        with self._lock:
            self._cleanup_old_metrics()

            # Convert histograms to serializable format
            serializable_histograms = {}
            for name, histograms in self._histograms.items():
                serializable_histograms[name] = []
                for histogram in histograms:
                    hist_copy = histogram.copy()
                    hist_copy["values"] = list(hist_copy["values"])
                    serializable_histograms[name].append(hist_copy)

            return {
                "counters": dict(self._counters),
                "timings": dict(self._timings),
                "gauges": dict(self._gauges),
                "histograms": serializable_histograms,
                "timestamp": time.time()
            }

    def _cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - self._retention_seconds

        # Clean counters
        for name in list(self._counters.keys()):
            self._counters[name] = [
                counter for counter in self._counters[name]
                if counter["last_updated"] > cutoff_time
            ]
            if not self._counters[name]:
                del self._counters[name]

        # Clean timings
        for name in list(self._timings.keys()):
            self._timings[name] = [
                timing for timing in self._timings[name]
                if timing["timestamp"] > cutoff_time
            ]
            if not self._timings[name]:
                del self._timings[name]

        # Clean histograms
        for name in list(self._histograms.keys()):
            self._histograms[name] = [
                histogram for histogram in self._histograms[name]
                if histogram["last_updated"] > cutoff_time
            ]
            if not self._histograms[name]:
                del self._histograms[name]

        # Clean gauges
        for name in list(self._gauges.keys()):
            if self._gauges[name]["timestamp"] <= cutoff_time:
                del self._gauges[name]


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def __init__(self, config: LoggingConfig):
        """Initialize formatter with configuration.
        
        Args:
            config: Logging configuration
        """
        super().__init__()
        self.config = config

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON or text.
        
        Args:
            record: Log record to format
            
        Returns:
            Formatted log message
        """
        # Extract custom fields from record
        context = getattr(record, 'context', {})
        event_type = getattr(record, 'event_type', None)

        if self.config.output_format == "json":
            return self._format_json(record, context, event_type)
        else:
            return self._format_text(record, context, event_type)

    def _format_json(self, record: logging.LogRecord, context: Dict[str, Any], event_type: Optional[str]) -> str:
        """Format as JSON."""
        # Extract special fields from context for top-level placement
        context_copy = context.copy()
        security_event = context_copy.pop("security_event", None)
        severity = context_copy.pop("severity", None)

        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger_name": record.name,
            "message": record.getMessage(),
            "context": self._sanitize_context(context_copy, preserve_types=True)
        }

        if event_type:
            log_data["event_type"] = event_type

        # Add security-specific fields at top level
        if security_event:
            log_data["security_event"] = security_event
        if severity:
            log_data["severity"] = severity

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[1].__class__.__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }

        # Add any additional custom fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName", "process",
                          "getMessage", "exc_info", "exc_text", "stack_info", "context", "event_type"]:
                log_data[key] = value

        return json.dumps(log_data, default=str)

    def _format_text(self, record: logging.LogRecord, context: Dict[str, Any], event_type: Optional[str]) -> str:
        """Format as human-readable text."""
        parts = [
            f"{datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')}",
            f"[{record.levelname}]",
            f"{record.name}:",
            record.getMessage()
        ]

        if event_type:
            parts.append(f"({event_type})")

        if context:
            context_str = " ".join(f"{k}={v}" for k, v in self._sanitize_context(context, preserve_types=False).items())
            parts.append(f"- {context_str}")

        if record.exc_info:
            parts.append(f"Exception: {record.exc_info[1]}")

        return " ".join(parts)

    def _sanitize_context(self, context: Dict[str, Any], preserve_types: bool = True) -> Dict[str, Any]:
        """Sanitize context data for logging.
        
        Args:
            context: Context dictionary
            preserve_types: Whether to preserve original data types (for JSON) or convert to strings (for text)
            
        Returns:
            Sanitized context dictionary
        """
        sanitized = {}

        for key, value in context.items():
            # Mask potential PHI or sensitive data (but not standalone "key" which is too generic)
            sensitive_patterns = ["password", "token", "secret", "ssn", "phone", "api_key", "auth_key", "private_key"]
            if any(sensitive in key.lower() for sensitive in sensitive_patterns):
                sanitized[key] = "[MASKED]"
            elif preserve_types:
                # For JSON output, check if string representation would be too long but preserve types
                str_value = str(value)
                if len(str_value) > self.config.max_context_size:
                    sanitized[key] = str_value[:self.config.max_context_size] + "..."
                else:
                    sanitized[key] = value  # Preserve original type
            else:
                # For text output, convert to string and truncate if needed
                str_value = str(value)
                if len(str_value) > self.config.max_context_size:
                    str_value = str_value[:self.config.max_context_size] + "..."
                sanitized[key] = str_value

        return sanitized


class StructuredLogger:
    """Enhanced logger with structured logging and metrics support."""

    def __init__(self, name: str, config: LoggingConfig, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize structured logger.
        
        Args:
            name: Logger name
            config: Logging configuration
            metrics_collector: Optional metrics collector
        """
        self.name = name
        self.config = config
        self._metrics_collector = metrics_collector

        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, config.level, logging.INFO))

        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Add structured formatter
        handler = logging.StreamHandler()
        formatter = StructuredFormatter(config)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def _log_with_context(self, level: int, message: str, context: Optional[Dict[str, Any]] = None,
                         event_type: Optional[str] = None, **kwargs) -> None:
        """Internal method to log with context.
        
        Args:
            level: Log level
            message: Log message
            context: Context dictionary
            event_type: Event type classification
            **kwargs: Additional logging arguments
        """
        context = context or {}

        # Separate exc_info from other kwargs since it's a special logging parameter
        exc_info = kwargs.pop('exc_info', None)

        # Create log record with custom attributes
        extra = {
            "context": context,
            "event_type": event_type
        }
        extra.update(kwargs)

        self.logger.log(level, message, extra=extra, exc_info=exc_info)

        # Update metrics if enabled
        if self._metrics_collector:
            tags = {
                "logger": self.name,
                "level": logging.getLevelName(level),
                "event_type": event_type or "general"
            }
            self._metrics_collector.increment_counter("log_messages", tags)

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log debug message."""
        self._log_with_context(logging.DEBUG, message, context, **kwargs)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log info message."""
        self._log_with_context(logging.INFO, message, context, **kwargs)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log warning message."""
        self._log_with_context(logging.WARNING, message, context, **kwargs)

    def error(self, message: str, context: Optional[Dict[str, Any]] = None,
              exc_info: Optional[Exception] = None, **kwargs) -> None:
        """Log error message with optional exception."""
        self._log_with_context(logging.ERROR, message, context, exc_info=exc_info, **kwargs)

    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Log critical message."""
        self._log_with_context(logging.CRITICAL, message, context, **kwargs)

    @contextmanager
    def log_performance(self, operation: str, context: Optional[Dict[str, Any]] = None):
        """Context manager for performance logging.
        
        Args:
            operation: Operation name
            context: Additional context
        """
        start_time = time.perf_counter()
        context = context or {}

        try:
            yield
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_context = context.copy()
            error_context.update({
                "operation": operation,
                "duration_ms": duration_ms,
                "status": "failed"
            })
            self.error(f"{operation} failed", error_context, exc_info=e)

            if self._metrics_collector:
                self._metrics_collector.record_timing(operation, duration_ms, {"status": "failed"})
                self._metrics_collector.increment_counter("operation_failures", {"operation": operation})

            raise
        else:
            duration_ms = (time.perf_counter() - start_time) * 1000
            success_context = context.copy()
            success_context.update({
                "operation": operation,
                "duration_ms": duration_ms,
                "status": "success"
            })
            self.info(f"{operation} completed", success_context, event_type="PERFORMANCE")

            if self._metrics_collector:
                self._metrics_collector.record_timing(operation, duration_ms, {"status": "success"})
                self._metrics_collector.increment_counter("operation_successes", {"operation": operation})

    def log_security_event(self, event_name: str, severity: str = "MEDIUM",
                          context: Optional[Dict[str, Any]] = None) -> None:
        """Log security-related events.
        
        Args:
            event_name: Security event name
            severity: Event severity (LOW, MEDIUM, HIGH, CRITICAL)
            context: Additional context
        """
        context = context or {}
        context.update({
            "security_event": event_name,
            "severity": severity
        })

        # Use configured security log level
        level = getattr(logging, self.config.security_log_level, logging.WARNING)
        self._log_with_context(level, f"Security event: {event_name}", context, "SECURITY")

        if self._metrics_collector:
            tags = {"event_name": event_name, "severity": severity}
            self._metrics_collector.increment_counter("security_events", tags)

    def log_phi_detection(self, detected_types: List[str], document_id: str,
                         confidence_scores: Optional[Dict[str, float]] = None,
                         redacted_count: int = 0, context: Optional[Dict[str, Any]] = None) -> None:
        """Log PHI detection events.
        
        Args:
            detected_types: List of detected PHI types
            document_id: Document identifier
            confidence_scores: Confidence scores for each type
            redacted_count: Number of items redacted
            context: Additional context
        """
        context = context or {}
        context.update({
            "detected_types": detected_types,
            "document_id": document_id,
            "redacted_count": redacted_count
        })

        if confidence_scores:
            context["confidence_scores"] = confidence_scores

        self._log_with_context(logging.INFO, f"PHI detected in document {document_id}", context, "PHI_DETECTION")

        if self._metrics_collector:
            # Record metrics for each detected type
            for phi_type in detected_types:
                tags = {"phi_type": phi_type, "document_id": document_id}
                self._metrics_collector.increment_counter("phi_detected", tags)

            # Record redaction count
            self._metrics_collector.record_histogram("redacted_items_per_document", redacted_count)


# Global metrics collector instance
_global_metrics_collector = None
_global_config = None


def setup_structured_logging(config: Optional[LoggingConfig] = None) -> MetricsCollector:
    """Set up the structured logging framework globally.
    
    Args:
        config: Optional logging configuration
        
    Returns:
        Global metrics collector instance
    """
    global _global_metrics_collector, _global_config

    if config is None:
        config = LoggingConfig.from_environment()

    _global_config = config

    if config.enable_metrics:
        _global_metrics_collector = MetricsCollector(config.metrics_retention_hours)
    else:
        _global_metrics_collector = None

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = StructuredFormatter(config)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, config.level, logging.INFO))

    return _global_metrics_collector


def get_logger_with_metrics(name: str) -> StructuredLogger:
    """Get a structured logger instance with metrics support.
    
    Args:
        name: Logger name
        
    Returns:
        Structured logger instance
    """
    global _global_metrics_collector, _global_config

    if _global_config is None:
        setup_structured_logging()

    return StructuredLogger(name, _global_config, _global_metrics_collector)


def get_global_metrics() -> Optional[Dict[str, Any]]:
    """Get global metrics snapshot.
    
    Returns:
        Dictionary containing all global metrics or None if metrics disabled
    """
    global _global_metrics_collector

    if _global_metrics_collector:
        return _global_metrics_collector.get_metrics()

    return None
