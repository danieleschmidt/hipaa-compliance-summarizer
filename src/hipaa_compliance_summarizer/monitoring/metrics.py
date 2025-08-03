"""Metrics collection and monitoring for HIPAA compliance system."""

import time
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric data point."""
    
    name: str
    value: float
    timestamp: datetime
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_prometheus_format(self) -> str:
        """Convert to Prometheus format."""
        labels_str = ""
        if self.labels:
            label_pairs = [f'{k}="{v}"' for k, v in self.labels.items()]
            labels_str = "{" + ",".join(label_pairs) + "}"
        
        timestamp_ms = int(self.timestamp.timestamp() * 1000)
        return f"{self.name}{labels_str} {self.value} {timestamp_ms}"


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    
    response_time_ms: float = 0.0
    throughput_rps: float = 0.0
    error_rate: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    active_requests: int = 0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "response_time_ms": self.response_time_ms,
            "throughput_rps": self.throughput_rps,
            "error_rate": self.error_rate,
            "cpu_usage": self.cpu_usage,
            "memory_usage_mb": self.memory_usage_mb,
            "active_requests": self.active_requests
        }


@dataclass 
class ComplianceMetrics:
    """HIPAA compliance specific metrics."""
    
    documents_processed: int = 0
    phi_entities_detected: int = 0
    compliance_violations: int = 0
    avg_compliance_score: float = 0.0
    high_risk_documents: int = 0
    audit_events: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "documents_processed": self.documents_processed,
            "phi_entities_detected": self.phi_entities_detected,
            "compliance_violations": self.compliance_violations,
            "avg_compliance_score": self.avg_compliance_score,
            "high_risk_documents": self.high_risk_documents,
            "audit_events": self.audit_events
        }


class MetricsCollector:
    """Centralized metrics collection and storage."""
    
    def __init__(self, max_points: int = 10000):
        """Initialize metrics collector.
        
        Args:
            max_points: Maximum number of metric points to store in memory
        """
        self.max_points = max_points
        self.metrics = defaultdict(lambda: deque(maxlen=max_points))
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self._lock = threading.Lock()
        
        # Start background collection thread
        self._collection_thread = threading.Thread(target=self._collect_system_metrics, daemon=True)
        self._collection_thread.start()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """Record a counter metric (monotonically increasing)."""
        with self._lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            metric_point = MetricPoint(
                name=name,
                value=self.counters[key],
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self.metrics[name].append(metric_point)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a gauge metric (can go up or down)."""
        with self._lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self.metrics[name].append(metric_point)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a histogram metric (for distribution analysis)."""
        with self._lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only recent values (last 1000 for memory efficiency)
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=datetime.utcnow(),
                labels=labels or {}
            )
            self.metrics[name].append(metric_point)
    
    def get_metric_values(self, name: str, since: datetime = None) -> List[MetricPoint]:
        """Get metric values, optionally filtered by time."""
        with self._lock:
            points = list(self.metrics[name])
            
            if since:
                points = [p for p in points if p.timestamp >= since]
            
            return points
    
    def get_latest_value(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get the latest value for a metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            if key in self.counters:
                return self.counters[key]
            elif key in self.gauges:
                return self.gauges[key]
            elif name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1].value
        
        return None
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """Get statistical summary of histogram metric."""
        key = self._make_key(name, labels)
        
        with self._lock:
            values = self.histograms.get(key, [])
            
            if not values:
                return {}
            
            values_sorted = sorted(values)
            count = len(values)
            
            return {
                "count": count,
                "sum": sum(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / count,
                "p50": values_sorted[int(count * 0.5)],
                "p95": values_sorted[int(count * 0.95)],
                "p99": values_sorted[int(count * 0.99)] if count >= 100 else values_sorted[-1]
            }
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """Create a unique key for metric storage."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def _collect_system_metrics(self):
        """Background thread to collect system metrics."""
        import psutil
        
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.record_gauge("system_cpu_usage_percent", cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_gauge("system_memory_usage_percent", memory.percent)
                self.record_gauge("system_memory_usage_bytes", memory.used)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.record_gauge("system_disk_usage_percent", disk.percent)
                self.record_gauge("system_disk_usage_bytes", disk.used)
                
                # Process-specific metrics
                process = psutil.Process()
                self.record_gauge("process_memory_bytes", process.memory_info().rss)
                self.record_gauge("process_cpu_percent", process.cpu_percent())
                
            except Exception as e:
                logger.error(f"Failed to collect system metrics: {e}")
            
            time.sleep(30)  # Collect every 30 seconds
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Export counters
            for key, value in self.counters.items():
                name, labels = self._parse_key(key)
                lines.append(f"# TYPE {name} counter")
                if labels:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
                    lines.append(f"{name}{label_str} {value}")
                else:
                    lines.append(f"{name} {value}")
            
            # Export gauges
            for key, value in self.gauges.items():
                name, labels = self._parse_key(key)
                lines.append(f"# TYPE {name} gauge")
                if labels:
                    label_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"
                    lines.append(f"{name}{label_str} {value}")
                else:
                    lines.append(f"{name} {value}")
        
        return "\n".join(lines)
    
    def _parse_key(self, key: str) -> tuple:
        """Parse metric key back to name and labels."""
        if "[" not in key:
            return key, {}
        
        name, label_part = key.split("[", 1)
        label_part = label_part.rstrip("]")
        
        labels = {}
        if label_part:
            for pair in label_part.split(","):
                k, v = pair.split("=", 1)
                labels[k] = v
        
        return name, labels


class PerformanceMonitor:
    """Monitor application performance metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize performance monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
        self.request_times = deque(maxlen=1000)
        self.error_count = 0
        self.request_count = 0
        self._lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            "response_time_ms": float(os.getenv("PERF_THRESHOLD_RESPONSE_MS", "5000")),
            "error_rate": float(os.getenv("PERF_THRESHOLD_ERROR_RATE", "0.05")),
            "cpu_usage": float(os.getenv("PERF_THRESHOLD_CPU", "80.0")),
            "memory_usage": float(os.getenv("PERF_THRESHOLD_MEMORY", "80.0"))
        }
    
    def record_request(self, duration_ms: float, success: bool = True, 
                      endpoint: str = None, method: str = None):
        """Record a request performance metric."""
        with self._lock:
            self.request_times.append(duration_ms)
            self.request_count += 1
            
            if not success:
                self.error_count += 1
        
        # Record metrics
        labels = {}
        if endpoint:
            labels["endpoint"] = endpoint
        if method:
            labels["method"] = method
        
        self.metrics.record_histogram("http_request_duration_ms", duration_ms, labels)
        self.metrics.record_counter("http_requests_total", 1.0, labels)
        
        if not success:
            self.metrics.record_counter("http_requests_failed_total", 1.0, labels)
    
    def record_document_processing(self, duration_ms: float, document_type: str,
                                  phi_count: int, compliance_score: float):
        """Record document processing performance."""
        labels = {"document_type": document_type}
        
        self.metrics.record_histogram("document_processing_duration_ms", duration_ms, labels)
        self.metrics.record_counter("documents_processed_total", 1.0, labels)
        self.metrics.record_histogram("phi_entities_detected", phi_count, labels)
        self.metrics.record_histogram("compliance_score", compliance_score, labels)
    
    def record_phi_detection_performance(self, pattern_name: str, duration_ms: float,
                                       cache_hit: bool, confidence: float):
        """Record PHI detection performance metrics."""
        labels = {
            "pattern": pattern_name,
            "cache_hit": str(cache_hit).lower()
        }
        
        self.metrics.record_histogram("phi_detection_duration_ms", duration_ms, labels)
        self.metrics.record_histogram("phi_detection_confidence", confidence, labels)
        self.metrics.record_counter("phi_detections_total", 1.0, labels)
    
    def get_current_performance(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        with self._lock:
            # Calculate response time
            avg_response_time = (
                sum(self.request_times) / len(self.request_times)
                if self.request_times else 0.0
            )
            
            # Calculate error rate
            error_rate = (
                self.error_count / self.request_count
                if self.request_count > 0 else 0.0
            )
            
            # Calculate throughput (requests per second over last minute)
            now = time.time()
            recent_requests = sum(1 for _ in self.request_times)  # Simplified
            throughput = recent_requests / 60.0  # Approximate
        
        # Get system metrics
        cpu_usage = self.metrics.get_latest_value("system_cpu_usage_percent") or 0.0
        memory_usage_bytes = self.metrics.get_latest_value("system_memory_usage_bytes") or 0.0
        memory_usage_mb = memory_usage_bytes / (1024 * 1024)
        
        return PerformanceMetrics(
            response_time_ms=avg_response_time,
            throughput_rps=throughput,
            error_rate=error_rate,
            cpu_usage=cpu_usage,
            memory_usage_mb=memory_usage_mb,
            active_requests=0  # Would track in production
        )
    
    def check_performance_thresholds(self) -> List[Dict[str, Any]]:
        """Check if performance metrics exceed thresholds."""
        current_perf = self.get_current_performance()
        violations = []
        
        # Check response time
        if current_perf.response_time_ms > self.thresholds["response_time_ms"]:
            violations.append({
                "metric": "response_time_ms",
                "current": current_perf.response_time_ms,
                "threshold": self.thresholds["response_time_ms"],
                "severity": "high"
            })
        
        # Check error rate
        if current_perf.error_rate > self.thresholds["error_rate"]:
            violations.append({
                "metric": "error_rate",
                "current": current_perf.error_rate,
                "threshold": self.thresholds["error_rate"],
                "severity": "critical"
            })
        
        # Check CPU usage
        if current_perf.cpu_usage > self.thresholds["cpu_usage"]:
            violations.append({
                "metric": "cpu_usage",
                "current": current_perf.cpu_usage,
                "threshold": self.thresholds["cpu_usage"],
                "severity": "medium"
            })
        
        return violations
    
    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report for specified time period."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get request duration statistics
        request_durations = self.metrics.get_metric_values("http_request_duration_ms", since)
        duration_values = [p.value for p in request_durations]
        
        duration_stats = {}
        if duration_values:
            duration_stats = {
                "count": len(duration_values),
                "avg": sum(duration_values) / len(duration_values),
                "min": min(duration_values),
                "max": max(duration_values),
                "p95": sorted(duration_values)[int(len(duration_values) * 0.95)] if duration_values else 0
            }
        
        # Get document processing statistics
        doc_durations = self.metrics.get_metric_values("document_processing_duration_ms", since)
        doc_stats = {}
        if doc_durations:
            doc_values = [p.value for p in doc_durations]
            doc_stats = {
                "documents_processed": len(doc_values),
                "avg_processing_time": sum(doc_values) / len(doc_values),
                "fastest": min(doc_values),
                "slowest": max(doc_values)
            }
        
        return {
            "period_hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "request_statistics": duration_stats,
            "document_processing": doc_stats,
            "current_performance": self.get_current_performance().to_dict(),
            "threshold_violations": self.check_performance_thresholds()
        }


class ComplianceMetricsMonitor:
    """Monitor HIPAA compliance specific metrics."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize compliance metrics monitor.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics = metrics_collector
        self._lock = threading.Lock()
        
        # Compliance thresholds
        self.thresholds = {
            "max_phi_per_document": int(os.getenv("COMPLIANCE_MAX_PHI_PER_DOC", "50")),
            "min_compliance_score": float(os.getenv("COMPLIANCE_MIN_SCORE", "95.0")),
            "max_high_risk_documents": int(os.getenv("COMPLIANCE_MAX_HIGH_RISK", "5")),
            "max_violations_per_hour": int(os.getenv("COMPLIANCE_MAX_VIOLATIONS_HOURLY", "10"))
        }
    
    def record_document_compliance(self, document_id: str, phi_count: int, 
                                 compliance_score: float, risk_level: str,
                                 processing_duration_ms: float):
        """Record compliance metrics for a processed document."""
        labels = {"risk_level": risk_level}
        
        self.metrics.record_counter("documents_processed_total", 1.0, labels)
        self.metrics.record_histogram("phi_entities_per_document", phi_count, labels)
        self.metrics.record_histogram("compliance_score", compliance_score, labels)
        self.metrics.record_histogram("document_processing_duration_ms", processing_duration_ms, labels)
        
        # Track high-risk documents separately
        if risk_level in ["high", "critical"]:
            self.metrics.record_counter("high_risk_documents_total", 1.0)
    
    def record_compliance_violation(self, violation_type: str, document_id: str,
                                  severity: str, phi_types: List[str]):
        """Record a compliance violation event."""
        labels = {
            "violation_type": violation_type,
            "severity": severity
        }
        
        self.metrics.record_counter("compliance_violations_total", 1.0, labels)
        
        # Track specific PHI types involved in violations
        for phi_type in phi_types:
            phi_labels = {"phi_type": phi_type, "violation_type": violation_type}
            self.metrics.record_counter("phi_violations_by_type", 1.0, phi_labels)
    
    def record_audit_event(self, event_type: str, user_id: str, 
                          resource_type: str, action: str):
        """Record audit events for compliance tracking."""
        labels = {
            "event_type": event_type,
            "resource_type": resource_type,
            "action": action
        }
        
        self.metrics.record_counter("audit_events_total", 1.0, labels)
        
        # Track user activity
        user_labels = {"user_id": user_id, "action": action}
        self.metrics.record_counter("user_actions_total", 1.0, user_labels)
    
    def get_compliance_metrics(self, hours: int = 24) -> ComplianceMetrics:
        """Get current compliance metrics."""
        since = datetime.utcnow() - timedelta(hours=hours)
        
        # Get document processing counts
        doc_metrics = self.metrics.get_metric_values("documents_processed_total", since)
        documents_processed = len(doc_metrics)
        
        # Get PHI detection counts
        phi_metrics = self.metrics.get_metric_values("phi_entities_per_document", since)
        phi_entities_detected = sum(p.value for p in phi_metrics)
        
        # Get violation counts
        violation_metrics = self.metrics.get_metric_values("compliance_violations_total", since)
        compliance_violations = len(violation_metrics)
        
        # Calculate average compliance score
        score_metrics = self.metrics.get_metric_values("compliance_score", since)
        avg_compliance_score = (
            sum(p.value for p in score_metrics) / len(score_metrics)
            if score_metrics else 0.0
        )
        
        # Get high-risk document count
        high_risk_metrics = self.metrics.get_metric_values("high_risk_documents_total", since)
        high_risk_documents = len(high_risk_metrics)
        
        # Get audit event count
        audit_metrics = self.metrics.get_metric_values("audit_events_total", since)
        audit_events = len(audit_metrics)
        
        return ComplianceMetrics(
            documents_processed=documents_processed,
            phi_entities_detected=int(phi_entities_detected),
            compliance_violations=compliance_violations,
            avg_compliance_score=avg_compliance_score,
            high_risk_documents=high_risk_documents,
            audit_events=audit_events
        )
    
    def check_compliance_thresholds(self) -> List[Dict[str, Any]]:
        """Check compliance metrics against thresholds."""
        violations = []
        current_metrics = self.get_compliance_metrics(hours=1)  # Check last hour
        
        # Check average compliance score
        if current_metrics.avg_compliance_score < self.thresholds["min_compliance_score"]:
            violations.append({
                "metric": "compliance_score",
                "current": current_metrics.avg_compliance_score,
                "threshold": self.thresholds["min_compliance_score"],
                "severity": "high"
            })
        
        # Check high-risk documents
        if current_metrics.high_risk_documents > self.thresholds["max_high_risk_documents"]:
            violations.append({
                "metric": "high_risk_documents",
                "current": current_metrics.high_risk_documents,
                "threshold": self.thresholds["max_high_risk_documents"],
                "severity": "medium"
            })
        
        # Check violations per hour
        if current_metrics.compliance_violations > self.thresholds["max_violations_per_hour"]:
            violations.append({
                "metric": "violations_per_hour",
                "current": current_metrics.compliance_violations,
                "threshold": self.thresholds["max_violations_per_hour"],
                "severity": "critical"
            })
        
        return violations
    
    def generate_compliance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        current_metrics = self.get_compliance_metrics(hours)
        
        # Get PHI type breakdown
        phi_breakdown = {}
        since = datetime.utcnow() - timedelta(hours=hours)
        phi_violations = self.metrics.get_metric_values("phi_violations_by_type", since)
        
        for metric in phi_violations:
            phi_type = metric.labels.get("phi_type", "unknown")
            phi_breakdown[phi_type] = phi_breakdown.get(phi_type, 0) + 1
        
        # Get violation type breakdown
        violation_breakdown = {}
        violation_metrics = self.metrics.get_metric_values("compliance_violations_total", since)
        
        for metric in violation_metrics:
            violation_type = metric.labels.get("violation_type", "unknown")
            violation_breakdown[violation_type] = violation_breakdown.get(violation_type, 0) + 1
        
        return {
            "period_hours": hours,
            "generated_at": datetime.utcnow().isoformat(),
            "summary": current_metrics.to_dict(),
            "phi_type_breakdown": phi_breakdown,
            "violation_type_breakdown": violation_breakdown,
            "threshold_violations": self.check_compliance_thresholds(),
            "compliance_status": "COMPLIANT" if not self.check_compliance_thresholds() else "NON_COMPLIANT"
        }