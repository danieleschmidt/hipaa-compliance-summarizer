"""
Performance Monitoring and Dashboard System

This module provides comprehensive monitoring capabilities for HIPAA compliance
processing, including real-time performance metrics, pattern analysis, system
resource monitoring, and alerting capabilities.
"""

from __future__ import annotations

import time
import threading
import logging
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from enum import Enum
import statistics

from .constants import BYTES_PER_MB

logger = logging.getLogger(__name__)

# Optional imports for system monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    logger.warning("psutil not available - system metrics will be limited")


class MetricType(str, Enum):
    """Types of metrics that can be monitored."""
    
    PROCESSING_TIME = "processing_time"
    COMPLIANCE_SCORE = "compliance_score"
    PHI_DETECTION = "phi_detection"
    PATTERN_PERFORMANCE = "pattern_performance"
    SYSTEM_RESOURCES = "system_resources"
    ERROR_RATE = "error_rate"


@dataclass
class ProcessingMetrics:
    """Comprehensive processing performance metrics."""
    
    total_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    avg_compliance_score: float = 0.0
    total_phi_detected: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return self.successful_documents / self.total_documents
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as a percentage."""
        if self.total_documents == 0:
            return 0.0
        return self.failed_documents / self.total_documents
    
    @property
    def avg_phi_per_document(self) -> float:
        """Calculate average PHI entities per document."""
        if self.successful_documents == 0:
            return 0.0
        return self.total_phi_detected / self.successful_documents
    
    @property
    def throughput_docs_per_minute(self) -> float:
        """Calculate processing throughput in documents per minute."""
        if self.total_processing_time == 0:
            return 0.0
        return (self.total_documents / self.total_processing_time) * 60


@dataclass
class PatternMetrics:
    """Metrics for individual PHI pattern performance."""
    
    pattern_name: str
    total_matches: int = 0
    avg_match_time: float = 0.0
    cache_hit_ratio: float = 0.0
    confidence_scores: List[float] = field(default_factory=list)
    
    @property
    def avg_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return statistics.mean(self.confidence_scores)
    
    @property
    def confidence_std(self) -> float:
        """Calculate standard deviation of confidence scores."""
        if len(self.confidence_scores) < 2:
            return 0.0
        return statistics.stdev(self.confidence_scores)


@dataclass
class SystemMetrics:
    """System resource utilization metrics."""
    
    cpu_usage: float = 0.0
    memory_usage: float = 0.0  # MB
    memory_peak: float = 0.0   # MB
    disk_io_read: float = 0.0  # bytes
    disk_io_write: float = 0.0 # bytes
    cache_size: float = 0.0    # MB
    cache_hit_ratio: float = 0.0


class PerformanceMonitor:
    """Central performance monitoring system for HIPAA processing."""
    
    def __init__(self):
        self.start_time = time.time()
        self.processing_times: List[float] = []
        self.compliance_scores: List[float] = []
        self.phi_counts: List[int] = []
        self.errors: List[str] = []
        
        # Document processing tracking
        self._active_processing: Dict[str, float] = {}
        self.successful_documents = 0
        self.failed_documents = 0
        self.total_phi_detected = 0
        
        # Pattern performance tracking
        self.pattern_metrics: Dict[str, Dict] = {}
        
        # System monitoring
        self._baseline_disk_io = None
        self._peak_memory = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Performance monitor initialized")
    
    @property
    def total_documents(self) -> int:
        """Get total number of documents processed."""
        return self.successful_documents + self.failed_documents
    
    def start_document_processing(self, document_id: str) -> None:
        """Start timing document processing."""
        with self._lock:
            self._active_processing[document_id] = time.time()
            logger.debug(f"Started monitoring document: {document_id}")
    
    def end_document_processing(
        self, 
        document_id: str, 
        success: bool, 
        result: Optional[Any] = None,
        error: Optional[str] = None
    ) -> None:
        """End timing document processing and record results."""
        with self._lock:
            if document_id not in self._active_processing:
                logger.warning(f"Document {document_id} was not being monitored")
                return
            
            start_time = self._active_processing.pop(document_id)
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            if success and result:
                self.successful_documents += 1
                if hasattr(result, 'compliance_score'):
                    self.compliance_scores.append(result.compliance_score)
                if hasattr(result, 'phi_detected_count'):
                    phi_count = result.phi_detected_count
                    self.phi_counts.append(phi_count)
                    self.total_phi_detected += phi_count
                
                logger.debug(f"Successfully processed {document_id} in {processing_time:.3f}s")
            else:
                self.failed_documents += 1
                if error:
                    self.errors.append(f"{document_id}: {error}")
                
                logger.debug(f"Failed to process {document_id} in {processing_time:.3f}s: {error}")
    
    def record_pattern_performance(
        self, 
        pattern_name: str, 
        match_time: float,
        cache_hit: bool, 
        confidence: float
    ) -> None:
        """Record performance metrics for a specific pattern."""
        with self._lock:
            if pattern_name not in self.pattern_metrics:
                self.pattern_metrics[pattern_name] = {
                    "match_times": [],
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "confidences": []
                }
            
            metrics = self.pattern_metrics[pattern_name]
            metrics["match_times"].append(match_time)
            metrics["confidences"].append(confidence)
            
            if cache_hit:
                metrics["cache_hits"] += 1
            else:
                metrics["cache_misses"] += 1
            
            logger.debug(f"Recorded pattern performance for {pattern_name}: "
                        f"time={match_time:.4f}s, cache_hit={cache_hit}, confidence={confidence}")
    
    def get_processing_metrics(self) -> ProcessingMetrics:
        """Get comprehensive processing metrics."""
        with self._lock:
            avg_processing_time = 0.0
            total_processing_time = 0.0
            avg_compliance_score = 0.0
            
            if self.processing_times:
                avg_processing_time = statistics.mean(self.processing_times)
                total_processing_time = sum(self.processing_times)
            
            if self.compliance_scores:
                avg_compliance_score = statistics.mean(self.compliance_scores)
            
            return ProcessingMetrics(
                total_documents=self.total_documents,
                successful_documents=self.successful_documents,
                failed_documents=self.failed_documents,
                avg_processing_time=avg_processing_time,
                total_processing_time=total_processing_time,
                avg_compliance_score=avg_compliance_score,
                total_phi_detected=self.total_phi_detected
            )
    
    def get_pattern_metrics(self) -> Dict[str, PatternMetrics]:
        """Get performance metrics for all patterns."""
        with self._lock:
            pattern_metrics = {}
            
            for pattern_name, metrics in self.pattern_metrics.items():
                match_times = metrics["match_times"]
                cache_hits = metrics["cache_hits"]
                cache_misses = metrics["cache_misses"]
                confidences = metrics["confidences"]
                
                total_requests = cache_hits + cache_misses
                cache_hit_ratio = cache_hits / total_requests if total_requests > 0 else 0.0
                avg_match_time = statistics.mean(match_times) if match_times else 0.0
                
                pattern_metrics[pattern_name] = PatternMetrics(
                    pattern_name=pattern_name,
                    total_matches=len(match_times),
                    avg_match_time=avg_match_time,
                    cache_hit_ratio=cache_hit_ratio,
                    confidence_scores=confidences.copy()
                )
            
            return pattern_metrics
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system resource metrics."""
        metrics = SystemMetrics()
        
        if not HAS_PSUTIL:
            logger.debug("psutil not available - returning empty system metrics")
            return metrics
        
        try:
            # CPU usage
            metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            metrics.memory_usage = memory.used / BYTES_PER_MB  # Convert to MB
            if metrics.memory_usage > self._peak_memory:
                self._peak_memory = metrics.memory_usage
            metrics.memory_peak = self._peak_memory
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                if self._baseline_disk_io is None:
                    self._baseline_disk_io = (disk_io.read_bytes, disk_io.write_bytes)
                
                baseline_read, baseline_write = self._baseline_disk_io
                metrics.disk_io_read = disk_io.read_bytes - baseline_read
                metrics.disk_io_write = disk_io.write_bytes - baseline_write
            
            # Cache metrics (from pattern system if available)
            from .phi import PHIRedactor
            cache_info = PHIRedactor.get_cache_info()
            
            pattern_cache = cache_info.get("pattern_compilation")
            phi_cache = cache_info.get("phi_detection")
            
            if pattern_cache and phi_cache:
                total_hits = pattern_cache.hits + phi_cache.hits
                total_requests = (pattern_cache.hits + pattern_cache.misses + 
                                phi_cache.hits + phi_cache.misses)
                metrics.cache_hit_ratio = total_hits / total_requests if total_requests > 0 else 0.0
                
                # Rough estimate of cache size (number of entries * estimated size per entry)
                metrics.cache_size = (pattern_cache.currsize + phi_cache.currsize) * 0.001  # Rough MB estimate
            
        except Exception as e:
            logger.warning(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def reset(self) -> None:
        """Reset all monitoring data."""
        with self._lock:
            self.start_time = time.time()
            self.processing_times.clear()
            self.compliance_scores.clear()
            self.phi_counts.clear()
            self.errors.clear()
            
            self._active_processing.clear()
            self.successful_documents = 0
            self.failed_documents = 0
            self.total_phi_detected = 0
            
            self.pattern_metrics.clear()
            
            self._baseline_disk_io = None
            self._peak_memory = 0.0
            
            logger.info("Performance monitor reset")
    
    def get_active_processing_count(self) -> int:
        """Get the number of documents currently being processed."""
        with self._lock:
            return len(self._active_processing)


class MonitoringDashboard:
    """Real-time monitoring dashboard for HIPAA processing performance."""
    
    def __init__(self, monitor: PerformanceMonitor, update_interval: float = 1.0):
        self.monitor = monitor
        self.update_interval = update_interval
        self._running = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Alert system
        self._alert_callback: Optional[Callable] = None
        self._alert_thresholds = {
            "max_processing_time": 10.0,  # seconds
            "min_success_rate": 0.95,
            "max_error_rate": 0.05,
            "max_memory_usage": 2048.0,  # MB
            "min_cache_hit_ratio": 0.8
        }
        
        logger.info("Monitoring dashboard initialized")
    
    def set_alert_callback(self, callback: Callable[[str, str, Dict], None]) -> None:
        """Set callback function for performance alerts."""
        self._alert_callback = callback
        logger.info("Alert callback configured")
    
    def set_alert_thresholds(self, **thresholds) -> None:
        """Update alert thresholds."""
        self._alert_thresholds.update(thresholds)
        logger.info(f"Alert thresholds updated: {thresholds}")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data."""
        processing_metrics = self.monitor.get_processing_metrics()
        pattern_metrics = self.monitor.get_pattern_metrics()
        system_metrics = self.monitor.get_system_metrics()
        
        # Convert metrics to dictionary format
        dashboard_data = {
            "timestamp": time.time(),
            "processing_metrics": {
                "total_documents": processing_metrics.total_documents,
                "successful_documents": processing_metrics.successful_documents,
                "failed_documents": processing_metrics.failed_documents,
                "success_rate": processing_metrics.success_rate,
                "error_rate": processing_metrics.error_rate,
                "avg_processing_time": processing_metrics.avg_processing_time,
                "total_processing_time": processing_metrics.total_processing_time,
                "avg_compliance_score": processing_metrics.avg_compliance_score,
                "total_phi_detected": processing_metrics.total_phi_detected,
                "avg_phi_per_document": processing_metrics.avg_phi_per_document,
                "throughput_docs_per_minute": processing_metrics.throughput_docs_per_minute,
                "active_processing_count": self.monitor.get_active_processing_count()
            },
            "pattern_metrics": {
                name: {
                    "total_matches": metrics.total_matches,
                    "avg_match_time": metrics.avg_match_time,
                    "cache_hit_ratio": metrics.cache_hit_ratio,
                    "avg_confidence": metrics.avg_confidence,
                    "confidence_std": metrics.confidence_std
                }
                for name, metrics in pattern_metrics.items()
            },
            "system_metrics": {
                "cpu_usage": system_metrics.cpu_usage,
                "memory_usage": system_metrics.memory_usage,
                "memory_peak": system_metrics.memory_peak,
                "disk_io_read": system_metrics.disk_io_read,
                "disk_io_write": system_metrics.disk_io_write,
                "cache_size": system_metrics.cache_size,
                "cache_hit_ratio": system_metrics.cache_hit_ratio
            },
            "error_summary": {
                "total_errors": len(self.monitor.errors),
                "recent_errors": self.monitor.errors[-5:] if self.monitor.errors else []
            }
        }
        
        # Check for alerts
        self._check_alerts(dashboard_data)
        
        return dashboard_data
    
    def _check_alerts(self, dashboard_data: Dict[str, Any]) -> None:
        """Check dashboard data against alert thresholds."""
        if not self._alert_callback:
            return
        
        processing = dashboard_data["processing_metrics"]
        system = dashboard_data["system_metrics"]
        
        # Check processing time threshold
        if processing["avg_processing_time"] > self._alert_thresholds["max_processing_time"]:
            self._alert_callback(
                "SLOW_PROCESSING",
                f"Average processing time ({processing['avg_processing_time']:.2f}s) exceeds threshold",
                {"avg_processing_time": processing["avg_processing_time"]}
            )
        
        # Check success rate threshold
        if processing["success_rate"] < self._alert_thresholds["min_success_rate"]:
            self._alert_callback(
                "LOW_SUCCESS_RATE", 
                f"Success rate ({processing['success_rate']:.2%}) below threshold",
                {"success_rate": processing["success_rate"]}
            )
        
        # Check error rate threshold
        if processing["error_rate"] > self._alert_thresholds["max_error_rate"]:
            self._alert_callback(
                "HIGH_ERROR_RATE",
                f"Error rate ({processing['error_rate']:.2%}) exceeds threshold", 
                {"error_rate": processing["error_rate"]}
            )
        
        # Check memory usage threshold
        if system["memory_usage"] > self._alert_thresholds["max_memory_usage"]:
            self._alert_callback(
                "HIGH_MEMORY_USAGE",
                f"Memory usage ({system['memory_usage']:.1f}MB) exceeds threshold",
                {"memory_usage": system["memory_usage"]}
            )
        
        # Check cache hit ratio threshold
        if system["cache_hit_ratio"] < self._alert_thresholds["min_cache_hit_ratio"]:
            self._alert_callback(
                "LOW_CACHE_HIT_RATIO",
                f"Cache hit ratio ({system['cache_hit_ratio']:.2%}) below threshold",
                {"cache_hit_ratio": system["cache_hit_ratio"]}
            )
    
    def save_dashboard_json(self, file_path: str) -> None:
        """Save current dashboard data to JSON file."""
        dashboard_data = self.generate_dashboard_data()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"Dashboard data saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save dashboard data to {file_path}: {e}")
            raise
    
    def start_real_time_monitoring(
        self, 
        callback: Callable[[Dict[str, Any]], None],
        interval: Optional[float] = None
    ) -> None:
        """Start real-time monitoring with callback updates."""
        if self._running:
            logger.warning("Real-time monitoring is already running")
            return
        
        self._running = True
        self._stop_event.clear()
        update_interval = interval or self.update_interval
        
        def monitoring_loop():
            while not self._stop_event.wait(update_interval):
                try:
                    dashboard_data = self.generate_dashboard_data()
                    callback(dashboard_data)
                except Exception as e:
                    logger.error(f"Error in monitoring callback: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        
        logger.info(f"Started real-time monitoring with {update_interval}s interval")
    
    def stop_real_time_monitoring(self) -> None:
        """Stop real-time monitoring."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)
        
        logger.info("Stopped real-time monitoring")
    
    def export_historical_data(self, file_path: str, format: str = "json") -> None:
        """Export historical monitoring data."""
        data = {
            "export_timestamp": time.time(),
            "monitor_start_time": self.monitor.start_time,
            "processing_times": self.monitor.processing_times,
            "compliance_scores": self.monitor.compliance_scores,
            "phi_counts": self.monitor.phi_counts,
            "errors": self.monitor.errors,
            "pattern_metrics": {
                name: {
                    "match_times": metrics["match_times"],
                    "cache_hits": metrics["cache_hits"],
                    "cache_misses": metrics["cache_misses"],
                    "confidences": metrics["confidences"]
                }
                for name, metrics in self.monitor.pattern_metrics.items()
            }
        }
        
        try:
            if format.lower() == "json":
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Historical data exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export historical data: {e}")
            raise