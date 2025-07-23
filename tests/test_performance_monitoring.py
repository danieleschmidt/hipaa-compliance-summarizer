"""
Tests for the performance monitoring dashboard system.

This module tests comprehensive monitoring capabilities including:
- Processing time metrics
- Memory usage tracking  
- Error rate monitoring
- Pattern performance analytics
- Real-time dashboard updates
"""

import pytest
import time
from unittest.mock import Mock, patch
from pathlib import Path
import tempfile
import json
from threading import Event

from hipaa_compliance_summarizer.monitoring import (
    PerformanceMonitor,
    MonitoringDashboard,
    MetricType,
    ProcessingMetrics,
    PatternMetrics,
    SystemMetrics
)
from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.processor import ComplianceLevel


class TestProcessingMetrics:
    """Test the ProcessingMetrics dataclass."""
    
    def test_processing_metrics_creation(self):
        """Test creating processing metrics."""
        from hipaa_compliance_summarizer.constants import TEST_CONSTANTS
        metrics = ProcessingMetrics(
            total_documents=TEST_CONSTANTS.TEST_DOCS_PROCESSED,
            successful_documents=95,
            failed_documents=5,
            avg_processing_time=2.5,
            total_processing_time=TEST_CONSTANTS.TEST_PROCESSING_TIME,
            avg_compliance_score=0.92,
            total_phi_detected=TEST_CONSTANTS.TEST_PHI_DETECTED
        )
        
        assert metrics.total_documents == TEST_CONSTANTS.TEST_DOCS_PROCESSED
        assert metrics.successful_documents == 95
        assert metrics.failed_documents == 5
        assert metrics.avg_processing_time == 2.5
        assert metrics.success_rate == 0.95
        assert metrics.error_rate == 0.05
    
    def test_processing_metrics_calculations(self):
        """Test calculated properties of processing metrics."""
        metrics = ProcessingMetrics(
            total_documents=50,
            successful_documents=48,
            failed_documents=2,
            avg_processing_time=1.8,
            total_processing_time=86.4,
            avg_compliance_score=0.96,
            total_phi_detected=750
        )
        
        assert metrics.success_rate == 0.96
        assert metrics.error_rate == 0.04
        assert abs(metrics.avg_phi_per_document - 15.0) < 0.01


class TestPatternMetrics:
    """Test the PatternMetrics dataclass."""
    
    def test_pattern_metrics_creation(self):
        """Test creating pattern metrics."""
        metrics = PatternMetrics(
            pattern_name="ssn",
            total_matches=45,
            avg_match_time=0.02,
            cache_hit_ratio=0.85,
            confidence_scores=[0.98, 0.96, 0.99, 0.97]
        )
        
        assert metrics.pattern_name == "ssn"
        assert metrics.total_matches == 45
        assert metrics.avg_match_time == 0.02
        assert metrics.cache_hit_ratio == 0.85
        assert abs(metrics.avg_confidence - 0.975) < 0.001


class TestSystemMetrics:
    """Test the SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test creating system metrics."""
        metrics = SystemMetrics(
            cpu_usage=65.5,
            memory_usage=1024.0,
            memory_peak=1536.0,
            disk_io_read=2048.0,
            disk_io_write=512.0,
            cache_size=128.0,
            cache_hit_ratio=0.78
        )
        
        assert metrics.cpu_usage == 65.5
        assert metrics.memory_usage == 1024.0
        assert metrics.memory_peak == 1536.0
        assert metrics.cache_hit_ratio == 0.78


class TestPerformanceMonitor:
    """Test the PerformanceMonitor class."""
    
    def setup_method(self):
        """Set up for each test."""
        self.monitor = PerformanceMonitor()
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization."""
        assert self.monitor.start_time is not None
        assert len(self.monitor.processing_times) == 0
        assert len(self.monitor.compliance_scores) == 0
        assert self.monitor.total_documents == 0
    
    def test_start_document_processing(self):
        """Test starting document processing measurement."""
        doc_id = "test_doc_001"
        self.monitor.start_document_processing(doc_id)
        
        assert doc_id in self.monitor._active_processing
        assert self.monitor._active_processing[doc_id] > 0
    
    def test_end_document_processing_success(self):
        """Test ending document processing measurement with success."""
        doc_id = "test_doc_001"
        self.monitor.start_document_processing(doc_id)
        
        # Simulate some processing time
        time.sleep(0.01)
        
        result = Mock()
        result.compliance_score = 0.95
        result.phi_detected_count = 8
        
        self.monitor.end_document_processing(doc_id, success=True, result=result)
        
        assert doc_id not in self.monitor._active_processing
        assert len(self.monitor.processing_times) == 1
        assert self.monitor.processing_times[0] > 0.01
        assert len(self.monitor.compliance_scores) == 1
        assert self.monitor.compliance_scores[0] == 0.95
        assert self.monitor.successful_documents == 1
        assert self.monitor.failed_documents == 0
        assert self.monitor.total_phi_detected == 8
    
    def test_end_document_processing_failure(self):
        """Test ending document processing measurement with failure."""
        doc_id = "test_doc_002"
        self.monitor.start_document_processing(doc_id)
        
        time.sleep(0.01)
        
        self.monitor.end_document_processing(doc_id, success=False, error="Parse error")
        
        assert doc_id not in self.monitor._active_processing
        assert self.monitor.successful_documents == 0
        assert self.monitor.failed_documents == 1
        assert len(self.monitor.errors) == 1
        assert "Parse error" in self.monitor.errors[0]
    
    def test_record_pattern_performance(self):
        """Test recording pattern performance metrics."""
        self.monitor.record_pattern_performance("ssn", 0.025, True, 0.98)
        self.monitor.record_pattern_performance("email", 0.018, False, 0.96)
        
        assert "ssn" in self.monitor.pattern_metrics
        assert "email" in self.monitor.pattern_metrics
        
        ssn_metrics = self.monitor.pattern_metrics["ssn"]
        assert len(ssn_metrics["match_times"]) == 1
        assert ssn_metrics["cache_hits"] == 1
        assert ssn_metrics["cache_misses"] == 0
    
    def test_get_processing_metrics(self):
        """Test getting comprehensive processing metrics."""
        # Simulate some processing
        for i in range(10):
            doc_id = f"doc_{i}"
            self.monitor.start_document_processing(doc_id)
            time.sleep(0.001)  # Small delay
            
            if i < 9:  # 9 successful, 1 failure
                result = Mock()
                result.compliance_score = 0.9 + (i * 0.01)
                result.phi_detected_count = i + 1
                self.monitor.end_document_processing(doc_id, success=True, result=result)
            else:
                self.monitor.end_document_processing(doc_id, success=False, error="Test error")
        
        metrics = self.monitor.get_processing_metrics()
        
        assert metrics.total_documents == 10
        assert metrics.successful_documents == 9
        assert metrics.failed_documents == 1
        assert abs(metrics.success_rate - 0.9) < 0.001
        assert metrics.avg_processing_time > 0
        assert metrics.total_phi_detected == sum(range(1, 10))  # 1+2+...+9
    
    def test_get_pattern_metrics(self):
        """Test getting pattern performance metrics."""
        # Record some pattern performance data
        patterns = ["ssn", "email", "phone"]
        for pattern in patterns:
            for i in range(5):
                self.monitor.record_pattern_performance(
                    pattern, 
                    0.02 + (i * 0.005), 
                    i % 2 == 0,  # Alternating cache hits/misses
                    0.95 + (i * 0.01)
                )
        
        pattern_metrics = self.monitor.get_pattern_metrics()
        
        assert len(pattern_metrics) == 3
        for pattern_name, metrics in pattern_metrics.items():
            assert pattern_name in patterns
            assert metrics.total_matches == 5
            assert metrics.cache_hit_ratio == 0.6  # 3 hits out of 5
            assert metrics.avg_match_time > 0.02
    
    def test_system_metrics_collection(self):
        """Test system resource metrics collection."""
        with patch('psutil.cpu_percent', return_value=45.5), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters') as mock_disk:
            
            mock_memory.return_value = Mock(
                used=1073741824,  # 1GB
                percent=60.0
            )
            mock_disk.return_value = Mock(
                read_bytes=2048,
                write_bytes=1024
            )
            
            metrics = self.monitor.get_system_metrics()
            
            assert metrics.cpu_usage == 45.5
            assert metrics.memory_usage == 1024.0  # MB
            assert metrics.disk_io_read == 2048
            assert metrics.disk_io_write == 1024
    
    def test_reset_metrics(self):
        """Test resetting all metrics."""
        # Add some data
        self.monitor.start_document_processing("test_doc")
        self.monitor.end_document_processing("test_doc", success=True, 
                                           result=Mock(compliance_score=0.95, phi_detected_count=5))
        self.monitor.record_pattern_performance("ssn", 0.02, True, 0.98)
        
        assert self.monitor.total_documents > 0
        assert len(self.monitor.processing_times) > 0
        assert len(self.monitor.pattern_metrics) > 0
        
        # Reset
        self.monitor.reset()
        
        assert self.monitor.total_documents == 0
        assert len(self.monitor.processing_times) == 0
        assert len(self.monitor.compliance_scores) == 0
        assert len(self.monitor.pattern_metrics) == 0
        assert len(self.monitor.errors) == 0


class TestMonitoringDashboard:
    """Test the MonitoringDashboard class."""
    
    def setup_method(self):
        """Set up for each test."""
        self.monitor = PerformanceMonitor()
        self.dashboard = MonitoringDashboard(self.monitor)
    
    def test_dashboard_initialization(self):
        """Test monitoring dashboard initialization."""
        assert self.dashboard.monitor is self.monitor
        assert self.dashboard.update_interval == 1.0
        assert not self.dashboard._running
    
    def test_generate_dashboard_data(self):
        """Test generating comprehensive dashboard data."""
        # Add some test data
        for i in range(5):
            doc_id = f"doc_{i}"
            self.monitor.start_document_processing(doc_id)
            time.sleep(0.001)
            
            result = Mock()
            result.compliance_score = 0.9 + (i * 0.02)
            result.phi_detected_count = i + 1
            self.monitor.end_document_processing(doc_id, success=True, result=result)
        
        self.monitor.record_pattern_performance("ssn", 0.025, True, 0.98)
        self.monitor.record_pattern_performance("email", 0.020, False, 0.95)
        
        dashboard_data = self.dashboard.generate_dashboard_data()
        
        assert "processing_metrics" in dashboard_data
        assert "pattern_metrics" in dashboard_data
        assert "system_metrics" in dashboard_data
        assert "timestamp" in dashboard_data
        
        processing = dashboard_data["processing_metrics"]
        assert processing["total_documents"] == 5
        assert processing["successful_documents"] == 5
        assert processing["success_rate"] == 1.0
    
    def test_save_dashboard_json(self):
        """Test saving dashboard data to JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            file_path = f.name
        
        try:
            # Add some test data
            doc_id = "test_doc"
            self.monitor.start_document_processing(doc_id)
            result = Mock()
            result.compliance_score = 0.95
            result.phi_detected_count = 3
            self.monitor.end_document_processing(doc_id, success=True, result=result)
            
            # Save dashboard
            self.dashboard.save_dashboard_json(file_path)
            
            # Verify file exists and contains valid JSON
            file_path_obj = Path(file_path)
            assert file_path_obj.exists()
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            assert "processing_metrics" in data
            assert data["processing_metrics"]["total_documents"] == 1
            
        finally:
            Path(file_path).unlink(missing_ok=True)
    
    def test_real_time_monitoring(self):
        """Test real-time monitoring capabilities."""
        callbacks_received = []
        
        def test_callback(data):
            callbacks_received.append(data)
        
        # Start monitoring with short interval
        self.dashboard.start_real_time_monitoring(
            callback=test_callback,
            interval=0.1
        )
        
        # Add some data
        doc_id = "test_doc"
        self.monitor.start_document_processing(doc_id)
        result = Mock()
        result.compliance_score = 0.95
        result.phi_detected_count = 3
        self.monitor.end_document_processing(doc_id, success=True, result=result)
        
        # Wait for callback
        time.sleep(0.15)
        
        # Stop monitoring
        self.dashboard.stop_real_time_monitoring()
        
        # Verify callback was called
        assert len(callbacks_received) > 0
        assert "processing_metrics" in callbacks_received[0]
    
    def test_performance_alerts(self):
        """Test performance alerting system."""
        alerts = []
        
        def alert_callback(alert_type, message, data):
            alerts.append({
                "type": alert_type,
                "message": message,
                "data": data
            })
        
        self.dashboard.set_alert_thresholds(
            max_processing_time=0.5,
            min_success_rate=0.9,
            max_error_rate=0.1
        )
        self.dashboard.set_alert_callback(alert_callback)
        
        # Simulate slow processing
        doc_id = "slow_doc"
        self.monitor.start_document_processing(doc_id)
        time.sleep(0.6)  # Exceed threshold
        result = Mock()
        result.compliance_score = 0.95
        result.phi_detected_count = 3
        self.monitor.end_document_processing(doc_id, success=True, result=result)
        
        # Generate dashboard data to trigger alerts
        self.dashboard.generate_dashboard_data()
        
        # Check for alerts
        assert len(alerts) > 0
        assert any(alert["type"] == "SLOW_PROCESSING" for alert in alerts)


class TestMonitoringIntegration:
    """Integration tests for monitoring with batch processing."""
    
    def test_batch_processor_with_monitoring(self, tmp_path):
        """Test batch processor integration with performance monitoring."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = tmp_path / f"test_doc_{i}.txt"
            test_file.write_text(f"Patient record {i}: SSN 123-45-678{i}, Email: patient{i}@hospital.com")
            test_files.append(test_file)
        
        # Create monitored batch processor
        monitor = PerformanceMonitor()
        processor = BatchProcessor()
        
        # Process files with monitoring
        results = []
        for test_file in test_files:
            doc_id = f"doc_{test_file.name}"
            monitor.start_document_processing(doc_id)
            
            try:
                # Simulate processing
                result = Mock()
                result.compliance_score = 0.95
                result.phi_detected_count = 2
                results.append(result)
                monitor.end_document_processing(doc_id, success=True, result=result)
            except Exception as e:
                monitor.end_document_processing(doc_id, success=False, error=str(e))
        
        # Check monitoring results
        metrics = monitor.get_processing_metrics()
        assert metrics.total_documents == 3
        assert metrics.successful_documents == 3
        assert metrics.avg_compliance_score == 0.95
    
    def test_pattern_performance_monitoring(self):
        """Test monitoring of PHI pattern performance."""
        monitor = PerformanceMonitor()
        
        # Simulate pattern matching with performance data
        patterns = ["ssn", "email", "phone", "date"]
        for pattern in patterns:
            for i in range(10):
                # Vary performance characteristics
                match_time = 0.01 + (i * 0.002)
                cache_hit = i % 3 != 0  # 2/3 cache hit rate
                confidence = 0.9 + (i * 0.01)
                
                monitor.record_pattern_performance(pattern, match_time, cache_hit, confidence)
        
        pattern_metrics = monitor.get_pattern_metrics()
        
        assert len(pattern_metrics) == 4
        for pattern_name, metrics in pattern_metrics.items():
            assert metrics.total_matches == 10
            assert 0.6 < metrics.cache_hit_ratio < 0.8
            assert metrics.avg_match_time > 0.01
            assert metrics.avg_confidence > 0.9