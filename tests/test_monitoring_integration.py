"""
Integration tests for the monitoring system with batch processing.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock

from hipaa_compliance_summarizer.monitoring import PerformanceMonitor, MonitoringDashboard
from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.phi import PHIRedactor
from hipaa_compliance_summarizer.processor import ComplianceLevel


class TestMonitoringIntegration:
    """Integration tests for monitoring with actual processing."""
    
    def test_batch_processor_with_monitoring(self, tmp_path):
        """Test batch processor with performance monitoring enabled."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = tmp_path / f"test_doc_{i}.txt"
            content = f"""
            Patient Record {i}
            SSN: 123-45-678{i}
            Email: patient{i}@hospital.com
            Phone: 555-123-456{i}
            Date of Birth: 01/0{i+1}/1980
            """
            test_file.write_text(content)
            test_files.append(test_file)
        
        # Set up monitoring
        monitor = PerformanceMonitor()
        
        # Create batch processor with monitoring
        processor = BatchProcessor(
            compliance_level=ComplianceLevel.STANDARD,
            performance_monitor=monitor
        )
        
        # Process files
        results = processor.process_directory(
            str(tmp_path),
            show_progress=False
        )
        
        # Verify processing results
        assert len(results) == 3
        successful_results = [r for r in results if hasattr(r, 'compliance_score')]
        assert len(successful_results) >= 1  # At least some should succeed
        
        # Verify monitoring captured data
        metrics = monitor.get_processing_metrics()
        assert metrics.total_documents >= 1
        assert metrics.successful_documents >= 1
        assert metrics.avg_processing_time > 0
        
        # Check pattern metrics
        pattern_metrics = monitor.get_pattern_metrics()
        assert len(pattern_metrics) > 0  # Should have captured some pattern performance
        
        # Generate performance dashboard
        dashboard_data = processor.generate_performance_dashboard()
        assert dashboard_data is not None
        assert "processing_metrics" in dashboard_data
        assert "pattern_metrics" in dashboard_data
        assert "system_metrics" in dashboard_data
    
    def test_phi_redactor_with_monitoring(self):
        """Test PHI redactor with performance monitoring."""
        monitor = PerformanceMonitor()
        
        # Create PHI redactor with monitoring
        redactor = PHIRedactor(performance_monitor=monitor)
        
        test_text = """
        Patient John Doe
        SSN: 123-45-6789
        Email: john.doe@hospital.com
        Phone: 555-123-4567
        """
        
        # Process text multiple times to test caching
        for i in range(3):
            entities = redactor.detect(test_text)
            assert len(entities) >= 3  # Should detect SSN, email, phone
        
        # Check pattern metrics were recorded
        pattern_metrics = monitor.get_pattern_metrics()
        assert len(pattern_metrics) > 0
        
        # Verify cache hit ratios improve
        for pattern_name, metrics in pattern_metrics.items():
            assert metrics.total_matches > 0
            # After multiple identical detections, cache hit ratio should be high
            if metrics.total_matches > 1:
                assert metrics.cache_hit_ratio > 0
    
    def test_monitoring_dashboard_generation(self, tmp_path):
        """Test comprehensive dashboard generation and export."""
        monitor = PerformanceMonitor()
        dashboard = MonitoringDashboard(monitor)
        
        # Create some test processing activity
        for i in range(5):
            doc_id = f"test_doc_{i}"
            monitor.start_document_processing(doc_id)
            
            # Simulate some pattern performance
            monitor.record_pattern_performance("ssn", 0.02, i % 2 == 0, 0.95)
            monitor.record_pattern_performance("email", 0.015, i % 3 == 0, 0.98)
            
            # Complete processing
            result = Mock()
            result.compliance_score = 0.95
            result.phi_detected_count = 2
            monitor.end_document_processing(doc_id, success=True, result=result)
        
        # Generate dashboard data
        dashboard_data = dashboard.generate_dashboard_data()
        
        # Verify comprehensive data structure
        assert "timestamp" in dashboard_data
        assert "processing_metrics" in dashboard_data
        assert "pattern_metrics" in dashboard_data
        assert "system_metrics" in dashboard_data
        assert "error_summary" in dashboard_data
        
        processing = dashboard_data["processing_metrics"]
        assert processing["total_documents"] == 5
        assert processing["successful_documents"] == 5
        assert processing["success_rate"] == 1.0
        assert processing["avg_processing_time"] > 0
        
        patterns = dashboard_data["pattern_metrics"]
        assert "ssn" in patterns
        assert "email" in patterns
        
        # Test JSON export
        json_file = tmp_path / "dashboard.json"
        dashboard.save_dashboard_json(str(json_file))
        
        assert json_file.exists()
        with open(json_file) as f:
            exported_data = json.load(f)
        
        assert "processing_metrics" in exported_data
        assert exported_data["processing_metrics"]["total_documents"] == 5
    
    def test_performance_alerts(self):
        """Test performance alerting system."""
        monitor = PerformanceMonitor()
        dashboard = MonitoringDashboard(monitor)
        
        alerts_triggered = []
        
        def alert_callback(alert_type, message, data):
            alerts_triggered.append({
                "type": alert_type,
                "message": message,
                "data": data
            })
        
        # Set up alerts with low thresholds for testing
        dashboard.set_alert_callback(alert_callback)
        dashboard.set_alert_thresholds(
            max_processing_time=0.001,  # Very low threshold
            min_success_rate=0.99,      # Very high threshold
            max_error_rate=0.01         # Very low threshold
        )
        
        # Create scenario that should trigger alerts
        doc_id = "slow_doc"
        monitor.start_document_processing(doc_id)
        
        # Simulate slow processing
        import time
        time.sleep(0.01)  # Exceed the 0.001s threshold
        
        result = Mock()
        result.compliance_score = 0.95
        result.phi_detected_count = 1
        monitor.end_document_processing(doc_id, success=True, result=result)
        
        # Generate dashboard data to trigger alert checks
        dashboard.generate_dashboard_data()
        
        # Verify alert was triggered
        assert len(alerts_triggered) > 0
        slow_processing_alerts = [a for a in alerts_triggered if a["type"] == "SLOW_PROCESSING"]
        assert len(slow_processing_alerts) > 0
    
    def test_real_time_monitoring(self):
        """Test real-time monitoring capabilities."""
        monitor = PerformanceMonitor()
        dashboard = MonitoringDashboard(monitor)
        
        callback_data = []
        
        def monitoring_callback(data):
            callback_data.append(data)
        
        # Start real-time monitoring with short interval
        dashboard.start_real_time_monitoring(
            callback=monitoring_callback,
            interval=0.05  # 50ms interval
        )
        
        # Simulate some processing activity
        doc_id = "test_doc"
        monitor.start_document_processing(doc_id)
        result = Mock()
        result.compliance_score = 0.95
        result.phi_detected_count = 2
        monitor.end_document_processing(doc_id, success=True, result=result)
        
        # Wait for monitoring callback
        import time
        time.sleep(0.1)
        
        # Stop monitoring
        dashboard.stop_real_time_monitoring()
        
        # Verify callbacks were received
        assert len(callback_data) > 0
        assert "processing_metrics" in callback_data[0]
    
    def test_error_handling_in_monitoring(self, tmp_path):
        """Test monitoring system handles errors gracefully."""
        monitor = PerformanceMonitor()
        processor = BatchProcessor(performance_monitor=monitor)
        
        # Create a file that will cause processing errors
        error_file = tmp_path / "corrupt_file.txt"
        error_file.write_bytes(b'\x00\x01\x02\x03')  # Binary data that might cause issues
        
        # Process the problematic file
        results = processor.process_directory(str(tmp_path))
        
        # Verify error was handled
        assert len(results) == 1
        
        # Check monitoring captured the error
        metrics = monitor.get_processing_metrics()
        assert metrics.total_documents == 1
        # The file might succeed or fail depending on processing robustness
        assert metrics.total_documents == metrics.successful_documents + metrics.failed_documents