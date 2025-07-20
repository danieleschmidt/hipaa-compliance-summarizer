"""Tests for structured logging framework with metrics support."""

import json
import logging
import time
from io import StringIO
from unittest.mock import patch, MagicMock
import pytest

from hipaa_compliance_summarizer.logging_framework import (
    StructuredLogger,
    LoggingConfig,
    MetricsCollector,
    StructuredFormatter,
    setup_structured_logging,
    get_logger_with_metrics
)


class TestLoggingConfig:
    """Test logging configuration management."""
    
    def test_default_config(self):
        """Test default logging configuration."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format_type == "structured"
        assert config.enable_metrics is True
        assert config.output_format == "json"
    
    def test_custom_config(self):
        """Test custom logging configuration."""
        config = LoggingConfig(
            level="DEBUG",
            format_type="simple",
            enable_metrics=False,
            output_format="text"
        )
        assert config.level == "DEBUG"
        assert config.format_type == "simple"
        assert config.enable_metrics is False
        assert config.output_format == "text"
    
    def test_config_from_env(self):
        """Test configuration loading from environment variables."""
        with patch.dict('os.environ', {
            'LOG_LEVEL': 'WARNING',
            'LOG_FORMAT_TYPE': 'structured',
            'ENABLE_METRICS': 'false',
            'LOG_OUTPUT_FORMAT': 'json'
        }):
            config = LoggingConfig.from_environment()
            assert config.level == "WARNING"
            assert config.format_type == "structured"
            assert config.enable_metrics is False
            assert config.output_format == "json"


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    def test_counter_increment(self):
        """Test counter metric incrementing."""
        collector = MetricsCollector()
        collector.increment_counter("test_counter", {"tag": "value"})
        collector.increment_counter("test_counter", {"tag": "value"})
        
        metrics = collector.get_metrics()
        assert "test_counter" in metrics["counters"]
        assert metrics["counters"]["test_counter"][0]["count"] == 2
        assert metrics["counters"]["test_counter"][0]["tags"] == {"tag": "value"}
    
    def test_timing_metric(self):
        """Test timing metric recording."""
        collector = MetricsCollector()
        
        with collector.time_operation("test_operation", {"module": "test"}):
            time.sleep(0.01)  # 10ms sleep
        
        metrics = collector.get_metrics()
        assert "test_operation" in metrics["timings"]
        assert len(metrics["timings"]["test_operation"]) == 1
        timing = metrics["timings"]["test_operation"][0]
        assert timing["duration_ms"] >= 10
        assert timing["tags"] == {"module": "test"}
    
    def test_gauge_metric(self):
        """Test gauge metric setting."""
        collector = MetricsCollector()
        collector.set_gauge("memory_usage", 1024, {"unit": "bytes"})
        
        metrics = collector.get_metrics()
        assert "memory_usage" in metrics["gauges"]
        assert metrics["gauges"]["memory_usage"]["value"] == 1024
        assert metrics["gauges"]["memory_usage"]["tags"] == {"unit": "bytes"}
    
    def test_histogram_metric(self):
        """Test histogram metric recording."""
        collector = MetricsCollector()
        values = [1, 2, 3, 4, 5]
        
        for value in values:
            collector.record_histogram("response_time", value, {"endpoint": "/api"})
        
        metrics = collector.get_metrics()
        assert "response_time" in metrics["histograms"]
        histogram = metrics["histograms"]["response_time"][0]
        assert histogram["count"] == 5
        assert histogram["sum"] == 15
        assert histogram["min"] == 1
        assert histogram["max"] == 5
        assert abs(histogram["avg"] - 3.0) < 0.001


class TestStructuredLogger:
    """Test structured logger functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = LoggingConfig(output_format="json")
        self.metrics_collector = MetricsCollector()
        self.stream = StringIO()
        self.logger = StructuredLogger("test_logger", self.config, self.metrics_collector)
        
        # Redirect logger output to our test stream while preserving formatter
        handler = logging.StreamHandler(self.stream)
        formatter = StructuredFormatter(self.config)
        handler.setFormatter(formatter)
        self.logger.logger.handlers = [handler]
        self.logger.logger.setLevel(logging.DEBUG)
    
    def test_info_logging_with_context(self):
        """Test info logging with context data."""
        self.logger.info("Test message", {
            "user_id": "user123",
            "operation": "phi_detection",
            "document_count": 5
        })
        
        log_output = self.stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["level"] == "INFO"
        assert log_data["message"] == "Test message"
        assert log_data["context"]["user_id"] == "user123"
        assert log_data["context"]["operation"] == "phi_detection"
        assert log_data["context"]["document_count"] == 5
        assert "timestamp" in log_data
        assert "logger_name" in log_data
    
    def test_error_logging_with_exception(self):
        """Test error logging with exception context."""
        try:
            raise ValueError("Test error")
        except ValueError as e:
            self.logger.error("Error occurred", {"operation": "test"}, exc_info=e)
        
        log_output = self.stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["level"] == "ERROR"
        assert log_data["message"] == "Error occurred"
        assert log_data["context"]["operation"] == "test"
        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test error"
    
    def test_performance_logging(self):
        """Test performance logging with automatic metrics."""
        with self.logger.log_performance("database_query", {"table": "users"}):
            time.sleep(0.01)  # Simulate work
        
        # Check log output
        log_output = self.stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["level"] == "INFO"
        assert "database_query completed" in log_data["message"]
        assert log_data["context"]["operation"] == "database_query"
        assert log_data["context"]["table"] == "users"
        assert log_data["context"]["duration_ms"] >= 10
        
        # Check metrics collection
        metrics = self.metrics_collector.get_metrics()
        assert "database_query" in metrics["timings"]
    
    def test_security_event_logging(self):
        """Test security event logging with special handling."""
        self.logger.log_security_event(
            "authentication_failure",
            severity="HIGH",
            context={
                "user_id": "user123",
                "ip_address": "192.168.1.1",
                "attempted_resource": "/admin"
            }
        )
        
        log_output = self.stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["level"] == "WARNING"
        assert log_data["event_type"] == "SECURITY"
        assert log_data["security_event"] == "authentication_failure"
        assert log_data["severity"] == "HIGH"
        assert log_data["context"]["user_id"] == "user123"
        
        # Verify security counter metric
        metrics = self.metrics_collector.get_metrics()
        assert "security_events" in metrics["counters"]
    
    def test_phi_detection_logging(self):
        """Test PHI detection specific logging."""
        self.logger.log_phi_detection(
            detected_types=["SSN", "EMAIL"],
            document_id="doc123",
            confidence_scores={"SSN": 0.95, "EMAIL": 0.87},
            redacted_count=5
        )
        
        log_output = self.stream.getvalue()
        log_data = json.loads(log_output.strip())
        
        assert log_data["level"] == "INFO"
        assert log_data["event_type"] == "PHI_DETECTION"
        assert log_data["context"]["detected_types"] == ["SSN", "EMAIL"]
        assert log_data["context"]["document_id"] == "doc123"
        assert log_data["context"]["redacted_count"] == 5
        
        # Verify PHI metrics
        metrics = self.metrics_collector.get_metrics()
        assert "phi_detected" in metrics["counters"]


class TestLoggingIntegration:
    """Test logging framework integration."""
    
    def test_setup_structured_logging(self):
        """Test setup of structured logging framework."""
        config = LoggingConfig(level="DEBUG")
        
        with patch('hipaa_compliance_summarizer.logging_framework.logging.getLogger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger
            
            setup_structured_logging(config)
            
            # Verify logger configuration was called
            mock_get_logger.assert_called()
    
    def test_get_logger_with_metrics(self):
        """Test getting a logger with metrics enabled."""
        logger = get_logger_with_metrics("test_module")
        
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'log_performance')
        assert hasattr(logger, 'log_security_event')
        assert hasattr(logger, 'log_phi_detection')
    
    def test_metrics_aggregation(self):
        """Test metrics aggregation across multiple loggers."""
        logger1 = get_logger_with_metrics("module1")
        logger2 = get_logger_with_metrics("module2")
        
        # Simulate some operations
        with patch.object(logger1, '_metrics_collector') as mock_metrics1:
            with patch.object(logger2, '_metrics_collector') as mock_metrics2:
                logger1.info("Test from module1", {"module": "1"})
                logger2.info("Test from module2", {"module": "2"})
                
                # Verify metrics were called
                mock_metrics1.increment_counter.assert_called()
                mock_metrics2.increment_counter.assert_called()


class TestLoggingFormatters:
    """Test different logging output formatters."""
    
    def test_json_formatter(self):
        """Test JSON output formatting."""
        config = LoggingConfig(output_format="json")
        logger = StructuredLogger("test", config, MetricsCollector())
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = StructuredFormatter(config)
        handler.setFormatter(formatter)
        logger.logger.handlers = [handler]
        
        logger.info("Test message", {"key": "value"})
        
        output = stream.getvalue().strip()
        parsed = json.loads(output)
        
        assert parsed["message"] == "Test message"
        assert parsed["context"]["key"] == "value"
    
    def test_text_formatter(self):
        """Test text output formatting."""
        config = LoggingConfig(output_format="text", format_type="simple")
        logger = StructuredLogger("test", config, MetricsCollector())
        
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        formatter = StructuredFormatter(config)
        handler.setFormatter(formatter)
        logger.logger.handlers = [handler]
        
        logger.info("Test message", {"key": "value"})
        
        output = stream.getvalue().strip()
        
        # Should contain the message and context in readable format
        assert "Test message" in output
        assert "key=value" in output


class TestLoggingPerformance:
    """Test logging framework performance characteristics."""
    
    def test_logging_overhead(self):
        """Test that logging overhead is minimal."""
        config = LoggingConfig(enable_metrics=False)
        logger = StructuredLogger("perf_test", config, None)
        
        # Disable actual output
        logger.logger.handlers = []
        
        start_time = time.time()
        
        # Log many messages
        for i in range(1000):
            logger.info(f"Message {i}", {"iteration": i})
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 1000 log calls in under 100ms
        assert duration < 0.1
    
    def test_metrics_performance(self):
        """Test metrics collection performance."""
        collector = MetricsCollector()
        
        start_time = time.time()
        
        # Record many metrics
        for i in range(1000):
            collector.increment_counter("test_counter", {"iteration": str(i % 10)})
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 1000 metric operations in under 50ms
        assert duration < 0.05