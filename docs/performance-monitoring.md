# Performance Monitoring Guide

## Overview

The HIPAA Compliance Summarizer includes a comprehensive performance monitoring system that provides real-time insights into processing performance, PHI pattern effectiveness, system resource usage, and error tracking.

## Features

### Core Monitoring Capabilities

- **Processing Performance**: Track document processing times, throughput, success rates
- **Pattern Analytics**: Monitor PHI pattern matching performance and cache effectiveness
- **System Resources**: CPU, memory, disk I/O monitoring (requires psutil)
- **Error Tracking**: Comprehensive error categorization and reporting
- **Real-time Dashboard**: Live performance metrics with alerting
- **Historical Data**: Export capabilities for trend analysis

### Key Metrics Tracked

#### Processing Metrics
- Total documents processed
- Success/failure rates
- Average processing time per document
- Throughput (documents per minute)
- Compliance scores distribution
- PHI detection counts

#### Pattern Metrics
- Individual pattern match times
- Cache hit/miss ratios
- Confidence score distributions
- Pattern usage frequency

#### System Metrics
- CPU utilization
- Memory usage and peak consumption
- Disk I/O operations
- Cache performance

## Basic Usage

### Setting up Monitoring

```python
from hipaa_compliance_summarizer.monitoring import PerformanceMonitor, MonitoringDashboard
from hipaa_compliance_summarizer.batch import BatchProcessor
from hipaa_compliance_summarizer.phi import PHIRedactor

# Create performance monitor
monitor = PerformanceMonitor()

# Create batch processor with monitoring
processor = BatchProcessor(
    compliance_level="standard",
    performance_monitor=monitor
)

# Process documents - monitoring happens automatically
results = processor.process_directory(
    "path/to/documents",
    output_dir="path/to/output"
)

# Generate comprehensive dashboard
dashboard_data = processor.generate_performance_dashboard()
print(f"Processed {dashboard_data['processing_metrics']['total_documents']} documents")
print(f"Average processing time: {dashboard_data['processing_metrics']['avg_processing_time']:.2f}s")
```

### Individual Component Monitoring

```python
# Monitor PHI redactor directly
monitor = PerformanceMonitor()
redactor = PHIRedactor(performance_monitor=monitor)

# Process text with automatic pattern performance tracking
text = "Patient SSN: 123-45-6789, Email: patient@hospital.com"
entities = redactor.detect(text)

# Get pattern performance metrics
pattern_metrics = monitor.get_pattern_metrics()
for pattern_name, metrics in pattern_metrics.items():
    print(f"{pattern_name}: {metrics.avg_match_time:.4f}s avg, "
          f"{metrics.cache_hit_ratio:.2%} cache hit rate")
```

## Advanced Features

### Real-time Monitoring

```python
from hipaa_compliance_summarizer.monitoring import MonitoringDashboard

monitor = PerformanceMonitor()
dashboard = MonitoringDashboard(monitor)

def monitoring_callback(dashboard_data):
    """Called periodically with updated metrics."""
    processing = dashboard_data["processing_metrics"]
    print(f"Active processing: {processing['active_processing_count']} documents")
    print(f"Throughput: {processing['throughput_docs_per_minute']:.1f} docs/min")

# Start real-time monitoring (updates every second)
dashboard.start_real_time_monitoring(
    callback=monitoring_callback,
    interval=1.0
)

# ... do processing work ...

# Stop monitoring
dashboard.stop_real_time_monitoring()
```

### Performance Alerts

```python
def alert_handler(alert_type, message, data):
    """Handle performance alerts."""
    print(f"ALERT [{alert_type}]: {message}")
    if alert_type == "SLOW_PROCESSING":
        print(f"Processing time: {data['avg_processing_time']:.2f}s")
    elif alert_type == "HIGH_ERROR_RATE":
        print(f"Error rate: {data['error_rate']:.2%}")

# Configure alerts
dashboard.set_alert_callback(alert_handler)
dashboard.set_alert_thresholds(
    max_processing_time=5.0,      # Alert if avg > 5 seconds
    min_success_rate=0.95,        # Alert if success rate < 95%
    max_error_rate=0.05,          # Alert if error rate > 5%
    max_memory_usage=1024.0,      # Alert if memory > 1GB
    min_cache_hit_ratio=0.8       # Alert if cache hit rate < 80%
)
```

### Data Export and Analysis

```python
# Export dashboard data to JSON
processor.save_performance_dashboard("performance_report.json")

# Export historical data for analysis
dashboard.export_historical_data("historical_data.json")

# Generate custom reports
dashboard_data = dashboard.generate_dashboard_data()

# Analysis example
processing = dashboard_data["processing_metrics"]
patterns = dashboard_data["pattern_metrics"]

print("=== Performance Report ===")
print(f"Documents processed: {processing['total_documents']}")
print(f"Success rate: {processing['success_rate']:.2%}")
print(f"Average processing time: {processing['avg_processing_time']:.2f}s")
print(f"Throughput: {processing['throughput_docs_per_minute']:.1f} docs/min")

print("\n=== Pattern Performance ===")
for name, metrics in patterns.items():
    print(f"{name}:")
    print(f"  Matches: {metrics['total_matches']}")
    print(f"  Avg time: {metrics['avg_match_time']:.4f}s")
    print(f"  Cache hit rate: {metrics['cache_hit_ratio']:.2%}")
    print(f"  Avg confidence: {metrics['avg_confidence']:.2f}")
```

## Dashboard Data Structure

The monitoring system generates comprehensive dashboard data with the following structure:

```json
{
  "timestamp": 1234567890.123,
  "processing_metrics": {
    "total_documents": 100,
    "successful_documents": 95,
    "failed_documents": 5,
    "success_rate": 0.95,
    "error_rate": 0.05,
    "avg_processing_time": 2.34,
    "total_processing_time": 234.0,
    "avg_compliance_score": 0.92,
    "total_phi_detected": 1450,
    "avg_phi_per_document": 15.26,
    "throughput_docs_per_minute": 25.6,
    "active_processing_count": 3
  },
  "pattern_metrics": {
    "ssn": {
      "total_matches": 45,
      "avg_match_time": 0.023,
      "cache_hit_ratio": 0.87,
      "avg_confidence": 0.96,
      "confidence_std": 0.02
    },
    "email": {
      "total_matches": 78,
      "avg_match_time": 0.019,
      "cache_hit_ratio": 0.92,
      "avg_confidence": 0.94,
      "confidence_std": 0.03
    }
  },
  "system_metrics": {
    "cpu_usage": 45.5,
    "memory_usage": 512.3,
    "memory_peak": 768.1,
    "disk_io_read": 2048,
    "disk_io_write": 1024,
    "cache_size": 64.2,
    "cache_hit_ratio": 0.85
  },
  "error_summary": {
    "total_errors": 5,
    "recent_errors": [
      "doc1.txt: File read error",
      "doc2.txt: Processing timeout"
    ]
  }
}
```

## Performance Optimization

### Interpreting Metrics

#### Processing Performance
- **Target processing time**: < 2 seconds per document for typical medical records
- **Target success rate**: > 95% for production workloads
- **Target throughput**: > 30 documents/minute for small files

#### Pattern Performance
- **Target cache hit ratio**: > 80% for repeated pattern usage
- **Target match time**: < 0.05 seconds per pattern
- **Target confidence**: > 0.90 for critical patterns (SSN, MRN)

#### System Resources
- **Memory usage**: Monitor for memory leaks in long-running processes
- **CPU usage**: Should stay below 80% for sustainable processing
- **Cache efficiency**: Cache hit ratio should improve over time

### Optimization Strategies

1. **Pattern Optimization**
   - Disable unused patterns to improve performance
   - Optimize regex patterns for better matching speed
   - Monitor cache effectiveness

2. **Resource Management**
   - Adjust worker thread counts based on system metrics
   - Monitor memory usage for large batch operations
   - Clear caches periodically if memory is constrained

3. **Error Reduction**
   - Monitor error patterns to identify problematic document types
   - Implement retry logic for transient errors
   - Validate input files before processing

## Integration Examples

### With Existing Batch Processing

```python
# Enhance existing batch processing with monitoring
monitor = PerformanceMonitor()
processor = BatchProcessor(performance_monitor=monitor)

# Process with monitoring
results = processor.process_directory("documents/")

# Generate reports
regular_dashboard = processor.generate_dashboard(results)
performance_dashboard = processor.generate_performance_dashboard()

# Compare metrics
print("Basic Dashboard:")
print(f"  Documents: {regular_dashboard.documents_processed}")
print(f"  Avg Score: {regular_dashboard.avg_compliance_score}")

print("Performance Dashboard:")
perf_processing = performance_dashboard["processing_metrics"]
print(f"  Success Rate: {perf_processing['success_rate']:.2%}")
print(f"  Avg Time: {perf_processing['avg_processing_time']:.2f}s")
```

### Custom Monitoring Integration

```python
# Custom processing loop with monitoring
monitor = PerformanceMonitor()
redactor = PHIRedactor(performance_monitor=monitor)

for document_path in document_list:
    doc_id = str(document_path)
    monitor.start_document_processing(doc_id)
    
    try:
        # Custom processing logic
        with open(document_path) as f:
            text = f.read()
        
        entities = redactor.detect(text)
        result = redactor.redact(text)
        
        # Create mock result for monitoring
        mock_result = type('Result', (), {
            'compliance_score': calculate_compliance_score(entities),
            'phi_detected_count': len(entities)
        })()
        
        monitor.end_document_processing(doc_id, success=True, result=mock_result)
        
    except Exception as e:
        monitor.end_document_processing(doc_id, success=False, error=str(e))

# Generate final report
metrics = monitor.get_processing_metrics()
print(f"Processed {metrics.total_documents} documents")
print(f"Success rate: {metrics.success_rate:.2%}")
```

## Troubleshooting

### Common Issues

**High Processing Times**
- Check system resource usage
- Verify pattern complexity
- Monitor cache hit ratios
- Consider reducing worker thread count

**Low Cache Hit Ratios**
- Ensure patterns are stable across processing sessions
- Check for dynamic pattern generation
- Monitor cache size limits

**Memory Usage Growth**
- Check for cache size limits
- Monitor for memory leaks in long-running processes
- Clear caches periodically: `PHIRedactor.clear_cache()`

**Missing System Metrics**
- Install psutil: `pip install psutil`
- Check system permissions for resource monitoring
- Verify compatible psutil version

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed monitoring logs
monitor = PerformanceMonitor()
# Debug logs will show detailed performance tracking
```

## API Reference

### PerformanceMonitor
- `start_document_processing(document_id)`: Begin tracking document
- `end_document_processing(document_id, success, result, error)`: Complete tracking
- `record_pattern_performance(pattern_name, time, cache_hit, confidence)`: Record pattern metrics
- `get_processing_metrics()`: Get ProcessingMetrics object
- `get_pattern_metrics()`: Get pattern performance data
- `get_system_metrics()`: Get system resource usage
- `reset()`: Clear all monitoring data

### MonitoringDashboard
- `generate_dashboard_data()`: Create comprehensive dashboard
- `save_dashboard_json(file_path)`: Export dashboard to JSON
- `start_real_time_monitoring(callback, interval)`: Begin live monitoring
- `stop_real_time_monitoring()`: End live monitoring
- `set_alert_callback(callback)`: Configure alerting
- `set_alert_thresholds(**kwargs)`: Set alert thresholds

### Integration Methods
- `BatchProcessor(performance_monitor=monitor)`: Enable batch monitoring
- `PHIRedactor(performance_monitor=monitor)`: Enable pattern monitoring
- `processor.generate_performance_dashboard()`: Get enhanced dashboard
- `processor.save_performance_dashboard(path)`: Export performance data