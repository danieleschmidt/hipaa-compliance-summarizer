# Performance Tests

This directory contains performance and load testing for the HIPAA Compliance Summarizer.

## Test Categories

### 1. Unit Performance Tests
- Individual function performance benchmarks
- Memory usage profiling
- Algorithm complexity validation
- PHI pattern matching performance

### 2. Integration Performance Tests
- End-to-end document processing performance
- Batch processing throughput
- Database query performance
- API response times

### 3. Load Testing
- High-volume document processing
- Concurrent user simulation
- System resource utilization
- Scalability limits

### 4. Memory Testing
- Memory leak detection
- Large file processing
- Cache efficiency
- Garbage collection impact

## Performance Targets

| Metric | Target | Measurement |
|--------|---------|-------------|
| Document Processing | <15 seconds | Average for clinical notes |
| PHI Detection | <2 seconds | Per 1000 words |
| Batch Throughput | >300 docs/hour | Standard compliance level |
| Memory Usage | <2GB | Per worker process |
| API Response | <500ms | 95th percentile |

## Test Data

Performance tests use:
- Synthetic documents of various sizes (1KB - 10MB)
- Realistic PHI density distributions
- Multiple document formats (PDF, DOCX, TXT)
- Various complexity levels

## Running Performance Tests

```bash
# Run all performance tests
pytest tests/performance/ -v

# Run with performance profiling
pytest tests/performance/ --profile

# Generate performance report
pytest tests/performance/ --benchmark-only --benchmark-html=reports/performance.html

# Memory profiling
pytest tests/performance/ --memory-profile

# Load testing
pytest tests/performance/test_load.py -v --workers=10
```

## Continuous Performance Monitoring

Performance tests are integrated into CI/CD:
- Regression detection on every PR
- Performance trending over time
- Automated alerts for degradation
- Resource usage monitoring

## Test Environment

Performance tests require:
- Consistent hardware configuration
- Isolated test environment
- Reproducible test conditions
- Baseline performance metrics

## Reporting

Performance results include:
- Response time percentiles
- Throughput measurements
- Resource utilization
- Trend analysis
- Regression detection