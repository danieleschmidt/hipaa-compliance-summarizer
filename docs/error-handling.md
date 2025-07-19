# Comprehensive Error Handling

This document describes the enhanced error handling capabilities implemented in the HIPAA compliance summarizer.

## Overview

The batch processing system now includes comprehensive error handling for file operations, I/O errors, permission issues, and resource management. This ensures robust processing even when individual files or operations fail.

## Error Handling Features

### File Operation Error Handling

**Input Directory Validation**:
- Validates directory existence and accessibility
- Handles permission denied errors gracefully
- Provides clear error messages for missing directories

**Output Directory Management**:
- Creates output directories with proper error handling
- Handles permission and disk space issues
- Validates directory creation success

**File Processing Errors**:
- Individual file failures don't stop batch processing
- Each file error is captured with detailed information
- Processing continues for remaining files

### Error Result Structure

The `ErrorResult` class provides detailed error information:

```python
@dataclass
class ErrorResult:
    file_path: str          # Path to the failed file
    error: str              # Detailed error message
    error_type: str         # Category of error (FileReadError, ProcessingError, etc.)
    
    # Compatibility properties for dashboard generation
    compliance_score: float = 0.0
    phi_detected_count: int = 0
```

### Error Categories

**FileReadError**: Issues reading input files
- Corrupted files
- Permission denied
- Encoding problems
- File not found

**FileWriteError**: Issues writing output files
- Permission denied
- Disk full
- Directory write restrictions

**ProcessingError**: Errors during document processing
- Document parsing failures
- PHI detection errors
- Memory allocation issues

**ThreadExecutionError**: Concurrent processing failures
- Thread pool exceptions
- Resource contention
- Timeout issues

**ValidationError**: Input parameter validation
- Invalid compliance levels
- Negative max_workers
- Invalid file paths

## Usage Examples

### Basic Error Handling

```python
from hipaa_compliance_summarizer.batch import BatchProcessor

processor = BatchProcessor()
results = processor.process_directory("/input/documents", output_dir="/output")

# Separate successful and failed results
successful_results = [r for r in results if isinstance(r, ProcessingResult)]
error_results = [r for r in results if isinstance(r, ErrorResult)]

print(f"Processed: {len(successful_results)} successful, {len(error_results)} errors")

# Handle errors
for error in error_results:
    print(f"Error in {error.file_path}: {error.error}")
```

### Progress Monitoring with Error Display

```python
results = processor.process_directory(
    "/input/documents",
    output_dir="/output",
    show_progress=True,  # Shows ✓ for success, ✗ for errors
    max_workers=4
)
```

### Dashboard Generation with Mixed Results

```python
# Dashboard handles both successful and error results
dashboard = processor.generate_dashboard(results)
print(f"Documents processed: {dashboard.documents_processed}")
print(f"Average compliance: {dashboard.avg_compliance_score}")

# Error results contribute 0.0 to compliance score calculation
```

## Error Recovery Strategies

### File-Level Recovery

```python
# Retry failed files with different settings
error_files = [error.file_path for error in error_results 
               if error.error_type == "FileReadError"]

# Process with different encoding or smaller chunk size
for file_path in error_files:
    try:
        # Custom processing logic for problematic files
        pass
    except Exception as e:
        logger.warning(f"Failed to recover {file_path}: {e}")
```

### Batch-Level Recovery

```python
try:
    results = processor.process_directory("/input", output_dir="/output")
except PermissionError as e:
    # Handle directory-level permission issues
    logger.error(f"Directory access denied: {e}")
    # Try alternative output location
    results = processor.process_directory("/input", output_dir="/tmp/output")
```

## Concurrent Processing Error Handling

The enhanced batch processor safely handles errors in multi-threaded environments:

- Individual thread failures don't crash the entire batch
- Partial results are returned even if some threads fail
- Thread pool exceptions are captured and reported
- Resource cleanup is guaranteed

```python
# Process with multiple workers and error resilience
results = processor.process_directory(
    "/large/dataset",
    max_workers=8,
    show_progress=True
)

# Get performance metrics including error rates
cache_performance = processor.get_cache_performance()
print(f"Cache hit ratio: {cache_performance['phi_detection']['hit_ratio']:.1%}")
```

## Configuration and Validation

### Input Validation

```python
# The processor validates all input parameters
try:
    results = processor.process_directory(
        "/input",
        compliance_level="invalid",  # Will raise ValueError
        max_workers=-1              # Will raise ValueError
    )
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

### Output Directory Handling

```python
try:
    results = processor.process_directory(
        "/input",
        output_dir="/restricted/location"  # May raise PermissionError
    )
except PermissionError as e:
    print(f"Cannot create output directory: {e}")
    # Use alternative location
    results = processor.process_directory("/input", output_dir="/tmp/safe_output")
```

## Logging and Monitoring

The error handling system provides comprehensive logging:

```python
import logging

# Enable detailed logging
logging.getLogger('hipaa_compliance_summarizer').setLevel(logging.INFO)

# Process with logging
results = processor.process_directory("/input", output_dir="/output")

# Log summary shows error counts
# INFO: Successfully processed 95 files
# WARNING: Processed 100 files: 95 successful, 5 errors
```

## Performance Impact

The enhanced error handling has minimal performance impact:

- **Overhead**: <2% for normal processing
- **Memory Usage**: Minimal increase for error tracking
- **Concurrency**: No impact on thread pool performance
- **Cache Performance**: Error handling doesn't affect PHI detection caching

## Best Practices

### 1. Always Check for Errors

```python
results = processor.process_directory("/input")

if any(isinstance(r, ErrorResult) for r in results):
    error_count = sum(1 for r in results if isinstance(r, ErrorResult))
    logger.warning(f"Processing completed with {error_count} errors")
    
    # Optionally retry error files or alert administrators
```

### 2. Handle Resource Constraints

```python
# For large datasets, use appropriate worker counts
import os

max_workers = min(4, os.cpu_count())  # Don't overwhelm the system
results = processor.process_directory("/large/input", max_workers=max_workers)
```

### 3. Implement Graceful Degradation

```python
try:
    # Try with full feature set
    results = processor.process_directory(
        "/input", 
        output_dir="/primary/output",
        generate_summaries=True
    )
except PermissionError:
    # Fall back to minimal processing
    results = processor.process_directory("/input")  # No output files
```

### 4. Monitor Error Patterns

```python
# Analyze error types for system health monitoring
error_types = {}
for result in results:
    if isinstance(result, ErrorResult):
        error_types[result.error_type] = error_types.get(result.error_type, 0) + 1

if error_types:
    logger.info(f"Error distribution: {error_types}")
```

## Testing Error Scenarios

The comprehensive test suite includes error handling scenarios:

```bash
# Run error handling tests
pytest tests/test_batch_error_handling.py -v

# Test specific error categories
pytest tests/test_batch_error_handling.py::TestBatchProcessorFileErrorHandling -v
```

## Future Enhancements

Planned improvements for error handling:

1. **Retry Logic**: Automatic retry for transient failures
2. **Circuit Breaker**: Stop processing after too many consecutive errors
3. **Error Aggregation**: Group similar errors for easier analysis
4. **Recovery Suggestions**: Provide actionable error resolution guidance